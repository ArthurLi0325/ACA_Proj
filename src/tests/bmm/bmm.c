// BATCH_MATMUL (one attention head) with zero points on both operands, see bmm.h.
//
// Zero points cannot be folded offline as in FC (D2), because both operands are activations.
// Expanding (a - za)(b - zb) gives a raw signed int8 dot product plus per-row and per-column
// corrections (DESIGN.md D4): rowc[m] = -zb*sum(A[m]) + K*za*zb, colc[n] = -za*sum(B[n]), each
// computed once per head, so the inner loop is the same as FC and the accumulators start at
// rowc[m] + colc[n] instead of the bias. Same 2x4 tiling as B1/X1. Variants:
//   default      qmac8 (sums too, against a vector of ones) + rq.a
//   SCALAR       B1-style scalar code: byte loads, mul + add, software requantization
//   TILE4        (Xqmac only) 4x4 tiles with Kp a compile-time constant (64 or 200), the software
//                option A of the X2 study (see fc_x1/fc.c)
#include "bmm.h"
#ifndef SCALAR
#include "xqmac.h"
#endif

#define MAXDIM 200

typedef uint64_t __attribute__((may_alias)) u64a;

#ifdef SCALAR
__attribute__((noinline))
static int32_t srdhm(int32_t a, int32_t b)
{
  if (a == INT32_MIN && b == INT32_MIN)
    return INT32_MAX;
  int64_t ab = (int64_t)a * b;
  int64_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
  return (int32_t)((ab + nudge) / ((int64_t)1 << 31));
}

// TFLite BATCH_MATMUL: SRDHM, then gemmlowp RoundingDivideByPOT (ties away from zero).
__attribute__((noinline))
static int32_t requant_a(int32_t acc, int32_t m0, int shift)
{
  int left = shift > 0 ? shift : 0;
  int right = shift > 0 ? 0 : -shift;
  int32_t v = srdhm(acc * (1 << left), m0);
  int32_t mask = (1 << right) - 1;
  int32_t threshold = (mask >> 1) + (v < 0 ? 1 : 0);
  return (v >> right) + ((v & mask) > threshold ? 1 : 0);
}

static inline int8_t store_q(int32_t acc, int32_t m0, int shift, int32_t zo, uint64_t rqp)
{
  (void)rqp;
  int32_t y = requant_a(acc, m0, shift) + zo;
  if (y < -128) y = -128;
  if (y > 127) y = 127;
  return (int8_t)y;
}

static int32_t row_sum(const int8_t *a, int Kp)
{
  int32_t s = 0;
  for (int k = 0; k < Kp; k++)
    s += a[k];
  return s;
}

// 2 rows x 4 columns, K unrolled 4x (Kp is a multiple of 8).
__attribute__((noinline, optimize("no-schedule-insns")))
static void tile_2x4(const int8_t *a0, const int8_t *a1, const int8_t *b0, int Kp,
                     int32_t r0, int32_t r1, const int32_t *colc, int32_t c[8])
{
  const int8_t *b1 = b0 + Kp, *b2 = b1 + Kp, *b3 = b2 + Kp;
  int32_t c00 = r0 + colc[0], c01 = r0 + colc[1], c02 = r0 + colc[2], c03 = r0 + colc[3];
  int32_t c10 = r1 + colc[0], c11 = r1 + colc[1], c12 = r1 + colc[2], c13 = r1 + colc[3];
  for (const int8_t *end = a0 + Kp; a0 != end; a0 += 4, a1 += 4, b0 += 4, b1 += 4, b2 += 4, b3 += 4) {
#define STEP(j) {                                                   \
      int32_t x0 = a0[j], x1 = a1[j];                               \
      int32_t y0 = b0[j], y1 = b1[j], y2 = b2[j], y3 = b3[j];       \
      c00 += x0 * y0; c01 += x0 * y1; c02 += x0 * y2; c03 += x0 * y3; \
      c10 += x1 * y0; c11 += x1 * y1; c12 += x1 * y2; c13 += x1 * y3; }
    STEP(0) STEP(1) STEP(2) STEP(3)
#undef STEP
  }
  c[0] = c00; c[1] = c01; c[2] = c02; c[3] = c03;
  c[4] = c10; c[5] = c11; c[6] = c12; c[7] = c13;
}

// one output (odd last row and the N % 4 tail columns)
__attribute__((noinline))
static int32_t dot1(const int8_t *a, const int8_t *b, int Kp, int32_t init)
{
  int32_t c = init;
  for (int k = 0; k < Kp; k++)
    c += a[k] * b[k];
  return c;
}

#else  // Xqmac

static inline int8_t store_q(int32_t acc, int32_t m0, int shift, int32_t zo, uint64_t rqp)
{
  (void)m0; (void)shift; (void)zo;
  return (int8_t)rq_a(acc, rqp);
}

static int32_t row_sum(const int8_t *a, int Kp)
{
  const u64a *x = (const u64a *)a;
  int64_t s = 0;
  for (int k = 0; k < Kp / 8; k++)
    s = qmac8(s, x[k], 0x0101010101010101ull);        // 8 bytes summed per instruction
  return (int32_t)s;
}

#ifndef TILE4
// 2 rows x 4 columns, 16 K-steps per iteration plus one 8-step tail when Kp / 8 is odd.
__attribute__((noinline))
static void tile_2x4(const int8_t *a0, const int8_t *a1, const int8_t *b0, int Kp,
                     int32_t r0, int32_t r1, const int32_t *colc, int32_t c[8])
{
  const u64a *x0 = (const u64a *)a0, *x1 = (const u64a *)a1;
  const u64a *y0 = (const u64a *)b0, *y1 = (const u64a *)(b0 + Kp);
  const u64a *y2 = (const u64a *)(b0 + 2 * Kp), *y3 = (const u64a *)(b0 + 3 * Kp);
  int64_t c00 = r0 + colc[0], c01 = r0 + colc[1], c02 = r0 + colc[2], c03 = r0 + colc[3];
  int64_t c10 = r1 + colc[0], c11 = r1 + colc[1], c12 = r1 + colc[2], c13 = r1 + colc[3];
#define STEP(j) {                                                   \
      uint64_t u0 = x0[j], u1 = x1[j];                              \
      uint64_t v0 = y0[j], v1 = y1[j], v2 = y2[j], v3 = y3[j];      \
      c00 = qmac8(c00, u0, v0); c01 = qmac8(c01, u0, v1);           \
      c02 = qmac8(c02, u0, v2); c03 = qmac8(c03, u0, v3);           \
      c10 = qmac8(c10, u1, v0); c11 = qmac8(c11, u1, v1);           \
      c12 = qmac8(c12, u1, v2); c13 = qmac8(c13, u1, v3); }
  for (const u64a *end = x0 + (Kp / 16) * 2; x0 != end; x0 += 2, x1 += 2, y0 += 2, y1 += 2, y2 += 2, y3 += 2) {
    STEP(0) STEP(1)
  }
  if (Kp & 8)
    STEP(0)
#undef STEP
  c[0] = c00; c[1] = c01; c[2] = c02; c[3] = c03;
  c[4] = c10; c[5] = c11; c[6] = c12; c[7] = c13;
}
#endif

__attribute__((noinline))
static int32_t dot1(const int8_t *a, const int8_t *b, int Kp, int32_t init)
{
  const u64a *x = (const u64a *)a, *y = (const u64a *)b;
  int64_t c = init;
  for (int k = 0; k < Kp / 8; k++)
    c = qmac8(c, x[k], y[k]);
  return (int32_t)c;
}
#endif

#if defined(TILE4) && !defined(SCALAR)
static inline __attribute__((always_inline))
void tile_4x4_k(const int8_t *a, const int8_t *b, const int Kp, const int32_t *rowc,
                const int32_t *colc, int32_t c[16])
{
  int64_t c00 = rowc[0] + colc[0], c01 = rowc[0] + colc[1], c02 = rowc[0] + colc[2], c03 = rowc[0] + colc[3];
  int64_t c10 = rowc[1] + colc[0], c11 = rowc[1] + colc[1], c12 = rowc[1] + colc[2], c13 = rowc[1] + colc[3];
  int64_t c20 = rowc[2] + colc[0], c21 = rowc[2] + colc[1], c22 = rowc[2] + colc[2], c23 = rowc[2] + colc[3];
  int64_t c30 = rowc[3] + colc[0], c31 = rowc[3] + colc[1], c32 = rowc[3] + colc[2], c33 = rowc[3] + colc[3];
#define LD(p, off) (*(const u64a *)((p) + (off)))
#define COL(n, j) { uint64_t v = LD(b, (n) * Kp + (j));                                    \
      c0##n = qmac8(c0##n, u0, v); c1##n = qmac8(c1##n, u1, v);                         \
      c2##n = qmac8(c2##n, u2, v); c3##n = qmac8(c3##n, u3, v); }
#define STEP(j) { uint64_t u0 = LD(a, j), u1 = LD(a, Kp + (j)), u2 = LD(a, 2 * Kp + (j)),     \
                           u3 = LD(a, 3 * Kp + (j));                                      \
      COL(0, j) COL(1, j) COL(2, j) COL(3, j) }
  for (const int8_t *end = a + (Kp & ~15); a != end; a += 16, b += 16) {
    STEP(0) STEP(8)
  }
  if (Kp & 8)
    STEP(0)
#undef STEP
#undef COL
#undef LD
  c[0] = c00;  c[1] = c01;  c[2] = c02;  c[3] = c03;
  c[4] = c10;  c[5] = c11;  c[6] = c12;  c[7] = c13;
  c[8] = c20;  c[9] = c21;  c[10] = c22; c[11] = c23;
  c[12] = c30; c[13] = c31; c[14] = c32; c[15] = c33;
}

__attribute__((noinline, optimize("no-schedule-insns")))
static void tile_4x4_64(const int8_t *a, const int8_t *b, const int32_t *rowc, const int32_t *colc,
                        int32_t c[16])
{
  tile_4x4_k(a, b, 64, rowc, colc, c);
}

__attribute__((noinline, optimize("no-schedule-insns")))
static void tile_4x4_200(const int8_t *a, const int8_t *b, const int32_t *rowc, const int32_t *colc,
                         int32_t c[16])
{
  tile_4x4_k(a, b, 200, rowc, colc, c);
}

void bmm_i8(const int8_t *A, const int8_t *B, int M, int N, int K, int Kp,
            int32_t za, int32_t zb, int32_t m0, int shift, int32_t zo, uint64_t rqp,
            int8_t *C, int ldc)
{
  static int32_t rowc[MAXDIM], colc[MAXDIM];
  for (int m = 0; m < M; m++)
    rowc[m] = -zb * row_sum(A + m * Kp, Kp) + K * za * zb;
  for (int n = 0; n < N; n++)
    colc[n] = -za * row_sum(B + n * Kp, Kp);

  int32_t c[16];
  int M4 = M & ~3, N4 = N & ~3;
  for (int m = 0; m < M4; m += 4) {
    const int8_t *a = A + m * Kp;
    int8_t *o = C + m * ldc;
    for (int n = 0; n < N4; n += 4) {
      if (Kp == 64)
        tile_4x4_64(a, B + n * Kp, rowc + m, colc + n, c);
      else
        tile_4x4_200(a, B + n * Kp, rowc + m, colc + n, c);
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          o[i * ldc + n + j] = store_q(c[4 * i + j], m0, shift, zo, rqp);
    }
    for (int i = 0; i < 4; i++)
      for (int n = N4; n < N; n++)
        o[i * ldc + n] = store_q(dot1(a + i * Kp, B + n * Kp, Kp, rowc[m + i] + colc[n]), m0, shift, zo, rqp);
  }
  for (int m = M4; m < M; m++)
    for (int n = 0; n < N; n++)
      C[m * ldc + n] = store_q(dot1(A + m * Kp, B + n * Kp, Kp, rowc[m] + colc[n]), m0, shift, zo, rqp);
}
#else
void bmm_i8(const int8_t *A, const int8_t *B, int M, int N, int K, int Kp,
            int32_t za, int32_t zb, int32_t m0, int shift, int32_t zo, uint64_t rqp,
            int8_t *C, int ldc)
{
  static int32_t rowc[MAXDIM], colc[MAXDIM];
  for (int m = 0; m < M; m++)
    rowc[m] = -zb * row_sum(A + m * Kp, Kp) + K * za * zb;
  for (int n = 0; n < N; n++)
    colc[n] = -za * row_sum(B + n * Kp, Kp);

  int32_t c[8];
  int N4 = N & ~3;
  int m = 0;
  for (; m + 1 < M; m += 2) {
    const int8_t *a0 = A + m * Kp, *a1 = a0 + Kp;
    int8_t *c0 = C + m * ldc, *c1 = c0 + ldc;
    for (int n = 0; n < N4; n += 4) {
      tile_2x4(a0, a1, B + n * Kp, Kp, rowc[m], rowc[m + 1], colc + n, c);
      for (int j = 0; j < 4; j++) {
        c0[n + j] = store_q(c[j], m0, shift, zo, rqp);
        c1[n + j] = store_q(c[4 + j], m0, shift, zo, rqp);
      }
    }
    for (int n = N4; n < N; n++) {
      c0[n] = store_q(dot1(a0, B + n * Kp, Kp, rowc[m] + colc[n]), m0, shift, zo, rqp);
      c1[n] = store_q(dot1(a1, B + n * Kp, Kp, rowc[m + 1] + colc[n]), m0, shift, zo, rqp);
    }
  }
  for (; m < M; m++)
    for (int n = 0; n < N; n++)
      C[m * ldc + n] = store_q(dot1(A + m * Kp, B + n * Kp, Kp, rowc[m] + colc[n]), m0, shift, zo, rqp);
}
#endif

void pack_rows(const int8_t *src, int rows, int K, int ld, int8_t *dst, int Kp)
{
  for (int r = 0; r < rows; r++) {
    const int8_t *s = src + r * ld;
    int8_t *d = dst + r * Kp;
    int k = 0;
    for (; k < K; k++) d[k] = s[k];
    for (; k < Kp; k++) d[k] = 0;
  }
}

void pack_transpose(const int8_t *src, int K, int N, int8_t *dst, int Kp)
{
  for (int n = 0; n < N; n++) {
    int8_t *d = dst + n * Kp;
    const int8_t *s = src + n;
    int k = 0;
    for (; k < K; k++, s += N) d[k] = *s;
    for (; k < Kp; k++) d[k] = 0;
  }
}
