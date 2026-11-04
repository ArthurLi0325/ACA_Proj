// X1 FC kernel with the Xqmac instructions (DESIGN.md D4/D5), same tiling as B1 (D3).
//
// The 2x4 register blocking of B1 is kept, but each load now brings 8 int8 values (one `ld`)
// and one qmac8 does 8 MACs: per 8 K-steps the tile needs 6 `ld` and 8 qmac8 for 64 MACs.
// K is unrolled 2x (16 bytes per iteration) so pointer increments and the branch are paid once
// per 128 MACs. Variants, selected at compile time:
//   default      qmac8, software requantization (same code as B1)
//   USE_DOT8     dot8 + addw: the 2-read-port alternative of D4
//   USE_RQ       qmac8 + the rq instruction with the packed parameter word rqp[n]
//   TILE4        4x4 tiles instead of 2x4 (software only, X2 study option A): 8 `ld` per 16 qmac8
//                instead of 6 per 8, and K is a compile-time constant (192 or 768) so all rows
//                are reached with immediate offsets from one activation and one weight pointer
#include "fc.h"
#include "xqmac.h"

// Lets the int8 arrays be read as packed 64-bit words without breaking strict aliasing.
typedef uint64_t __attribute__((may_alias)) u64a;

#ifdef USE_DOT8
#define MAC(c, a, b) ((c) += dot8(a, b))
#else
#define MAC(c, a, b) ((c) = qmac8(c, a, b))
#endif

#ifdef USE_RQ
static inline int8_t store_q(int32_t acc, int n, const int32_t *m0, const int8_t *shift,
                             int32_t out_zp, const uint64_t *rqp)
{
  (void)m0; (void)shift; (void)out_zp;
  return (int8_t)rq(acc, rqp[n]);
}
#else
// Same fixed-point requantization as B0/B1 (gemmlowp SRDHM, final shift ties toward +inf).
__attribute__((noinline))
static int32_t srdhm(int32_t a, int32_t b)
{
  if (a == INT32_MIN && b == INT32_MIN)
    return INT32_MAX;
  int64_t ab = (int64_t)a * b;
  int64_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
  return (int32_t)((ab + nudge) / ((int64_t)1 << 31));
}

__attribute__((noinline))
static int32_t requant(int32_t acc, int32_t m0, int shift)
{
  int left = shift > 0 ? shift : 0;
  int right = shift > 0 ? 0 : -shift;
  int32_t v = srdhm(acc * (1 << left), m0);
  if (right == 0)
    return v;
  return (v + (1 << (right - 1))) >> right;
}

static inline int8_t store_q(int32_t acc, int n, const int32_t *m0, const int8_t *shift,
                             int32_t out_zp, const uint64_t *rqp)
{
  (void)rqp;
  int32_t y = requant(acc, m0[n], shift[n]) + out_zp;
  if (y < -128) y = -128;
  if (y > 127) y = 127;
  return (int8_t)y;
}
#endif

#ifndef TILE4
// 2 rows x 4 channels, 16 K-steps per iteration.
__attribute__((noinline))
static void tile_2x4(const int8_t *a0, const int8_t *a1, const int8_t *w0, int K,
                     const int32_t *bias, int32_t c[8])
{
  const u64a *x0 = (const u64a *)a0, *x1 = (const u64a *)a1;
  const u64a *y0 = (const u64a *)w0, *y1 = (const u64a *)(w0 + K);
  const u64a *y2 = (const u64a *)(w0 + 2 * K), *y3 = (const u64a *)(w0 + 3 * K);
  int64_t c00 = bias[0], c01 = bias[1], c02 = bias[2], c03 = bias[3];   // see xqmac.h
  int64_t c10 = bias[0], c11 = bias[1], c12 = bias[2], c13 = bias[3];
  for (const u64a *end = x0 + K / 8; x0 != end; x0 += 2, x1 += 2, y0 += 2, y1 += 2, y2 += 2, y3 += 2) {
#define STEP(j) {                                                   \
      uint64_t u0 = x0[j], u1 = x1[j];                              \
      uint64_t v0 = y0[j], v1 = y1[j], v2 = y2[j], v3 = y3[j];      \
      MAC(c00, u0, v0); MAC(c01, u0, v1); MAC(c02, u0, v2); MAC(c03, u0, v3); \
      MAC(c10, u1, v0); MAC(c11, u1, v1); MAC(c12, u1, v2); MAC(c13, u1, v3); }
    STEP(0) STEP(1)
#undef STEP
  }
  c[0] = c00; c[1] = c01; c[2] = c02; c[3] = c03;
  c[4] = c10; c[5] = c11; c[6] = c12; c[7] = c13;
}
#endif

// 1 row x 4 channels, for the last row when M is odd (M = 197).
__attribute__((noinline))
static void tile_1x4(const int8_t *a0, const int8_t *w0, int K, const int32_t *bias, int32_t c[4])
{
  const u64a *x0 = (const u64a *)a0;
  const u64a *y0 = (const u64a *)w0, *y1 = (const u64a *)(w0 + K);
  const u64a *y2 = (const u64a *)(w0 + 2 * K), *y3 = (const u64a *)(w0 + 3 * K);
  int64_t c0 = bias[0], c1 = bias[1], c2 = bias[2], c3 = bias[3];
  for (const u64a *end = x0 + K / 8; x0 != end; x0++, y0++, y1++, y2++, y3++) {
    uint64_t u = *x0;
    MAC(c0, u, *y0); MAC(c1, u, *y1); MAC(c2, u, *y2); MAC(c3, u, *y3);
  }
  c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3;
}

#ifdef TILE4
// 4 rows x 4 channels, 16 K-steps per iteration: 16 ld + 32 qmac8 + 2 pointer increments + 1 branch
// per 256 MACs. Inlined into one function per K so that every row offset is an immediate.
static inline __attribute__((always_inline))
void tile_4x4_k(const int8_t *a, const int8_t *w, const int K, const int32_t *bias, int32_t c[16])
{
  int64_t c00 = bias[0], c01 = bias[1], c02 = bias[2], c03 = bias[3];
  int64_t c10 = bias[0], c11 = bias[1], c12 = bias[2], c13 = bias[3];
  int64_t c20 = bias[0], c21 = bias[1], c22 = bias[2], c23 = bias[3];
  int64_t c30 = bias[0], c31 = bias[1], c32 = bias[2], c33 = bias[3];
  for (const int8_t *end = a + K; a != end; a += 16, w += 16) {
#define LD(p, off) (*(const u64a *)((p) + (off)))
#define COL(n, j) { uint64_t v = LD(w, (n) * K + (j));                              \
      MAC(c0##n, u0, v); MAC(c1##n, u1, v); MAC(c2##n, u2, v); MAC(c3##n, u3, v); }
#define STEP(j) { uint64_t u0 = LD(a, j), u1 = LD(a, K + (j)), u2 = LD(a, 2 * K + (j)),  \
                           u3 = LD(a, 3 * K + (j));                                \
      COL(0, j) COL(1, j) COL(2, j) COL(3, j) }
    STEP(0) STEP(8)
#undef STEP
#undef COL
#undef LD
  }
  c[0] = c00;  c[1] = c01;  c[2] = c02;  c[3] = c03;
  c[4] = c10;  c[5] = c11;  c[6] = c12;  c[7] = c13;
  c[8] = c20;  c[9] = c21;  c[10] = c22; c[11] = c23;
  c[12] = c30; c[13] = c31; c[14] = c32; c[15] = c33;
}

__attribute__((noinline, optimize("no-schedule-insns")))
static void tile_4x4_192(const int8_t *a, const int8_t *w, const int32_t *bias, int32_t c[16])
{
  tile_4x4_k(a, w, 192, bias, c);
}

__attribute__((noinline, optimize("no-schedule-insns")))
static void tile_4x4_768(const int8_t *a, const int8_t *w, const int32_t *bias, int32_t c[16])
{
  tile_4x4_k(a, w, 768, bias, c);
}

void fc_i8(const int8_t *in, const int8_t *W, const int32_t *bias,
           const int32_t *m0, const int8_t *shift, int32_t out_zp, const uint64_t *rqp,
           int8_t *out, int M, int N, int K)
{
  int32_t c[16];
  int m = 0;
  for (; m + 3 < M; m += 4) {
    const int8_t *a0 = in + m * K;
    int8_t *o = out + m * N;
    for (int n = 0; n < N; n += 4) {
      if (K == 192)
        tile_4x4_192(a0, W + n * K, bias + n, c);
      else
        tile_4x4_768(a0, W + n * K, bias + n, c);
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          o[i * N + n + j] = store_q(c[4 * i + j], n + j, m0, shift, out_zp, rqp);
    }
  }
  for (; m < M; m++) {                          // M = 197: one row left
    const int8_t *a0 = in + m * K;
    for (int n = 0; n < N; n += 4) {
      tile_1x4(a0, W + n * K, K, bias + n, c);
      for (int j = 0; j < 4; j++)
        out[m * N + n + j] = store_q(c[j], n + j, m0, shift, out_zp, rqp);
    }
  }
}
#else
void fc_i8(const int8_t *in, const int8_t *W, const int32_t *bias,
           const int32_t *m0, const int8_t *shift, int32_t out_zp, const uint64_t *rqp,
           int8_t *out, int M, int N, int K)
{
  int32_t c[8];
  int m = 0;
  for (; m + 1 < M; m += 2) {
    const int8_t *a0 = in + m * K;
    for (int n = 0; n < N; n += 4) {          // N is a multiple of 4 (576, 192, 768)
      tile_2x4(a0, a0 + K, W + n * K, K, bias + n, c);
      for (int j = 0; j < 4; j++) {
        out[m * N + n + j]       = store_q(c[j],     n + j, m0, shift, out_zp, rqp);
        out[(m + 1) * N + n + j] = store_q(c[4 + j], n + j, m0, shift, out_zp, rqp);
      }
    }
  }
  if (m < M) {
    const int8_t *a0 = in + m * K;
    for (int n = 0; n < N; n += 4) {
      tile_1x4(a0, W + n * K, K, bias + n, c);
      for (int j = 0; j < 4; j++)
        out[m * N + n + j] = store_q(c[j], n + j, m0, shift, out_zp, rqp);
    }
  }
}
#endif
