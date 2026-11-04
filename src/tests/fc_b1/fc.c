// B1 software-optimized FC + requantization (DESIGN.md D3), same interface as B0 (fc.h).
//
// Register blocking: each inner loop computes a 2-row x 4-channel tile of outputs, so every
// loaded activation is used 4 times and every loaded weight twice (6 loads per 8 MACs
// instead of 16). K is unrolled 4x with constant load offsets, so the six pointer
// increments and the loop branch are paid once per 32 MACs. Weight layout is unchanged
// (row-major [N][K], K contiguous), which is also the layout a packed-MAC instruction wants.
#include "fc.h"

// Same fixed-point requantization as B0 (gemmlowp SRDHM, final shift ties toward +inf).
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

static inline int8_t store_q(int32_t acc, int32_t m0, int shift, int32_t out_zp)
{
  int32_t y = requant(acc, m0, shift) + out_zp;
  if (y < -128) y = -128;
  if (y > 127) y = 127;
  return (int8_t)y;
}

// 2 rows x 4 channels; K must be a multiple of 4 (192 and 768 in DeiT-Tiny).
// GCC's pre-register-allocation scheduler hoists all 24 loads of the unrolled body and then
// spills accumulators to the stack; disabling it for this function keeps everything in registers.
__attribute__((noinline, optimize("no-schedule-insns")))
static void tile_2x4(const int8_t *a0, const int8_t *a1, const int8_t *w0, int K,
                     const int32_t *bias, int32_t c[8])
{
  const int8_t *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
  int32_t c00 = bias[0], c01 = bias[1], c02 = bias[2], c03 = bias[3];
  int32_t c10 = bias[0], c11 = bias[1], c12 = bias[2], c13 = bias[3];
  for (const int8_t *end = a0 + K; a0 != end; a0 += 4, a1 += 4, w0 += 4, w1 += 4, w2 += 4, w3 += 4) {
#define STEP(j) {                                                   \
      int32_t x0 = a0[j], x1 = a1[j];                               \
      int32_t y0 = w0[j], y1 = w1[j], y2 = w2[j], y3 = w3[j];       \
      c00 += x0 * y0; c01 += x0 * y1; c02 += x0 * y2; c03 += x0 * y3; \
      c10 += x1 * y0; c11 += x1 * y1; c12 += x1 * y2; c13 += x1 * y3; }
    STEP(0) STEP(1) STEP(2) STEP(3)
#undef STEP
  }
  c[0] = c00; c[1] = c01; c[2] = c02; c[3] = c03;
  c[4] = c10; c[5] = c11; c[6] = c12; c[7] = c13;
}

// 1 row x 4 channels, for the last row when M is odd (M = 197).
__attribute__((noinline))
static void tile_1x4(const int8_t *a0, const int8_t *w0, int K, const int32_t *bias, int32_t c[4])
{
  const int8_t *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
  int32_t c0 = bias[0], c1 = bias[1], c2 = bias[2], c3 = bias[3];
  for (const int8_t *end = a0 + K; a0 != end; a0 += 4, w0 += 4, w1 += 4, w2 += 4, w3 += 4) {
    for (int j = 0; j < 4; j++) {
      int32_t x = a0[j];
      c0 += x * w0[j]; c1 += x * w1[j]; c2 += x * w2[j]; c3 += x * w3[j];
    }
  }
  c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3;
}

void fc_i8(const int8_t *in, const int8_t *W, const int32_t *bias,
           const int32_t *m0, const int8_t *shift, int32_t out_zp,
           int8_t *out, int M, int N, int K)
{
  int32_t c[8];
  int m = 0;
  for (; m + 1 < M; m += 2) {
    const int8_t *a0 = in + m * K;
    for (int n = 0; n < N; n += 4) {          // N is a multiple of 4 (576, 192, 768)
      tile_2x4(a0, a0 + K, W + n * K, K, bias + n, c);
      for (int j = 0; j < 4; j++) {
        out[m * N + n + j]       = store_q(c[j],     m0[n + j], shift[n + j], out_zp);
        out[(m + 1) * N + n + j] = store_q(c[4 + j], m0[n + j], shift[n + j], out_zp);
      }
    }
  }
  if (m < M) {
    const int8_t *a0 = in + m * K;
    for (int n = 0; n < N; n += 4) {
      tile_1x4(a0, W + n * K, K, bias + n, c);
      for (int j = 0; j < 4; j++)
        out[m * N + n + j] = store_q(c[j], m0[n + j], shift[n + j], out_zp);
    }
  }
}
