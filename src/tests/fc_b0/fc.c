// B0 baseline FC + requantization (see fc.h for the exact semantics).
#include "fc.h"

// round(a * b / 2^31) as int32 (gemmlowp SaturatingRoundingDoublingHighMul).
// m0 is a Q31 fixed-point number: m0 = 2^30 means 0.5, so a * m0 / 2^31 = a * (m0 / 2^31).
// The product needs 64 bits; adding 'nudge' before dividing rounds to nearest.
__attribute__((noinline))
static int32_t srdhm(int32_t a, int32_t b)
{
  if (a == INT32_MIN && b == INT32_MIN)     // the only case whose result does not fit
    return INT32_MAX;
  int64_t ab = (int64_t)a * b;
  int64_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
  return (int32_t)((ab + nudge) / ((int64_t)1 << 31));
}

// acc * m0 * 2^(shift-31), rounded. In DeiT-Tiny shift is <= 0 (real multiplier < 1),
// so in practice this is: srdhm, then a rounding right shift by -shift.
// Ties round toward +inf (LiteRT FC rule, DESIGN.md D2).
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

void fc_i8(const int8_t *in, const int8_t *W, const int32_t *bias,
           const int32_t *m0, const int8_t *shift, int32_t out_zp,
           int8_t *out, int M, int N, int K)
{
  for (int m = 0; m < M; m++) {             // one input row (token)
    const int8_t *a = in + m * K;           // a[k] = in[m][k]
    for (int n = 0; n < N; n++) {           // one output channel
      const int8_t *w = W + n * K;          // w[k] = W[n][k]
      int32_t acc = bias[n];
      for (int k = 0; k < K; k++)
        acc += a[k] * w[k];


      int32_t y = requant(acc, m0[n], shift[n]) + out_zp;
      if (y < -128) y = -128;
      if (y > 127) y = 127;
      out[m * N + n] = (int8_t)y;
    }
  }
}
