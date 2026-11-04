// int8 FULLY_CONNECTED kernel interface for the block-0 FC tests.
#ifndef FC_H
#define FC_H
#include <stdint.h>

// For m < M, n < N:
//   acc = bias[n] + sum_{k<K} in[m*K + k] * W[n*K + k]        (int32; zp_in already folded into bias)
//   out[m*N + n] = clamp(requant(acc, m0[n], shift[n]) + out_zp, -128, 127)
// requant: real multiplier = m0 * 2^(shift-31), shift > 0 means left shift.
// LiteRT's FC rounds the final right shift with ties toward +inf (DESIGN.md D2, EXPERIMENTS.md E1);
// that is the rule the golden output follows.
void fc_i8(const int8_t *in, const int8_t *W, const int32_t *bias,
           const int32_t *m0, const int8_t *shift, int32_t out_zp,
           int8_t *out, int M, int N, int K);

#endif
