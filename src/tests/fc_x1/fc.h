// int8 FULLY_CONNECTED kernel interface for the X1 tests (Xqmac instructions).
#ifndef FC_H
#define FC_H
#include <stdint.h>

// Same result as fc_b0/fc_b1 (see fc_b0/fc.h):
//   out[m*N + n] = clamp(requant(bias[n] + sum_k in[m*K + k] * W[n*K + k], m0[n], shift[n]) + out_zp)
// rqp[n] is the same requantization packed for the rq instruction (M0 | shift << 32 | out_zp << 40,
// DESIGN.md D5); the kernel uses either m0/shift/out_zp or rqp, depending on USE_RQ.
// in and W must be 8-byte aligned and K a multiple of 16.
void fc_i8(const int8_t *in, const int8_t *W, const int32_t *bias,
           const int32_t *m0, const int8_t *shift, int32_t out_zp, const uint64_t *rqp,
           int8_t *out, int M, int N, int K);

#endif
