// X2 study FC interface (outer-product prototype), see fc.c.
#ifndef FC2_H
#define FC2_H
#include <stdint.h>

#ifdef QOP8
#define TG 8
#else
#define TG 4
#endif

// Byte offset of element (r, k) of a [rows][K] operand in the qop4 / qop8 layout (K even).
static inline int pidx(int r, int k, int K)
{
#if TG == 4
  return ((r >> 2) * (K >> 1) + (k >> 1)) * 8 + (r & 3) * 2 + (k & 1);
#else
  return ((r >> 3) * K + k) * 8 + (r & 7);
#endif
}

// Along a row (k -> k + 1, k even) the packed offset grows by PK1, and by PK2 from k to k + 2, so
// producers write two elements per step with constant offsets.
#define PK1 (TG == 4 ? 1 : 8)
#define PK2 (TG == 4 ? 8 : 16)

// dst = packed copy of the [rows][K] matrix with element (r, k) at src[r*rs + k*ks]; rows are padded
// to a multiple of TG and K to Kp with zeros.
void pack_g(const int8_t *src, int rows, int K, int rs, int ks, int Kp, int8_t *dst);

// out[m*N + n] = rq(bias[n] + sum_k A[m][k] * W[n][k], rqp[n]); Ap, Wp packed by pack_g.
void fc_x2(const int8_t *Ap, const int8_t *Wp, const int32_t *bias, const uint64_t *rqp,
           int8_t *out, int M, int N, int K);

#endif
