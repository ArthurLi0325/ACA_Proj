// X2 study: BATCH_MATMUL (one head) on the outer-product prototype (DESIGN.md D10), see bmm.c.
#ifndef BMM2_H
#define BMM2_H
#include <stdint.h>

// C[m][n] = rq.a(sum_{k<K} (A[m][k] - za) * (B[n][k] - zb), rqp) for m < M, n < N, where Ap and Bp
// are A [M][K] and B [N][K] in the qop4 / qop8 operand layout with K padded to Kp (pack_g, pidx)
// and sumA[m], sumB[n] are their row sums (the producers can count them while writing).
// packK == 0: C is row-major with stride ldc >= N rounded up to the tile size (the extra columns
// are overwritten). packK > 0: C is written in the operand layout of a [M][packK] matrix (pidx),
// i.e. directly as the input of the next FC; N must then be a multiple of the tile size.
void bmm_x2(const int8_t *Ap, const int8_t *Bp, int M, int N, int K, int Kp,
            const int32_t *sumA, const int32_t *sumB, int32_t za, int32_t zb, uint64_t rqp,
            int8_t *C, int ldc, int packK);

#endif
