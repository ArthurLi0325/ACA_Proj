// X3: BATCH_MATMUL (one head) with the row-vector dot products (DESIGN.md D12), see bmm.c.
#ifndef BMM3_H
#define BMM3_H
#include <stdint.h>

// C[m][n] = rq.a(sum_{k<K} (A[m][k] - za) * (B[n][k] - zb), rqp) for m < M, n < N, where Ap and Bp
// are A [M][K] and B [N][K] in the X3 operand layout with K padded to Kp (pack_g, pidx) and
// sumA[m], sumB[n] are their row sums (the producers count them while writing).
// packK == 0: C is row-major with stride ldc >= N rounded up to 8 (the extra columns are
// overwritten). packK > 0: C is written in the operand layout of a [M][packK] matrix (pidx), i.e.
// directly as the input of the next FC; N must then be a multiple of 8.
void bmm_x3(const int8_t *Ap, const int8_t *Bp, int M, int N, int K, int Kp,
            const int32_t *sumA, const int32_t *sumB, int32_t za, int32_t zb, uint64_t rqp,
            int8_t *C, int ldc, int packK);

#define bmm_x2 bmm_x3               // the name src/common/vit.h calls

#endif
