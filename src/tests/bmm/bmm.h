// int8 BATCH_MATMUL kernel interface for the block-0 attention tests (QK^T and AV), one head.
#ifndef BMM_H
#define BMM_H
#include <stdint.h>

// For m < M, n < N:
//   acc = sum_{k<K} (A[m*Kp + k] - za) * (B[n*Kp + k] - zb)
//       = sum a*b - zb*sum_k a - za*sum_k b + K*za*zb      (the form the kernel computes)
//   C[m*ldc + n] = clamp(requant_a(acc) + zo, -128, 127)
// Both operands are K-contiguous (B holds the rows that are dotted with the rows of A), 8-byte
// aligned, with row stride Kp (a multiple of 8) and zeros in columns K..Kp-1, so padding adds
// nothing to any sum. requant_a is TFLite's BATCH_MATMUL rounding: SRDHM, then a right shift with
// ties away from zero (DESIGN.md D2/D5). rqp is the same multiplier packed for rq.a.
void bmm_i8(const int8_t *A, const int8_t *B, int M, int N, int K, int Kp,
            int32_t za, int32_t zb, int32_t m0, int shift, int32_t zo, uint64_t rqp,
            int8_t *C, int ldc);

// dst[r*Kp + k] = src[r*ld + k] for k < K, 0 for K <= k < Kp (rows of A that are not 8-aligned).
void pack_rows(const int8_t *src, int rows, int K, int ld, int8_t *dst, int Kp);

// dst[n*Kp + k] = src[k*N + n] for k < K, 0 for K <= k < Kp (V is [K][N]; the kernel wants V^T).
void pack_transpose(const int8_t *src, int K, int N, int8_t *dst, int Kp);

#endif
