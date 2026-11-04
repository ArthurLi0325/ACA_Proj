// X3 FC / operand layout interface (row-vector dot products, DESIGN.md D12), see fc.c.
#ifndef FC3_H
#define FC3_H
#include <stdint.h>

#ifdef QDOT8
#define QD 8                        // bytes per lane per instruction (vqdot8.vx)
#else
#define QD 4                        // vqdot4.vx
#endif
#define TG 8                        // rows per tile: 8 output rows x 8 output columns

// Byte offset of element (r, k) of a [rows][K] operand in the X3 layout [rows/8][K/QD][8][QD]
// (K a multiple of QD): one 8-row group holds, for each QD-wide slice of K, QD bytes of each row.
// The activation rows and the weight columns use the same layout.
static inline int pidx(int r, int k, int K)
{
  return ((r >> 3) * (K / QD) + k / QD) * (8 * QD) + (r & 7) * QD + (k & (QD - 1));
}

// Along a row, element k + 1 (k even) is at +PK1, and the pair at k + 2 is at +PSTEP(k).
#define PK1 1
#define PSTEP(k) ((((k) & (QD - 2)) == QD - 2) ? 8 * QD - QD + 2 : 2)

// dst = packed copy of the [rows][K] matrix with element (r, k) at src[r*rs + k*ks]; rows are padded
// to a multiple of 8 and K to Kp with zeros.
void pack_g(const int8_t *src, int rows, int K, int rs, int ks, int Kp, int8_t *dst);

// out[m*N + n] = rq(bias[n] + sum_k A[m][k] * W[n][k], rqp[n]); Ap, Wp packed by pack_g, N % 8 == 0.
void fc_x3(const int8_t *Ap, const int8_t *Wp, const int32_t *bias, const uint64_t *rqp,
           int8_t *out, int M, int N, int K);

#define fc_x2 fc_x3                 // the name src/common/vit.h calls

#endif
