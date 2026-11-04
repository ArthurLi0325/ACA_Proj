// X3: BATCH_MATMUL with the row-vector dot products (DESIGN.md D12).
//
// As in src/tests/bmm, (a - za)(b - zb) is expanded into a raw int8 product plus per-row and
// per-column terms; each output row starts at rowc[m] + colc[n] (one vadd.vx per row). The loop is
// the one of fc_x3 (kern.S); the requantization is rq.a (per tensor).
#include "../fc_x3/fc3.h"
#include "bmm3.h"

#define MAXDIM 200

typedef uint64_t __attribute__((may_alias)) u64a;
typedef uint32_t __attribute__((may_alias)) u32a;

void bmm3_tile(const int8_t *a, const int8_t *b, const int8_t *aend, const int32_t *rowc,
               const int32_t *colc, const int32_t prm[4], int8_t *o, long ldo);

void bmm_x3(const int8_t *Ap, const int8_t *Bp, int M, int N, int K, int Kp,
            const int32_t *sumA, const int32_t *sumB, int32_t za, int32_t zb, uint64_t rqp,
            int8_t *C, int ldc, int packK)
{
  static int32_t rowc[MAXDIM], colc[MAXDIM];
  static int8_t tmp[64] __attribute__((aligned(8)));     // last row tile, rows past M dropped
  for (int m = 0; m < M; m++)
    rowc[m] = -zb * sumA[m] + K * za * zb;
  for (int n = 0; n < N; n++)
    colc[n] = -za * sumB[n];
  const int sh = (int)((int64_t)(rqp << 26) >> 58);
  // row r of a tile: row-major at o + r*ldc; packed: QD bytes at o + r*QD and (QD = 4) o + r*4 + 32
  const int32_t prm[4] = { (int32_t)rqp, -sh, (int8_t)(rqp >> 40), packK && QD == 4 ? 32 : 4 };
  const int32_t prm_tmp[4] = { prm[0], prm[1], prm[2], 4 };
  const long ldo = packK ? QD : ldc;
  for (int m = 0; m < M; m += TG) {
    const int8_t *a = Ap + m * Kp;
    const int rows = M - m < TG ? M - m : TG;
    for (int n = 0; n < N; n += TG) {
      int8_t *o = packK ? C + pidx(m, n, packK) : C + m * ldc + n;
      if (rows == TG) {
        bmm3_tile(a, Bp + n * Kp, a + TG * Kp, rowc + m, colc + n, prm, o, ldo);
        continue;
      }
      bmm3_tile(a, Bp + n * Kp, a + TG * Kp, rowc + m, colc + n, prm_tmp, tmp, TG);
      for (int r = 0; r < rows; r++) {
        if (!packK)
          *(u64a *)(o + r * ldc) = ((const u64a *)tmp)[r];
        else if (QD == 8)
          *(u64a *)(o + r * 8) = ((const u64a *)tmp)[r];
        else {
          *(u32a *)(o + r * 4) = ((const u32a *)tmp)[2 * r];
          *(u32a *)(o + r * 4 + 32) = ((const u32a *)tmp)[2 * r + 1];
        }
      }
    }
  }
}
