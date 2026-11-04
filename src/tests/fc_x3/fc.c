// X3: FC with the row-vector dot products (DESIGN.md D12). Each tile is 8 output rows x 8 output
// columns; output row r is accumulated in vector register v<r> (kern.S, fc3_tile). The loop issues,
// per QD-byte slice of K, one vector load of the 8 weight columns, 8 scalar loads (QD bytes of each
// activation row) and 8 vqdot instructions. Rows are padded to a multiple of 8 (197 -> 200).
#include "fc3.h"

typedef uint64_t __attribute__((may_alias)) u64a;

void fc3_tile(const int8_t *a, const int8_t *w, const int8_t *aend, const int32_t *bias,
              const uint64_t *rqp, int32_t zo, int8_t *o, long ldo);

void pack_g(const int8_t *src, int rows, int K, int rs, int ks, int Kp, int8_t *dst)
{
  int R = (rows + TG - 1) / TG * TG;
  for (int r = 0; r < R; r++)
    for (int k = 0; k < Kp; k++)
      dst[pidx(r, k, Kp)] = r < rows && k < K ? src[r * rs + k * ks] : 0;
}

void fc_x3(const int8_t *Ap, const int8_t *Wp, const int32_t *bias, const uint64_t *rqp,
           int8_t *out, int M, int N, int K)
{
  static int8_t tmp[64] __attribute__((aligned(8)));     // last row tile, rows past M dropped
  const int32_t zo = (int8_t)(rqp[0] >> 40);
  for (int m = 0; m < M; m += TG) {
    const int8_t *a = Ap + m * K;
    const int rows = M - m < TG ? M - m : TG;
    for (int n = 0; n < N; n += TG) {
      int8_t *o = out + m * N + n;
      fc3_tile(a, Wp + n * K, a + TG * K, bias + n, rqp + n, zo, rows == TG ? o : tmp,
               rows == TG ? N : TG);
      if (rows < TG)
        for (int r = 0; r < rows; r++)
          *(u64a *)(o + r * N) = ((const u64a *)tmp)[r];
    }
  }
}
