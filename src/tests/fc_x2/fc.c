// X2 study: FC on the outer-product prototype (xqop.h).
//   default   qop4: 4x4 output tile, operands are 4 rows x 2 K-steps (layout [M/4][K/2][4][2])
//   QOP8      qop8: 8x8 output tile, operands are 8 rows x 1 K-step  (layout [M/8][K][8])
// Rows are padded to a multiple of TG with zeros (197 -> 200); W is packed the same way offline.
// Per tile: QACC_COL sets every row of the accumulators to the folded bias, the loop issues
// one qop per two 8-byte loads, and each output is one qacc.rq + one sb.
#include "fc2.h"
#include "xqop.h"

typedef uint64_t __attribute__((may_alias)) u64a;
#define LD(p, off) (*(const u64a *)((p) + (off)))

void pack_g(const int8_t *src, int rows, int K, int rs, int ks, int Kp, int8_t *dst)
{
  int R = (rows + TG - 1) / TG * TG;
  for (int r = 0; r < R; r++)
    for (int k = 0; k < Kp; k++)
      dst[pidx(r, k, Kp)] = r < rows && k < K ? src[r * rs + k * ks] : 0;
}

#if TG == 4
#define OUT(i, j) if (m + (i) < M) o[(i) * N + (j)] = (int8_t)QACC_RQ(p##j, (i) * 8 + (j));
#define OUTROW(i) OUT(i, 0) OUT(i, 1) OUT(i, 2) OUT(i, 3)
__attribute__((noinline))
static void tile(const int8_t *a, const int8_t *w, int K, const int32_t *bias, const uint64_t *rqp,
                 int8_t *o, int m, int M, int N)
{
  QACC_COL(LD((const int8_t *)bias, 0), 0);
  QACC_COL(LD((const int8_t *)bias, 8), 1);
  for (const int8_t *end = a + 4 * K; a != end; a += 32, w += 32) {
    qop4(LD(a, 0), LD(w, 0));
    qop4(LD(a, 8), LD(w, 8));
    qop4(LD(a, 16), LD(w, 16));
    qop4(LD(a, 24), LD(w, 24));
  }
  uint64_t p0 = rqp[0], p1 = rqp[1], p2 = rqp[2], p3 = rqp[3];
  OUTROW(0) OUTROW(1) OUTROW(2) OUTROW(3)
}
#else
#define OUT(i, j) o[(i) * N + (j)] = (int8_t)QACC_RQ(p##j, (i) * 8 + (j));
#define OUTROW(i) if (m + (i) < M) { OUT(i, 0) OUT(i, 1) OUT(i, 2) OUT(i, 3) \
                                     OUT(i, 4) OUT(i, 5) OUT(i, 6) OUT(i, 7) }
__attribute__((noinline))
static void tile(const int8_t *a, const int8_t *w, int K, const int32_t *bias, const uint64_t *rqp,
                 int8_t *o, int m, int M, int N)
{
  QACC_COL(LD((const int8_t *)bias, 0), 0);
  QACC_COL(LD((const int8_t *)bias, 8), 1);
  QACC_COL(LD((const int8_t *)bias, 16), 2);
  QACC_COL(LD((const int8_t *)bias, 24), 3);
  for (const int8_t *end = a + 8 * K; a != end; a += 32, w += 32) {
    qop8(LD(a, 0), LD(w, 0));
    qop8(LD(a, 8), LD(w, 8));
    qop8(LD(a, 16), LD(w, 16));
    qop8(LD(a, 24), LD(w, 24));
  }
  uint64_t p0 = rqp[0], p1 = rqp[1], p2 = rqp[2], p3 = rqp[3];
  uint64_t p4 = rqp[4], p5 = rqp[5], p6 = rqp[6], p7 = rqp[7];
  OUTROW(0) OUTROW(1) OUTROW(2) OUTROW(3) OUTROW(4) OUTROW(5) OUTROW(6) OUTROW(7)
}
#endif

void fc_x2(const int8_t *Ap, const int8_t *Wp, const int32_t *bias, const uint64_t *rqp,
           int8_t *out, int M, int N, int K)
{
  for (int m = 0; m < M; m += TG)
    for (int n = 0; n < N; n += TG)
      tile(Ap + (m / TG) * TG * K, Wp + (n / TG) * TG * K, K, bias + n, rqp + n, out + m * N + n, m, M, N);
}
