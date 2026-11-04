// X2 study: BATCH_MATMUL on the outer-product prototype (DESIGN.md D10).
//
// As in src/tests/bmm, (a - za)(b - zb) is expanded into a raw int8 product plus per-row and
// per-column terms; here the accumulators start at them: qacc.col writes colc[n] into every row,
// qacc.row adds rowc[m] to every column. The loop and the readout are the same as fc_x2.
#include "../fc_x2/fc2.h"
#include "bmm2.h"
#include "xqop.h"

#define MAXDIM 200

typedef uint64_t __attribute__((may_alias)) u64a;
#define LD(p, off) (*(const u64a *)((const int8_t *)(p) + (off)))

#if TG == 4
#define QOP qop4
#define INIT(rc, cc) { QACC_COL(LD(cc, 0), 0); QACC_COL(LD(cc, 8), 1);                   \
                       QACC_ROW(LD(rc, 0), 0); QACC_ROW(LD(rc, 8), 1); }
#define ROW(X, i) X(i, 0) X(i, 1) X(i, 2) X(i, 3)
#define ROWS(G) G(0) G(1) G(2) G(3)
#define OFF(i, j) (((j) >> 1) * 8 + (i) * 2 + ((j) & 1))      // pidx(m0 + i, n0 + j) - pidx(m0, n0)
#else
#define QOP qop8
#define INIT(rc, cc) { QACC_COL(LD(cc, 0), 0); QACC_COL(LD(cc, 8), 1);                   \
                       QACC_COL(LD(cc, 16), 2); QACC_COL(LD(cc, 24), 3);                 \
                       QACC_ROW(LD(rc, 0), 0); QACC_ROW(LD(rc, 8), 1);                   \
                       QACC_ROW(LD(rc, 16), 2); QACC_ROW(LD(rc, 24), 3); }
#define ROW(X, i) X(i, 0) X(i, 1) X(i, 2) X(i, 3) X(i, 4) X(i, 5) X(i, 6) X(i, 7)
#define ROWS(G) G(0) G(1) G(2) G(3) G(4) G(5) G(6) G(7)
#define OFF(i, j) ((j) * 8 + (i))
#endif

#define LOOP()                                                                         \
  for (const int8_t *end = a + TG * Kp; a != end; a += 32, b += 32) {                 \
    QOP(LD(a, 0), LD(b, 0)); QOP(LD(a, 8), LD(b, 8));                                  \
    QOP(LD(a, 16), LD(b, 16)); QOP(LD(a, 24), LD(b, 24));                              \
  }

#define RM(i, j) o[(i) * ldc + (j)] = (int8_t)QACC_RQA(p, (i) * 8 + (j));
#define RM_ROW(i) if (m + (i) < M) { ROW(RM, i) }
__attribute__((noinline))
static void tile_rm(const int8_t *a, const int8_t *b, int Kp, const int32_t *rc, const int32_t *cc,
                    uint64_t p, int8_t *o, int ldc, int m, int M)
{
  INIT(rc, cc)
  LOOP()
  ROWS(RM_ROW)
}

#define PK(i, j) o[OFF(i, j)] = (int8_t)QACC_RQA(p, (i) * 8 + (j));
#define PK_ROW(i) if (m + (i) < M) { ROW(PK, i) }
__attribute__((noinline))
static void tile_pk(const int8_t *a, const int8_t *b, int Kp, const int32_t *rc, const int32_t *cc,
                    uint64_t p, int8_t *o, int m, int M)
{
  INIT(rc, cc)
  LOOP()
  ROWS(PK_ROW)
}

void bmm_x2(const int8_t *Ap, const int8_t *Bp, int M, int N, int K, int Kp,
            const int32_t *sumA, const int32_t *sumB, int32_t za, int32_t zb, uint64_t rqp,
            int8_t *C, int ldc, int packK)
{
  static int32_t rowc[MAXDIM] __attribute__((aligned(8))), colc[MAXDIM] __attribute__((aligned(8)));
  for (int m = 0; m < M; m++)
    rowc[m] = -zb * sumA[m] + K * za * zb;
  for (int n = 0; n < N; n++)
    colc[n] = -za * sumB[n];
  for (int m = 0; m < M; m += TG) {
    const int8_t *a = Ap + m * Kp;
    for (int n = 0; n < N; n += TG) {
      const int8_t *b = Bp + n * Kp;
      if (packK)
        tile_pk(a, b, Kp, rowc + m, colc + n, rqp, C + pidx(m, n, packK), m, M);
      else
        tile_rm(a, b, Kp, rowc + m, colc + n, rqp, C + m * ldc + n, ldc, m, M);
    }
  }
}
