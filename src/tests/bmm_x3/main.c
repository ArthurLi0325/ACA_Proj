// X3: runs bmm_x3() (bmm.c, row-vector dot products) on one block-0 attention BATCH_MATMUL,
// all 3 heads, and compares with the LiteRT golden (run `make export` first).
//   make -B TEST=bmm_x3 run EXT=xqmac VEC=1                         QK^T, vqdot4.vx
//   make -B TEST=bmm_x3 run EXT=xqmac VEC=1 OPT="-O2 -DQDOT8"       QK^T, vqdot8.vx
//   add -DBMM_SEL=1 to OPT for AV
// Packing and row sums are counted separately: in the block, the producers (Q/K/V split, softmax)
// write the packed layout and count the sums.
#include "runtime.h"
#include "block0_params.h"
#include "../fc_x3/fc3.h"
#include "bmm3.h"

#ifndef BMM_SEL
#define BMM_SEL 0
#endif

#define H 3
#define T 197
#define D 64
#define TP 200

#if BMM_SEL == 0
#include "block0_golden_qk.h"
#define L(x) B0_QK_##x
#define G(x) B0_GOLDEN_QK_##x
#define NAME "QK^T"
#define M T
#define N T
#define K D
#define KP D
#define LDC TP
#else
#include "block0_golden_av.h"
#define L(x) B0_AV_##x
#define G(x) B0_GOLDEN_AV_##x
#define NAME "AV"
#define M T
#define N D
#define K T
#define KP TP
#define LDC D
#endif

#ifdef QDOT8
#define VARIANT "vqdot8.vx"
#else
#define VARIANT "vqdot4.vx"
#endif

static int8_t ap[TP * KP] __attribute__((aligned(8)));
static int8_t bp[TP * KP] __attribute__((aligned(8)));
static int32_t sa[TP], sb[TP];
static int8_t out[TP * LDC];

static void sums(const int8_t *src, int rows, int Kn, int rs, int ks, int32_t *s)
{
  for (int r = 0; r < rows; r++) {
    int32_t t = 0;
    for (int k = 0; k < Kn; k++)
      t += src[r * rs + k * ks];
    s[r] = t;
  }
}

int main(void)
{
  const uint64_t macs = (uint64_t)H * M * N * K;
  printf("BMM %s (%s): %d heads x M=%d N=%d K=%d, %lu MACs\n", NAME, VARIANT, H, M, N, K, macs);
  uint64_t prep = 0, comp = 0;
  int bad = 0, first = -1;
  for (int h = 0; h < H; h++) {
    uint64_t i0 = rdinstret();
    const int8_t *A = G(LHS) + h * M * K, *B = G(RHS) + h * K * N;
#if BMM_SEL == 0
    pack_g(A, M, K, K, 1, KP, ap); sums(A, M, K, K, 1, sa);
    pack_g(B, N, K, K, 1, KP, bp); sums(B, N, K, K, 1, sb);         // K[h] is [N][K]
#else
    pack_g(A, M, K, K, 1, KP, ap); sums(A, M, K, K, 1, sa);
    pack_g(B, N, K, 1, N, KP, bp); sums(B, N, K, 1, N, sb);         // V[h] is [K][N]
#endif
    uint64_t i1 = rdinstret();
    MARK(1);
    bmm_x3(ap, bp, M, N, K, KP, sa, sb, L(LHS_ZP), L(RHS_ZP), L(RQP), out, LDC, 0);
    MARK(2);
    uint64_t i2 = rdinstret();
    prep += i1 - i0;
    comp += i2 - i1;
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++)
        if (out[m * LDC + n] != G(OUT)[(h * M + m) * N + n]) {
          if (first < 0) first = (h * M + m) * N + n;
          bad++;
        }
  }
  printf("instret: compute = %lu (%lu.%03lu instr/MAC), pack + sums = %lu\n",
         comp, comp / macs, (comp % macs) * 1000 / macs, prep);
  if (bad)
    printf("FAIL: %d / %d outputs differ; first at %d: golden %d\n", bad, H * M * N, first, G(OUT)[first]);
  else
    printf("PASS: all %d outputs bit-exact\n", H * M * N);
  return bad ? 1 : 0;
}
