// Runs bmm_i8() (bmm.c) on one block-0 attention BATCH_MATMUL, all 3 heads, and compares every
// output with the LiteRT golden from scripts/export_block.py (run `make export` first).
//   make -B TEST=bmm run EXT=xqmac                        QK^T, qmac8 + rq.a
//   make -B TEST=bmm run EXT=xqmac OPT="-O2 -DBMM_SEL=1"  AV
//   make -B TEST=bmm run OPT="-O2 -DSCALAR"               B1-style scalar baseline (no EXT needed)
//   make -B TEST=bmm hist EXT=xqmac HIST_ARGS="--funcs bmm_i8,tile_2x4,dot1,row_sum,requant_a,srdhm --macs <MACs>"
// AV's operands are first packed (pack_rows, pack_transpose); that part is counted separately,
// since the producers (softmax, the V requantization) could write the packed layout directly.
#include "runtime.h"
#include "block0_params.h"
#include "bmm.h"

#ifndef BMM_SEL
#define BMM_SEL 0
#endif

#define H 3
#define T 197                       // tokens
#define D 64                        // head dimension

#if BMM_SEL == 0
#include "block0_golden_qk.h"
#define L(x) B0_QK_##x
#define G(x) B0_GOLDEN_QK_##x
#define NAME "QK^T"
#define M T
#define N T
#define K D
#define KP D                        // Q and K rows are already 64 bytes: no packing
#else
#include "block0_golden_av.h"
#define L(x) B0_AV_##x
#define G(x) B0_GOLDEN_AV_##x
#define NAME "AV"
#define M T
#define N D
#define K T
#define KP 200                      // 197 rounded up to a multiple of 8
static int8_t pa[M * KP] __attribute__((aligned(8)));
static int8_t pb[N * KP] __attribute__((aligned(8)));
#endif

#ifdef SCALAR
#define VARIANT "scalar, software requant"
#else
#define VARIANT "qmac8 + rq.a"
#endif

static int8_t out[H * M * N];

int main(void)
{
  const uint64_t macs = (uint64_t)H * M * N * K;
  printf("BMM %s (%s): %d heads x M=%d N=%d K=%d, %lu MACs\n", NAME, VARIANT, H, M, N, K, macs);

  uint64_t pack = 0, comp = 0, c0 = rdcycle();
  for (int h = 0; h < H; h++) {
    uint64_t i0 = rdinstret();
#if BMM_SEL == 0
    const int8_t *a = G(LHS) + h * M * K, *b = G(RHS) + h * N * K;
#else
    MARK(3);
    pack_rows(G(LHS) + h * M * K, M, K, K, pa, KP);          // P[h] is [M][K], rows of 197 bytes
    pack_transpose(G(RHS) + h * K * N, K, N, pb, KP);        // V[h] is [K][N]
    MARK(4);
    const int8_t *a = pa, *b = pb;
#endif
    uint64_t i1 = rdinstret();
    MARK(1);
    bmm_i8(a, b, M, N, K, KP, L(LHS_ZP), L(RHS_ZP), L(M0), L(SHIFT), L(OUT_ZP), L(RQP),
           out + h * M * N, N);
    MARK(2);
    uint64_t i2 = rdinstret();
    pack += i1 - i0;
    comp += i2 - i1;
  }
  uint64_t c1 = rdcycle();

  int bad = 0, first = -1;
  for (int i = 0; i < H * M * N; i++)
    if (out[i] != G(OUT)[i]) { if (first < 0) first = i; bad++; }

  printf("instret: compute = %lu (%lu.%03lu instr/MAC), pack = %lu; cycles = %lu\n",
         comp, comp / macs, (comp % macs) * 1000 / macs, pack, c1 - c0);
  if (bad)
    printf("FAIL: %d / %d outputs differ; first at %d: got %d, golden %d\n",
           bad, H * M * N, first, out[first], G(OUT)[first]);
  else
    printf("PASS: all %d outputs bit-exact\n", H * M * N);
  return bad ? 1 : 0;
}
