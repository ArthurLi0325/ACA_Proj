// X3: runs fc_x3() (fc.c, row-vector dot products) on one block-0 FC layer and compares every
// output with the LiteRT golden (run `make export` first). Needs patches/spike-xqv.patch.
//   make -B TEST=fc_x3 run EXT=xqmac VEC=1                      vqdot4.vx
//   make -B TEST=fc_x3 run EXT=xqmac VEC=1 OPT="-O2 -DQDOT8"    vqdot8.vx
//   add -DFC_SEL=n to OPT for another layer: 0 QKV, 1 PROJ, 2 FC1, 3 FC2
// Weight packing is done here at start-up but would be offline (export_block.py); activation
// packing is counted separately, since the producer of the activations could write that layout.
#include "runtime.h"
#include "block0_params.h"
#include "fc3.h"

#ifdef QDOT8
#define VARIANT "vqdot8.vx, one output row per vector register"
#else
#define VARIANT "vqdot4.vx, one output row per vector register"
#endif

#ifndef FC_SEL
#define FC_SEL 0
#endif

#if FC_SEL == 0
#include "block0_golden_qkv.h"
#define L(x) B0_QKV_##x
#define G(x) B0_GOLDEN_QKV_##x
#define NAME "QKV"
#elif FC_SEL == 1
#include "block0_golden_proj.h"
#define L(x) B0_PROJ_##x
#define G(x) B0_GOLDEN_PROJ_##x
#define NAME "PROJ"
#elif FC_SEL == 2
#include "block0_golden_fc1.h"
#define L(x) B0_FC1_##x
#define G(x) B0_GOLDEN_FC1_##x
#define NAME "FC1"
#else
#include "block0_golden_fc2.h"
#define L(x) B0_FC2_##x
#define G(x) B0_GOLDEN_FC2_##x
#define NAME "FC2"
#endif

static int8_t out[L(M) * L(N)];
static int8_t ap[200 * L(K)] __attribute__((aligned(8)));
static int8_t wp[L(N) * L(K)] __attribute__((aligned(8)));

int main(void)
{
  const int M = L(M), N = L(N), K = L(K);
  const uint64_t macs = (uint64_t)M * N * K;
  printf("FC %s (%s): M=%d N=%d K=%d, %lu MACs\n", NAME, VARIANT, M, N, K, macs);

  pack_g(L(W), N, K, K, 1, K, wp);                       // offline in a real flow
  uint64_t p0 = rdinstret();
  pack_g(G(IN), M, K, K, 1, K, ap);
  uint64_t i0 = rdinstret(), c0 = rdcycle();
  MARK(1);
  fc_x3(ap, wp, L(BIAS), L(RQP), out, M, N, K);
  MARK(2);
  uint64_t i1 = rdinstret(), c1 = rdcycle();
  printf("activation packing = %lu instr\n", i0 - p0);

  int bad = 0, first = -1;
  for (int i = 0; i < M * N; i++)
    if (out[i] != G(OUT)[i]) { if (first < 0) first = i; bad++; }

  printf("instret = %lu, cycles = %lu, instr/MAC = %lu.%03lu\n",
         i1 - i0, c1 - c0, (i1 - i0) / macs, ((i1 - i0) % macs) * 1000 / macs);
  if (bad)
    printf("FAIL: %d / %d outputs differ; first at m=%d n=%d: got %d, golden %d\n",
           bad, M * N, first / N, first % N, out[first], G(OUT)[first]);
  else
    printf("PASS: all %d outputs bit-exact\n", M * N);
  return bad ? 1 : 0;
}
