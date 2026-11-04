// Runs the Xqmac fc_i8() (fc.c) on one block-0 FC layer and compares every output with the
// LiteRT golden from scripts/export_block.py (run `make export` first).
//   make TEST=fc_x1 run EXT=xqmac                         QKV, qmac8 + software requant (X1)
//   make TEST=fc_x1 run EXT=xqmac OPT="-O2 -DUSE_DOT8"    dot8 + addw instead of qmac8
//   make TEST=fc_x1 run EXT=xqmac OPT="-O2 -DUSE_RQ"      qmac8 + rq instruction (X1+RQ)
//   add -DFC_SEL=n to OPT for another layer: 0 QKV, 1 PROJ, 2 FC1, 3 FC2
//   make TEST=fc_x1 hist EXT=xqmac HIST_ARGS="--funcs fc_i8,tile_2x4,tile_1x4,requant,srdhm --macs <M*N*K>"
#include "runtime.h"
#include "block0_params.h"
#include "fc.h"

#if defined(USE_RQ)
#define VARIANT "qmac8 + rq"
#elif defined(USE_DOT8)
#define VARIANT "dot8 + addw, software requant"
#else
#define VARIANT "qmac8, software requant"
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

int main(void)
{
  const int M = L(M), N = L(N), K = L(K);
  const uint64_t macs = (uint64_t)M * N * K;
  printf("FC %s (%s): M=%d N=%d K=%d, %lu MACs\n", NAME, VARIANT, M, N, K, macs);

  uint64_t i0 = rdinstret(), c0 = rdcycle();
  MARK(1);
  fc_i8(G(IN), L(W), L(BIAS), L(M0), L(SHIFT), L(OUT_ZP), L(RQP), out, M, N, K);
  MARK(2);
  uint64_t i1 = rdinstret(), c1 = rdcycle();

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
