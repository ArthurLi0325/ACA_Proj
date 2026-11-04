// One whole DeiT-Tiny encoder block (block 0, TFLite ops 10..54) in integer arithmetic, run from
// the block input and compared with LiteRT's output and with every intermediate tensor on the way
// (run `make export` first). The block itself is src/common/vit.h, shared with the whole-model
// program src/tests/model.
//   make -B TEST=block run EXT=xqmac              qmac8 (FC, BMM, row sums) + rq / rq.a
//   make -B TEST=block run OPT="-O2 -DSCALAR"     B1-style scalar baseline (no EXT needed)
//   make -B TEST=block run EXT=xqmac OPT="-O2 -DTILE4"   Xqmac with 4x4 FC/BMM tiles (option A)
//   make -B TEST=block_x2 run EXT=xqmac [OPT="-O2 -DQOP8"]   the X2 prototype (same source)
//   make -B TEST=block hist EXT=xqmac             instruction mix per function
#include "block0_params.h"
#include "block0_golden.h"
#include "block0_golden_qkv.h"
#include "block0_golden_qk.h"
#include "block0_golden_av.h"
#include "block0_golden_proj.h"
#include "block0_golden_fc1.h"
#include "block0_golden_fc2.h"

static void hook(int stage, int h);
#define VIT_HOOK(stage, h) hook((stage), (h))
#include "vit.h"

static blk_t blk;
BUF(out, T * C);

static void hook(int stage, int h)
{
  switch (stage) {
  case S_LN0: check_p("LayerNorm 0", ln0_p, C, B0_GOLDEN_QKV_IN, T, C, C, 1); break;
  case S_QKV: check("FC QKV", qkv, 3 * C, B0_GOLDEN_QKV_OUT, T, 3 * C); break;
  case S_SPLIT:
    for (int i = 0; i < H; i++) {
      check_p("Q", q_p + i * TP * D, D, B0_GOLDEN_QK_LHS + i * T * D, T, D, D, 1);
      check_p("K", k_p + i * TP * D, D, B0_GOLDEN_QK_RHS + i * T * D, T, D, D, 1);
      check_p("V", v_p + i * D * TP, TP, B0_GOLDEN_AV_RHS + i * T * D, D, T, 1, D);
    }
    break;
  case S_QK: check("BMM QK^T", att, TP, B0_GOLDEN_QK_OUT + h * T * T, T, T); break;
  case S_SOFTMAX: check_p("softmax", p_p, TP, B0_GOLDEN_AV_LHS + h * T * T, T, T, T, 1); break;
  case S_AV: check_p("BMM AV", attn_p, C, B0_GOLDEN_PROJ_IN, T, C, C, 1); break;
  case S_PROJ: check("FC PROJ", proj, C, B0_GOLDEN_PROJ_OUT, T, C); break;
  case S_LN1: check_p("LayerNorm 1", ln1_p, C, B0_GOLDEN_FC1_IN, T, C, C, 1); break;
  case S_FC1: check("FC1", hid, F, B0_GOLDEN_FC1_OUT, T, F); break;
  case S_GELU: check_p("GELU", hid_p, F, B0_GOLDEN_FC2_IN, T, F, F, 1); break;
  case S_FC2: check("FC2", fc2o, C, B0_GOLDEN_FC2_OUT, T, C); break;
  }
}

int main(void)
{
  printf("Block 0, ops 10..54 (%s)\n", VARIANT);
  BLOCK_WEIGHTS(&blk, B0);                // X2: weights packed here (offline in a real flow), not counted
  uint64_t c0 = rdcycle(), setup = 0, st[S_BLOCK_N] = {0};

  RUN(setup, BLOCK_INIT(&blk, B0));
  vit_block(&blk, B0_GOLDEN_IN, out, st);
  check("block output", out, C, B0_GOLDEN_OUT, T, C);
  uint64_t c1 = rdcycle();

  uint64_t total = setup, macs = 0;
  for (int i = 0; i < S_BLOCK_N; i++) { total += st[i]; macs += block_stage_macs[i]; }
  printf("stage                instret      %%   instr/MAC\n");
  stage_line("tables (setup)", setup, total, 0);
  for (int i = 0; i < S_BLOCK_N; i++)
    stage_line(block_stage_name[i], st[i], total, block_stage_macs[i]);
  printf("total            %s%lu  (%lu MACs in FC/BMM)\n", spaces(total, 11), total, macs);
  printf("cycles, checks included: %lu\n", c1 - c0);
  if (fails)
    printf("FAIL: %d checks differ from LiteRT\n", fails);
  else
    printf("PASS: block output and all intermediate checks bit-exact\n");
  return fails ? 1 : 0;
}
