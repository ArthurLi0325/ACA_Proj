// Whole DeiT-Tiny W8A8 inference (all 561 TFLite ops) in integer arithmetic on bare-metal RV64:
// int8 image in, 1000 int8 logits and the predicted class out. The stem output, every block output
// and the logits are compared with LiteRT (outside the counted regions). Run
// `python scripts/export_model.py models/deit_tiny_w8a8.tflite [--input img.npy]` first.
//   make -B TEST=model run OPT="-O2 -DSCALAR"                 scalar baseline (no EXT needed)
//   make -B TEST=model run EXT=xqmac                          X1: qmac8 + rq / rq.a
//   make -B TEST=model run EXT=xqmac OPT="-O2 -DTILE4"        X1 with 4x4 tiles (option A)
//   make -B TEST=model_x2 run EXT=xqmac [OPT="-O2 -DQOP8"]    X2 prototype, qop4 / qop8
//   make -B TEST=model_x3 run EXT=xqmac VEC=1 [OPT="-O2 -DQDOT8"]   X3, vqdot4.vx / vqdot8.vx
//
// Graph (ops):  0 TRANSPOSE NCHW->NHWC, 1 CONV_2D 16x16 stride 16 (patch embedding), 2-4 layout
// only, 5 class token (constant), 6/7 QUANTIZE (requantization of the class token / the patches),
// 8 CONCATENATION, 9 ADD position embedding; 10..549 12 encoder blocks (src/common/vit.h);
// 550..558 final LayerNorm, 559 STRIDED_SLICE (class token row), 560 FULLY_CONNECTED 192->1000.
// The convolution runs as an FC over the 196 patches (im2col straight from the NCHW image, weights
// reordered offline to (c, kh, kw)); the final LayerNorm runs on the class-token row only, the
// one row the slice keeps (LayerNorm is per row, so this is the same result).
#include "stem_params.h"
#include "stem_golden.h"
#include "block0_params.h"
#include "block0_golden.h"
#include "block1_params.h"
#include "block1_golden.h"
#include "block2_params.h"
#include "block2_golden.h"
#include "block3_params.h"
#include "block3_golden.h"
#include "block4_params.h"
#include "block4_golden.h"
#include "block5_params.h"
#include "block5_golden.h"
#include "block6_params.h"
#include "block6_golden.h"
#include "block7_params.h"
#include "block7_golden.h"
#include "block8_params.h"
#include "block8_golden.h"
#include "block9_params.h"
#include "block9_golden.h"
#include "block10_params.h"
#include "block10_golden.h"
#include "block11_params.h"
#include "block11_golden.h"
#include "fin_params.h"
#include "fin_golden.h"
#include "vit.h"

#define NB      12                      // encoder blocks
#define NP      196                     // patches
#define KP      768                     // patch length: 3 x 16 x 16
#define CLASSES 1000

static blk_t blk[NB];
static const int8_t *const golden_out[NB] = {
  B0_GOLDEN_OUT, B1_GOLDEN_OUT, B2_GOLDEN_OUT, B3_GOLDEN_OUT, B4_GOLDEN_OUT, B5_GOLDEN_OUT,
  B6_GOLDEN_OUT, B7_GOLDEN_OUT, B8_GOLDEN_OUT, B9_GOLDEN_OUT, B10_GOLDEN_OUT, B11_GOLDEN_OUT };
static fc_t conv, head;
static add_t add_pos;
static ln_t lnf;
static rq_t rq_cls, rq_patch;
#ifdef X2
static int8_t w_conv[C * KP] __attribute__((aligned(8))), w_head[CLASSES * C] __attribute__((aligned(8)));
#define WB(w) (w)
#else
#define WB(w) 0
#endif

BUF(patches_p, TP * KP);                // im2col output = conv input
BUF(conv_out, NP * C);
BUF(tokens, T * C);                     // [class token; patches] before the position embedding
BUF(xa, T * C);                         // block input / output, ping-pong
BUF(xb, T * C);
BUF(lnf_p, 8 * C);                      // final LayerNorm of the class token = head input
static int8_t logits[CLASSES];

static void setup(void)
{
  BLOCK_INIT(&blk[0], B0);  BLOCK_INIT(&blk[1], B1);  BLOCK_INIT(&blk[2], B2);
  BLOCK_INIT(&blk[3], B3);  BLOCK_INIT(&blk[4], B4);  BLOCK_INIT(&blk[5], B5);
  BLOCK_INIT(&blk[6], B6);  BLOCK_INIT(&blk[7], B7);  BLOCK_INIT(&blk[8], B8);
  BLOCK_INIT(&blk[9], B9);  BLOCK_INIT(&blk[10], B10); BLOCK_INIT(&blk[11], B11);
  ADD_INIT(&add_pos, STEM, 9);
  rq_t a = RQ(STEM_OP6_REQUANT_M0, STEM_OP6_REQUANT_SHIFT, STEM_OP6_REQUANT_OUT_ZP);
  rq_t b = RQ(STEM_OP7_REQUANT_M0, STEM_OP7_REQUANT_SHIFT, STEM_OP7_REQUANT_OUT_ZP);
  rq_cls = a; rq_patch = b;
  LN_INIT(&lnf, FIN, 10, 11, 12, 13, 14, 15, 16, 17, 18);
}

// weights: pointers, or packed copies for X2 (offline in a real flow, not counted)
static void weights(void)
{
  BLOCK_WEIGHTS(&blk[0], B0);  BLOCK_WEIGHTS(&blk[1], B1);  BLOCK_WEIGHTS(&blk[2], B2);
  BLOCK_WEIGHTS(&blk[3], B3);  BLOCK_WEIGHTS(&blk[4], B4);  BLOCK_WEIGHTS(&blk[5], B5);
  BLOCK_WEIGHTS(&blk[6], B6);  BLOCK_WEIGHTS(&blk[7], B7);  BLOCK_WEIGHTS(&blk[8], B8);
  BLOCK_WEIGHTS(&blk[9], B9);  BLOCK_WEIGHTS(&blk[10], B10); BLOCK_WEIGHTS(&blk[11], B11);
  FC_INIT(&conv, STEM, CONV, WB(w_conv), NP);
  FC_INIT(&head, FIN, HEAD, WB(w_head), 1);
}

// ops 0-1 input side: patch p = (py, px) becomes row p of [196][768], k = c*256 + kh*16 + kw
__attribute__((noinline))
static void im2col(const int8_t *img, int8_t *ap)
{
  for (int p = 0; p < NP; p++) {
    const int8_t *base = img + (p / 14) * 16 * 224 + (p % 14) * 16;
    int8_t *y = ap + pidx(p, 0, KP);
    for (int c = 0; c < 3; c++)
      for (int kh = 0; kh < 16; kh++) {
        const int8_t *s = base + (c * 224 + kh) * 224;
#if defined(X3)
        for (int kw = 0; kw < 16; kw += QD, y += 8 * QD)        // 16 / QD whole slices
#pragma GCC unroll 8
          for (int u = 0; u < QD; u++)
            y[u] = s[kw + u];
#elif defined(X2)
        for (int kw = 0; kw < 16; y += PSTEP(kw), kw += 2) {
          y[0] = s[kw];
          y[PK1] = s[kw + 1];
        }
#else
        ((u64a *)y)[0] = ((const u64a *)s)[0];                // 16 contiguous bytes
        ((u64a *)y)[1] = ((const u64a *)s)[1];
        y += 16;
#endif
      }
  }
}

// ops 5-9: class token and patches requantized to the common scale (QUANTIZE), concatenated,
// plus the position embedding (ADD with a constant)
__attribute__((noinline))
static void embed(const int8_t *patches, int8_t *y)
{
  const rq_t rc = rq_cls, rp = rq_patch;
  for (int i = 0; i < C; i++)
    tokens[i] = (int8_t)RQA(STEM_OP5_BROADCAST_CONST[i] - STEM_OP6_REQUANT_IN_ZP, rc);
  for (int i = 0; i < NP * C; i++)
    tokens[C + i] = (int8_t)RQA(patches[i] - STEM_OP7_REQUANT_IN_ZP, rp);
  residual_add(&add_pos, tokens, STEM_OP9_ADD_IN1_CONST, y, T * C);
}

static int argmax(const int8_t *v, int n)
{
  int best = 0;
  for (int i = 1; i < n; i++)
    if (v[i] > v[best]) best = i;
  return best;
}

enum { M_IM2COL, M_CONV, M_EMBED, M_LNF, M_HEAD, M_N };

int main(void)
{
  printf("DeiT-Tiny W8A8, whole model (%s), input %s\n", VARIANT, STEM_INPUT);
  weights();
  uint64_t c0 = rdcycle(), t_setup = 0, st[S_BLOCK_N] = {0}, ms[M_N] = {0}, per_block[NB] = {0};

  RUN(t_setup, setup());

  RUN(ms[M_IM2COL], im2col(STEM_GOLDEN_IN, patches_p));
  RUN(ms[M_CONV], fc(&conv, patches_p, conv_out));
  RUN(ms[M_EMBED], embed(conv_out, xa));
  check("stem output", xa, C, STEM_GOLDEN_OUT, T, C);

  int8_t *x = xa, *y = xb;
  for (int b = 0; b < NB; b++) {
    uint64_t before = 0, after = 0;
    for (int i = 0; i < S_BLOCK_N; i++) before += st[i];
    vit_block(&blk[b], x, y, st);
    for (int i = 0; i < S_BLOCK_N; i++) after += st[i];
    per_block[b] = after - before;
    if (fails == 0) {                                 // report the first block that differs
      check("block output", y, C, golden_out[b], T, C);
      if (fails) printf("  (block %d)\n", b);
    }
    int8_t *t = x; x = y; y = t;
  }

  RUN(ms[M_LNF], layernorm(&lnf, x, lnf_p, 1));     // class-token row only (op 559 keeps it)
  RUN(ms[M_HEAD], fc(&head, lnf_p, logits));
  uint64_t c1 = rdcycle();
  check("logits", logits, CLASSES, FIN_GOLDEN_OUT, 1, CLASSES);

  // ---- report ----
  const uint64_t conv_macs = (uint64_t)NP * C * KP, head_macs = (uint64_t)C * CLASSES;
  uint64_t total = 0, macs = conv_macs + head_macs;
  for (int i = 0; i < M_N; i++) total += ms[i];
  for (int i = 0; i < S_BLOCK_N; i++) { total += st[i]; macs += NB * block_stage_macs[i]; }
  printf("stage (12 blocks summed) instret     %%   instr/MAC\n");
  stage_line("patch im2col", ms[M_IM2COL], total, 0);
  stage_line("patch conv (FC)", ms[M_CONV], total, conv_macs);
  stage_line("cls+requant+pos", ms[M_EMBED], total, 0);
  for (int i = 0; i < S_BLOCK_N; i++)
    stage_line(block_stage_name[i], st[i], total, NB * block_stage_macs[i]);
  stage_line("final LayerNorm", ms[M_LNF], total, 0);
  stage_line("head FC", ms[M_HEAD], total, head_macs);
  printf("total            %s%lu  (%lu MACs)\n", spaces(total, 11), total, macs);
  printf("per block:");
  for (int b = 0; b < NB; b++)
    printf(" %lu", per_block[b]);
  printf("\nsetup (tables, once per model): %lu; cycles incl. setup and checks: %lu\n", t_setup, c1 - c0);

  int cls = argmax(logits, CLASSES), gold = argmax(FIN_GOLDEN_OUT, CLASSES);
  printf("predicted class %d (logit %d), LiteRT %d\n", cls, logits[cls], gold);
  printf("RESULT variant=\"%s\" instret=%lu class=%d golden_class=%d exact=%d\n",
         VARIANT, total, cls, gold, fails == 0);
  if (fails)
    printf("FAIL: differs from LiteRT\n");
  else
    printf("PASS: stem, all 12 block outputs and the 1000 logits bit-exact\n");
  return fails ? 1 : 0;
}
