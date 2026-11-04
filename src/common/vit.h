// One DeiT-Tiny encoder block in integer arithmetic, shared by src/tests/block* (block 0 with a
// check after every stage) and src/tests/model* (the whole network). Parameters come from headers
// written by scripts/export_block.py; BLOCK_INIT(&blk, Bn) copies block n's into a blk_t, so the
// same code runs all 12 blocks.
//
// Kernels (DESIGN.md D9, D10):
//   -DSCALAR  B1-style scalar FC / BATCH_MATMUL, software requantization (no custom instructions)
//   default   X1: qmac8 FC / BMM with 2x4 tiles, rq / rq.a; -DTILE4 uses 4x4 tiles (option A)
//   X2        (#defined by the including file) outer-product qop4, or qop8 with -DQOP8
//   X3        (#defined with X2 by the including file) row-vector vqdot4.vx, or vqdot8.vx with
//             -DQDOT8; uses the X2 data flow (packed operands, row sums) with the fc3.h layout
// X2 kernels read their int8 operands K-interleaved (pidx); for the other variants pidx is plain
// row-major, so every producer below is the same code for all variants.
//
// Operators other than FC / BMM (the same for all variants; checked against LiteRT by
// export_block.py, which uses the same arithmetic):
//   LayerNorm  MEAN / SUB / MUL / MEAN / ADD eps / RSQRT table / MUL / MUL gamma / ADD beta, fused
//              per row. MEAN and MUL use gemmlowp double rounding (rq.a); ADD and SUB use single
//              rounding through 64-bit tables (qnn.h).
//   attention  QKV FC; Q and K requantized per head (rq.a), V transposed with rows padded to 200;
//              QK^T per head; the MUL by the constant 1/8 is a 256-entry table folded into softmax;
//              softmax uses an integer exp table and one division per row, and recomputes a row with
//              LiteRT's float32 arithmetic (emulated in integers) when an output could be a near-tie.
//   MLP        FC1, GELU table, FC2; two residual ADDs.
#ifndef VIT_H
#define VIT_H
#include "runtime.h"
#include "qnn.h"

typedef uint64_t __attribute__((may_alias)) u64a;

#if defined(X3)
#include "../tests/fc_x3/fc3.h"
#include "../tests/bmm_x3/bmm3.h"
#ifdef QDOT8
#define VARIANT "X3: vqdot8.vx, one output row per vector register"
#else
#define VARIANT "X3: vqdot4.vx, one output row per vector register"
#endif
#elif defined(X2)
#include "../tests/fc_x2/fc2.h"
#include "../tests/bmm_x2/bmm2.h"
#ifdef QOP8
#define VARIANT "X2 prototype: qop8, 8x8 accumulators"
#else
#define VARIANT "X2 prototype: qop4, 4x4 accumulators"
#endif
#else
#include "../tests/bmm/bmm.h"
static inline int pidx(int r, int k, int K) { return r * K + k; }
#define PK1 1
#define PK2 2
#if defined(SCALAR)
#define VARIANT "scalar B1 kernels, software requant"
void fc_i8(const int8_t *in, const int8_t *W, const int32_t *bias, const int32_t *m0,
           const int8_t *shift, int32_t out_zp, int8_t *out, int M, int N, int K);
#elif defined(TILE4)
#define VARIANT "Xqmac: qmac8 + rq / rq.a, 4x4 tiles"
#else
#define VARIANT "Xqmac: qmac8 + rq / rq.a"
#endif
#ifndef SCALAR
void fc_i8(const int8_t *in, const int8_t *W, const int32_t *bias, const int32_t *m0,
           const int8_t *shift, int32_t out_zp, const uint64_t *rqp, int8_t *out, int M, int N, int K);
#endif
#endif
#ifndef PSTEP
#define PSTEP(k) PK2                // offset from element pair k to pair k + 2 (k even)
#endif

#define T   197                     // tokens
#define C   192                     // embedding width
#define H   3                       // heads
#define D   64                      // head dimension
#define TP  200                     // T rounded up to a multiple of 8
#define F   768                     // MLP hidden width

static int32_t row_sum(const int8_t *x, int n)
{
#ifdef SCALAR
  int32_t s = 0;
  for (int i = 0; i < n; i++)
    s += x[i];
  return s;
#else
  const u64a *v = (const u64a *)x;
  int64_t s = 0;
  for (int i = 0; i < n / 8; i++)
    s = qmac8(s, v[i], 0x0101010101010101ull);
  return (int32_t)s;
#endif
}

// ---- parameters ----

typedef struct {                    // one FULLY_CONNECTED (or the patch-embedding convolution)
  const int8_t *W;                  // [N][K] row-major, or packed (X2)
  const int32_t *bias, *m0;
  const int8_t *shift;
  const uint64_t *rqp;
  int32_t zo;
  int M, N, K;
} fc_t;

typedef struct { int32_t za, zb, m0, sh, zo; uint64_t rqp; } bmm_t;

typedef struct {                    // one int8 ADD / SUB: y = clamp((a[x] + b[y]) >> ts)
  int64_t a[256], b[256];
  int ts;
} add_t;

typedef struct {                    // one LayerNorm (9 TFLite ops)
  rq_t mean, sq, var, norm, gamma;
  int32_t zx, zc, zs, zn, zr;
  int ts_sub, ts_eps, ts_beta;
  int64_t leps;                     // eps through its ADD input stage
  const int8_t *rsqrt;
  int64_t lx[256], lmu[256], lv[256], lg[256];
  int32_t gm[C];                    // gamma - zp
  int64_t lbeta[C];                 // beta through its ADD input stage
} ln_t;

typedef struct {
  ln_t ln0, ln1;
  add_t add1, add2;                 // residual ADDs
  fc_t qkv, proj, fc1, fc2;
  bmm_t qk, av;
  rq_t rq_q, rq_k, rq_v;            // per-head requantization of Q, K, V (SQUEEZE)
  int32_t zq, zk, zv;
  int v_ident;                      // V's requantization is the identity
  int8_t scale34[256];              // op 34: scores * (1/sqrt(64)), as a table
  const uint32_t *exp, *expf;       // softmax tables: integer at 2^24, and float32 bits
  uint32_t sm_scale;                // softmax output scale, float32 bits
  int32_t sm_zo;
  const int8_t *gelu;
#ifdef X2
  int8_t w_qkv[3 * C * C], w_proj[C * C], w_fc1[F * C], w_fc2[C * F];   // packed weights
#endif
} blk_t;

static void ln_build(ln_t *L, int32_t sub_z0, int32_t sub_m0, int sub_s0, int32_t sub_z1,
                     int32_t sub_m1, int sub_s1, int32_t sub_mo, int sub_so, int32_t sub_zo,
                     int32_t eps_z0, int32_t eps_m0, int eps_s0, int32_t eps_z1, int32_t eps_m1,
                     int eps_s1, int32_t eps_mo, int eps_so, int32_t eps_zo, int eps_const,
                     int32_t bet_z0, int32_t bet_m0, int bet_s0, int32_t bet_z1, int32_t bet_m1,
                     int bet_s1, int32_t bet_mo, int bet_so, int32_t bet_zo, const int8_t *beta,
                     const int8_t *gamma, int32_t gamma_zp)
{
  L->ts_sub = 31 - sub_so;
  addsub_table(L->lx, sub_z0, sub_m0, sub_s0, sub_mo, add_bias(L->ts_sub, sub_zo));
  addsub_table(L->lmu, sub_z1, sub_m1, sub_s1, -(int64_t)sub_mo, 0);
  L->ts_eps = 31 - eps_so;
  addsub_table(L->lv, eps_z0, eps_m0, eps_s0, eps_mo, add_bias(L->ts_eps, eps_zo));
  L->leps = (int64_t)addsub_stage(eps_const, eps_z1, eps_m1, eps_s1) * eps_mo;
  L->ts_beta = 31 - bet_so;
  addsub_table(L->lg, bet_z0, bet_m0, bet_s0, bet_mo, add_bias(L->ts_beta, bet_zo));
  for (int i = 0; i < C; i++) {
    L->gm[i] = gamma[i] - gamma_zp;
    L->lbeta[i] = (int64_t)addsub_stage(beta[i], bet_z1, bet_m1, bet_s1) * bet_mo;
  }
}

// ops a0..a8 of one LayerNorm, in block-0 numbering: MEAN SUB MUL MEAN ADD RSQRT MUL MUL ADD
#define P(pre, k, x) pre##_OP##k##_##x
#define LN_INIT(L, pre, a0, a1, a2, a3, a4, a5, a6, a7, a8) do {                                   \
    rq_t r_[5] = { RQ(P(pre, a0, MEAN_M0), P(pre, a0, MEAN_SHIFT), P(pre, a0, MEAN_OUT_ZP)),     \
                   RQ(P(pre, a2, MUL_M0), P(pre, a2, MUL_SHIFT), P(pre, a2, MUL_OUT_ZP)),        \
                   RQ(P(pre, a3, MEAN_M0), P(pre, a3, MEAN_SHIFT), P(pre, a3, MEAN_OUT_ZP)),     \
                   RQ(P(pre, a6, MUL_M0), P(pre, a6, MUL_SHIFT), P(pre, a6, MUL_OUT_ZP)),        \
                   RQ(P(pre, a7, MUL_M0), P(pre, a7, MUL_SHIFT), P(pre, a7, MUL_OUT_ZP)) };      \
    (L)->mean = r_[0]; (L)->sq = r_[1]; (L)->var = r_[2]; (L)->norm = r_[3]; (L)->gamma = r_[4]; \
    (L)->zx = P(pre, a0, MEAN_IN0_ZP); (L)->zc = P(pre, a2, MUL_IN0_ZP);                         \
    (L)->zs = P(pre, a3, MEAN_IN0_ZP); (L)->zr = P(pre, a6, MUL_IN1_ZP);                         \
    (L)->zn = P(pre, a7, MUL_IN0_ZP); (L)->rsqrt = P(pre, a5, RSQRT_LUT);                        \
    ln_build((L), P(pre, a1, SUB_IN0_ZP), P(pre, a1, SUB_IN0_M0), P(pre, a1, SUB_IN0_SHIFT),     \
             P(pre, a1, SUB_IN1_ZP), P(pre, a1, SUB_IN1_M0), P(pre, a1, SUB_IN1_SHIFT),          \
             P(pre, a1, SUB_OUT_M0), P(pre, a1, SUB_OUT_SHIFT), P(pre, a1, SUB_OUT_ZP),          \
             P(pre, a4, ADD_IN0_ZP), P(pre, a4, ADD_IN0_M0), P(pre, a4, ADD_IN0_SHIFT),          \
             P(pre, a4, ADD_IN1_ZP), P(pre, a4, ADD_IN1_M0), P(pre, a4, ADD_IN1_SHIFT),          \
             P(pre, a4, ADD_OUT_M0), P(pre, a4, ADD_OUT_SHIFT), P(pre, a4, ADD_OUT_ZP),          \
             P(pre, a4, ADD_IN1_CONST)[0],                                                       \
             P(pre, a8, ADD_IN0_ZP), P(pre, a8, ADD_IN0_M0), P(pre, a8, ADD_IN0_SHIFT),          \
             P(pre, a8, ADD_IN1_ZP), P(pre, a8, ADD_IN1_M0), P(pre, a8, ADD_IN1_SHIFT),          \
             P(pre, a8, ADD_OUT_M0), P(pre, a8, ADD_OUT_SHIFT), P(pre, a8, ADD_OUT_ZP),          \
             P(pre, a8, ADD_IN1_CONST), P(pre, a7, MUL_IN1_CONST), P(pre, a7, MUL_IN1_ZP));      \
  } while (0)

static void add_build(add_t *A, int32_t z0, int32_t m0, int s0, int32_t z1, int32_t m1, int s1,
                      int32_t mo, int so, int32_t zo)
{
  A->ts = 31 - so;
  addsub_table(A->a, z0, m0, s0, mo, add_bias(A->ts, zo));
  addsub_table(A->b, z1, m1, s1, mo, 0);
}

#define ADD_INIT(A, pre, k)                                                                       \
  add_build((A), P(pre, k, ADD_IN0_ZP), P(pre, k, ADD_IN0_M0), P(pre, k, ADD_IN0_SHIFT),         \
            P(pre, k, ADD_IN1_ZP), P(pre, k, ADD_IN1_M0), P(pre, k, ADD_IN1_SHIFT),              \
            P(pre, k, ADD_OUT_M0), P(pre, k, ADD_OUT_SHIFT), P(pre, k, ADD_OUT_ZP))

#ifdef X2
#define FC_WEIGHTS(f, w_, wbuf, n, k) (pack_g((w_), (n), (k), (k), 1, (k), (wbuf)), (f)->W = (wbuf))
#else
#define FC_WEIGHTS(f, w_, wbuf, n, k) ((void)(wbuf), (f)->W = (w_))
#endif
// one FC from its generated names (pre##_##role##_W, ...); X2 packs W into wbuf (offline in a
// real flow, so callers keep it out of the measured setup)
#define FC_INIT(f, pre, role, wbuf, rows) do {                                                    \
    FC_WEIGHTS((f), pre##_##role##_W, wbuf, pre##_##role##_N, pre##_##role##_K);                 \
    (f)->bias = pre##_##role##_BIAS; (f)->m0 = pre##_##role##_M0;                               \
    (f)->shift = pre##_##role##_SHIFT; (f)->rqp = pre##_##role##_RQP;                           \
    (f)->zo = pre##_##role##_OUT_ZP; (f)->M = (rows);                                           \
    (f)->N = pre##_##role##_N; (f)->K = pre##_##role##_K;                                       \
  } while (0)

#define BMM_INIT(b, pre, role) do {                                                              \
    (b)->za = pre##_##role##_LHS_ZP; (b)->zb = pre##_##role##_RHS_ZP;                           \
    (b)->m0 = pre##_##role##_M0; (b)->sh = pre##_##role##_SHIFT;                                \
    (b)->zo = pre##_##role##_OUT_ZP; (b)->rqp = pre##_##role##_RQP;                             \
  } while (0)

static int fails;

static void scale_table(int8_t *t, int32_t zi, int32_t c, rq_t r)
{
  for (int q = -128; q < 128; q++)
    t[q & 0xff] = (int8_t)RQA((q - zi) * c, r);
  for (int q = -128; q < 127; q++)
    if (t[(q + 1) & 0xff] < t[q & 0xff]) {                     // softmax relies on this
      printf("  scale table not monotonic\n");
      fails++;
      break;
    }
}

// Tables and parameters of one block (the "setup" stage); W packing is done by BLOCK_WEIGHTS.
#define BLOCK_INIT(B, pre) do {                                                                  \
    LN_INIT(&(B)->ln0, pre, 10, 11, 12, 13, 14, 15, 16, 17, 18);                                 \
    LN_INIT(&(B)->ln1, pre, 42, 43, 44, 45, 46, 47, 48, 49, 50);                                 \
    ADD_INIT(&(B)->add1, pre, 41);                                                               \
    ADD_INIT(&(B)->add2, pre, 54);                                                               \
    rq_t r34_ = RQ(P(pre, 34, MUL_M0), P(pre, 34, MUL_SHIFT), P(pre, 34, MUL_OUT_ZP));           \
    scale_table((B)->scale34, P(pre, 34, MUL_IN0_ZP),                                            \
                P(pre, 34, MUL_IN1_CONST)[0] - P(pre, 34, MUL_IN1_ZP), r34_);                    \
    rq_t rq_[3] = { RQ(P(pre, 23, REQUANT_M0), P(pre, 23, REQUANT_SHIFT), P(pre, 23, REQUANT_OUT_ZP)), \
                    RQ(P(pre, 26, REQUANT_M0), P(pre, 26, REQUANT_SHIFT), P(pre, 26, REQUANT_OUT_ZP)), \
                    RQ(P(pre, 29, REQUANT_M0), P(pre, 29, REQUANT_SHIFT), P(pre, 29, REQUANT_OUT_ZP)) }; \
    (B)->rq_q = rq_[0]; (B)->rq_k = rq_[1]; (B)->rq_v = rq_[2];                                 \
    (B)->zq = P(pre, 23, REQUANT_IN_ZP); (B)->zk = P(pre, 26, REQUANT_IN_ZP);                    \
    (B)->zv = P(pre, 29, REQUANT_IN_ZP); (B)->v_ident = P(pre, 29, REQUANT_IDENTITY);            \
    (B)->exp = P(pre, 35, SOFTMAX_EXP); (B)->expf = P(pre, 35, SOFTMAX_EXPF);                    \
    (B)->sm_scale = P(pre, 35, SOFTMAX_OUT_SCALE_F32); (B)->sm_zo = P(pre, 35, SOFTMAX_OUT_ZP);  \
    (B)->gelu = P(pre, 52, GELU_LUT);                                                           \
    BMM_INIT(&(B)->qk, pre, QK);                                                                 \
    BMM_INIT(&(B)->av, pre, AV);                                                                 \
  } while (0)

#ifdef X2
#define WBUF(B, w) (B)->w
#else
#define WBUF(B, w) 0
#endif
#define BLOCK_WEIGHTS(B, pre) do {                                                               \
    FC_INIT(&(B)->qkv, pre, QKV, WBUF(B, w_qkv), T);                                             \
    FC_INIT(&(B)->proj, pre, PROJ, WBUF(B, w_proj), T);                                          \
    FC_INIT(&(B)->fc1, pre, FC1, WBUF(B, w_fc1), T);                                             \
    FC_INIT(&(B)->fc2, pre, FC2, WBUF(B, w_fc2), T);                                             \
  } while (0)

// ---- buffers (rows padded to TP, so packed operands always hold whole tiles of zeros) ----
#define BUF(name, n) static int8_t name[n] __attribute__((aligned(8)))
BUF(ln0_p, TP * C);                     // LayerNorm 0 output = QKV input
BUF(qkv, T * 3 * C);
BUF(q_p, H * TP * D);                   // per head [T][D]
BUF(k_p, H * TP * D);
BUF(v_p, H * D * TP);                   // per head V^T [D][TP]
BUF(att, T * TP);                       // scores of one head, row-major
BUF(p_p, TP * TP);                      // probabilities of one head = AV input
BUF(attn_p, TP * C);                    // AV output = PROJ input
BUF(proj, T * C);
BUF(res1, T * C);
BUF(ln1_p, TP * C);                     // LayerNorm 1 output = FC1 input
BUF(hid, T * F);                        // FC1 output
BUF(hid_p, TP * F);                     // GELU output = FC2 input
BUF(fc2o, T * C);
#ifdef X2
static int32_t q_s[H * TP], k_s[H * TP], v_s[H * D], p_s[TP];   // row sums for the zero points
#endif

// ---- operators ----

static inline void fc(const fc_t *f, const int8_t *in, int8_t *out)
{
#if defined(X2)
  fc_x2(in, f->W, f->bias, f->rqp, out, f->M, f->N, f->K);
#elif defined(SCALAR)
  fc_i8(in, f->W, f->bias, f->m0, f->shift, f->zo, out, f->M, f->N, f->K);
#else
  fc_i8(in, f->W, f->bias, f->m0, f->shift, f->zo, f->rqp, out, f->M, f->N, f->K);
#endif
}

// LayerNorm of `rows` rows of x [rows][C]; output in pidx layout (the next FC's operand)
__attribute__((noinline))
static void layernorm(const ln_t *L, const int8_t *x, int8_t *yp, int rows)
{
  const rq_t mean = L->mean, sq = L->sq, var = L->var, norm = L->norm, gamma = L->gamma;
  const int32_t zx = L->zx, zc = L->zc, zs = L->zs, zn = L->zn, zr = L->zr;
  const int ts_sub = L->ts_sub, ts_beta = L->ts_beta;
  int8_t c[C];
  for (int t = 0; t < rows; t++, x += C) {
    int32_t mu = RQA(row_sum(x, C) - C * zx, mean);                       // MEAN
    int64_t lmu = L->lmu[mu & 0xff];
    int32_t ssum = 0;
    for (int i = 0; i < C; i++) {
      int32_t ci = ADDOUT(L->lx[x[i] & 0xff] + lmu, ts_sub);              // SUB: x - mean
      c[i] = (int8_t)ci;
      ssum += RQA((ci - zc) * (ci - zc), sq);                             // MUL: squared
    }
    int32_t v = RQA(ssum - C * zs, var);                                  // MEAN: variance
    int32_t ve = ADDOUT(L->lv[v & 0xff] + L->leps, L->ts_eps);            // ADD eps
    int32_t r = L->rsqrt[ve & 0xff] - zr;                                 // RSQRT table
    int8_t *y = yp + pidx(t, 0, C);
#define LN_OUT(i) ADDOUT(L->lg[RQA((RQA((c[i] - zc) * r, norm) - zn) * L->gm[i], gamma) & 0xff] + \
                         L->lbeta[i], ts_beta)                            // * rsqrt, * gamma, + beta
#ifdef X3
    for (int i = 0; i < C; i += QD, y += 8 * QD)                         // one QD-byte slice
#pragma GCC unroll 8
      for (int u = 0; u < QD; u += 2) {
        y[u] = (int8_t)LN_OUT(i + u);
        y[u + 1] = (int8_t)LN_OUT(i + u + 1);
      }
#else
    for (int i = 0; i < C; y += PSTEP(i), i += 2) {
      y[0] = (int8_t)LN_OUT(i);
      y[PK1] = (int8_t)LN_OUT(i + 1);
    }
#endif
#undef LN_OUT
  }
}

__attribute__((noinline))
static void residual_add(const add_t *A, const int8_t *a, const int8_t *b, int8_t *y, int n)
{
  const int ts = A->ts;
  for (int i = 0; i < n; i++)
    y[i] = (int8_t)ADDOUT(A->a[a[i] & 0xff] + A->b[b[i] & 0xff], ts);
}

// QKV is [T][q/k/v][H][D]. Per head: q and k become [T][D] (rows t), v becomes [D][TP] (rows d,
// K = t), in pidx layout; X2 also gets their row sums.
static inline __attribute__((always_inline))
void split_body(const blk_t *B, const int8_t *qkv, const int vid)
{
  const rq_t rq0 = B->rq_q, rq1 = B->rq_k, rq2 = B->rq_v;
  const int32_t zq = B->zq, zk = B->zk, zv = B->zv;
#ifdef X2
  for (int i = 0; i < H * D; i++)
    v_s[i] = 0;
#endif
  for (int t = 0; t < T; t++, qkv += 3 * C)
    for (int h = 0; h < H; h++) {
      const int8_t *s = qkv + h * D;
      int8_t *dq = q_p + h * TP * D + pidx(t, 0, D), *dk = k_p + h * TP * D + pidx(t, 0, D);
      int8_t *dv = v_p + h * D * TP + pidx(0, t, TP);
#ifdef X2
      int32_t *vsum = v_s + h * D, sq = 0, sk = 0;
#endif
#ifdef X3
      for (int d0 = 0; d0 < D; d0 += QD, dq += 8 * QD, dk += 8 * QD) {
      int8_t *dvg = dv + pidx(d0, 0, TP);                  // rows d0 .. d0 + QD - 1 of V^T: +QD each
#pragma GCC unroll 8
      for (int u = 0; u < QD; u += 2) {
        const int d = d0 + u;
        int32_t q0 = RQA(s[d] - zq, rq0), q1 = RQA(s[d + 1] - zq, rq0);
        int32_t k0 = RQA(s[C + d] - zk, rq1), k1 = RQA(s[C + d + 1] - zk, rq1);
        int32_t v0 = vid ? s[2 * C + d] : RQA(s[2 * C + d] - zv, rq2);
        int32_t v1 = vid ? s[2 * C + d + 1] : RQA(s[2 * C + d + 1] - zv, rq2);
        dq[u] = (int8_t)q0; dq[u + 1] = (int8_t)q1;
        dk[u] = (int8_t)k0; dk[u + 1] = (int8_t)k1;
        dvg[u * QD] = (int8_t)v0; dvg[(u + 1) * QD] = (int8_t)v1;
        sq += q0 + q1; sk += k0 + k1;
        vsum[d] += v0; vsum[d + 1] += v1;
      }
      }
#else
      for (int d = 0; d < D; dq += PSTEP(d), dk += PSTEP(d), d += 2) {
        int32_t q0 = RQA(s[d] - zq, rq0), q1 = RQA(s[d + 1] - zq, rq0);
        int32_t k0 = RQA(s[C + d] - zk, rq1), k1 = RQA(s[C + d + 1] - zk, rq1);
        int32_t v0 = vid ? s[2 * C + d] : RQA(s[2 * C + d] - zv, rq2);
        int32_t v1 = vid ? s[2 * C + d + 1] : RQA(s[2 * C + d + 1] - zv, rq2);
        dq[0] = (int8_t)q0; dq[PK1] = (int8_t)q1;
        dk[0] = (int8_t)k0; dk[PK1] = (int8_t)k1;
        dv[pidx(d, 0, TP)] = (int8_t)v0; dv[pidx(d + 1, 0, TP)] = (int8_t)v1;
#ifdef X2
        sq += q0 + q1; sk += k0 + k1;
        vsum[d] += v0; vsum[d + 1] += v1;
#endif
      }
#endif
#ifdef X2
      q_s[h * TP + t] = sq;
      k_s[h * TP + t] = sk;
#endif
    }
}

__attribute__((noinline))
static void split_qkv(const blk_t *B, const int8_t *qkv)
{
  if (B->v_ident)
    split_body(B, qkv, 1);
  else
    split_body(B, qkv, 0);
}

// op 34 (scale table) + SOFTMAX on T rows of scores s (stride TP); probabilities to p_p (pidx,
// rows t, K = t'), X2 also their row sums. The scale table is non-decreasing (a positive constant;
// checked in setup), so the row maximum is taken on the raw scores. Fast path: e = exp table at
// scale34[s] - max + 255 (exp * 2^24), R = 2^54 / sum, q = e * R / 2^46, y = round(q) + zo. q is
// within g = q*2^-15 + 2^-16 of LiteRT's float32 value (export_block.py, softmax_int), so if
// round(q - g) == round(q + g) for every element the row is exact; otherwise the row is recomputed
// by softmax_row_exact (about 1% of rows).
__attribute__((noinline))
static void softmax(const blk_t *B, const int8_t *s)
{
  int32_t e[T];
  int8_t x[T], y[T];
  const int8_t *sc = B->scale34;
  const int32_t zo = B->sm_zo;
  const int64_t c_hi = ((int64_t)1 << 45) + ((int64_t)1 << 30);         // 1/2 + 2^-16, at 2^46
  const int64_t c_lo = ((int64_t)1 << 45) - ((int64_t)1 << 30);
  for (int t = 0; t < T; t++, s += TP) {
    int32_t mr = -128;
    for (int j = 0; j < T; j++) {
      int32_t v = s[j];
      mr = v > mr ? v : mr;
    }
    const uint32_t *E = B->exp + 255 - sc[mr & 0xff];
    int64_t sum = 0;
    for (int j = 0; j < T; j++)
      sum += e[j] = (int32_t)E[sc[s[j] & 0xff]];
    const int64_t R = ((int64_t)1 << 54) / sum;
    int8_t *o = p_p + pidx(t, 0, TP);
    int32_t rs = 0;
#define SM_Q(j, v) do { int64_t q_ = e[j] * R, a_ = q_ >> 15;                                   \
                        int64_t hi_ = (q_ + a_ + c_hi) >> 46, lo_ = (q_ - a_ + c_lo) >> 46;     \
                        if (__builtin_expect(hi_ != lo_, 0)) goto exact;                       \
                        int32_t w_ = (int32_t)hi_ + zo; (v) = w_ > 127 ? 127 : w_; } while (0)
    int j = 0;
#ifdef X3
    for (; j + QD < T; j += QD, o += 8 * QD)                              // whole QD-byte slices
#pragma GCC unroll 8
      for (int u = 0; u < QD; u += 2) {
        int32_t v0, v1;
        SM_Q(j + u, v0);
        SM_Q(j + u + 1, v1);
        o[u] = (int8_t)v0;
        o[u + 1] = (int8_t)v1;
        rs += v0 + v1;
      }
#endif
    for (; j + 1 < T; o += PSTEP(j), j += 2) {
      int32_t v0, v1;
      SM_Q(j, v0);
      SM_Q(j + 1, v1);
      o[0] = (int8_t)v0;
      o[PK1] = (int8_t)v1;
#ifdef X2
      rs += v0 + v1;
#endif
    }
    {
      int32_t v;
      SM_Q(j, v);                                                         // T is odd
      o[0] = (int8_t)v;
      rs += v;
    }
#undef SM_Q
    goto done;
exact:                                                                    // some q is near a tie
    for (j = 0; j < T; j++)
      x[j] = sc[s[j] & 0xff];
    softmax_row_exact(B->expf + 255 - sc[mr & 0xff], x, T, B->sm_scale, zo, y);
    o = p_p + pidx(t, 0, TP);
    rs = 0;
    for (j = 0; j + 1 < T; o += PSTEP(j), j += 2) {
      o[0] = y[j];
      o[PK1] = y[j + 1];
      rs += y[j] + y[j + 1];
    }
    o[0] = y[j];
    rs += y[j];
done:
#ifdef X2
    p_s[t] = rs;
#else
    (void)rs;
#endif
  }
}

// GELU table, row-major [T][F] in, pidx layout out (the operand of FC2)
__attribute__((noinline))
static void gelu(const int8_t *lut, const int8_t *x, int8_t *yp)
{
  for (int t = 0; t < T; t++, x += F) {
    int8_t *y = yp + pidx(t, 0, F);
#ifdef X3
    for (int i = 0; i < F; i += QD, y += 8 * QD)
#pragma GCC unroll 8
      for (int u = 0; u < QD; u++)
        y[u] = lut[x[i + u] & 0xff];
#else
    for (int i = 0; i < F; y += PSTEP(i), i += 2) {
      y[0] = lut[x[i] & 0xff];
      y[PK1] = lut[x[i + 1] & 0xff];
    }
#endif
  }
}

static inline void bmm_qk(const blk_t *B, int h)
{
  const bmm_t *b = &B->qk;
#ifdef X2
  bmm_x2(q_p + h * TP * D, k_p + h * TP * D, T, T, D, D, q_s + h * TP, k_s + h * TP, b->za, b->zb,
         b->rqp, att, TP, 0);
#else
  bmm_i8(q_p + h * TP * D, k_p + h * TP * D, T, T, D, D, b->za, b->zb, b->m0, b->sh, b->zo,
         b->rqp, att, TP);
#endif
}

static inline void bmm_av(const blk_t *B, int h)
{
  const bmm_t *b = &B->av;
#ifdef X2
  bmm_x2(p_p, v_p + h * D * TP, T, D, T, TP, p_s, v_s + h * D, b->za, b->zb, b->rqp,
         attn_p + pidx(0, h * D, C), 0, C);
#else
  bmm_i8(p_p, v_p + h * D * TP, T, D, T, TP, b->za, b->zb, b->m0, b->sh, b->zo, b->rqp,
         attn_p + h * D, C);
#endif
}

// ---- one block, x [T][C] -> y [T][C], with per-stage instruction counts ----

enum { S_LN0, S_QKV, S_SPLIT, S_QK, S_SOFTMAX, S_AV, S_PROJ, S_ADD1, S_LN1, S_FC1, S_GELU,
       S_FC2, S_ADD2, S_BLOCK_N };
static const char *const block_stage_name[S_BLOCK_N] = {
  "LayerNorm 0", "FC QKV", "Q/K/V split", "BMM QK^T", "scale+softmax", "BMM AV", "FC PROJ",
  "residual ADD 1", "LayerNorm 1", "FC1", "GELU", "FC2", "residual ADD 2" };
static const uint64_t block_stage_macs[S_BLOCK_N] = {
  0, (uint64_t)T * C * 3 * C, 0, (uint64_t)H * T * T * D, 0, (uint64_t)H * T * D * T,
  (uint64_t)T * C * C, 0, 0, (uint64_t)T * C * F, 0, (uint64_t)T * F * C, 0 };

#define RUN(acc, stmt) do { uint64_t t0_ = rdinstret(); MARK(100); stmt; MARK(101);              \
                            (acc) += rdinstret() - t0_; } while (0)

// VIT_HOOK(stage, head) runs after each stage, outside the counted region (block tests check
// intermediate tensors there); head is -1 outside the per-head loop.
#ifndef VIT_HOOK
#define VIT_HOOK(stage, head) ((void)0)
#endif

static void vit_block(const blk_t *B, const int8_t *x, int8_t *y, uint64_t st[S_BLOCK_N])
{
  RUN(st[S_LN0], layernorm(&B->ln0, x, ln0_p, T));
  VIT_HOOK(S_LN0, -1);
  RUN(st[S_QKV], fc(&B->qkv, ln0_p, qkv));
  VIT_HOOK(S_QKV, -1);
  RUN(st[S_SPLIT], split_qkv(B, qkv));
  VIT_HOOK(S_SPLIT, -1);
  for (int h = 0; h < H; h++) {
    RUN(st[S_QK], bmm_qk(B, h));
    VIT_HOOK(S_QK, h);
    RUN(st[S_SOFTMAX], softmax(B, att));
    VIT_HOOK(S_SOFTMAX, h);
    RUN(st[S_AV], bmm_av(B, h));
  }
  VIT_HOOK(S_AV, -1);
  RUN(st[S_PROJ], fc(&B->proj, attn_p, proj));
  VIT_HOOK(S_PROJ, -1);
  RUN(st[S_ADD1], residual_add(&B->add1, x, proj, res1, T * C));
  RUN(st[S_LN1], layernorm(&B->ln1, res1, ln1_p, T));
  VIT_HOOK(S_LN1, -1);
  RUN(st[S_FC1], fc(&B->fc1, ln1_p, hid));
  VIT_HOOK(S_FC1, -1);
  RUN(st[S_GELU], gelu(B->gelu, hid, hid_p));
  VIT_HOOK(S_GELU, -1);
  RUN(st[S_FC2], fc(&B->fc2, hid_p, fc2o));
  VIT_HOOK(S_FC2, -1);
  RUN(st[S_ADD2], residual_add(&B->add2, res1, fc2o, y, T * C));
}

// ---- reporting ----

static void pad(const char *s, int w)
{
  int n = 0;
  while (s[n]) n++;
  for (; n < w; n++) printf(" ");
}

// leading spaces that right-align v in a field of w digits
static const char *spaces(uint64_t v, int w)
{
  static const char sp[] = "                ";
  int n = 1;
  while (v >= 10) { v /= 10; n++; }
  return n >= w ? sp + 16 : sp + 16 - (w - n);
}

// one line of the stage table: name, instret, % of total, instr/MAC
static void stage_line(const char *name, uint64_t n, uint64_t total, uint64_t macs)
{
  uint64_t pm = total ? n * 1000 / total : 0;
  printf("%s", name);
  pad(name, 17);
  printf("%s%lu  %s%lu.%lu", spaces(n, 11), n, spaces(pm / 10, 3), pm / 10, pm % 10);
  if (macs)
    printf("   %lu.%03lu", n / macs, (n % macs) * 1000 / macs);
  printf("\n");
}

// ---- checking (outside the measured regions) ----

static void report(const char *what, int bad, int total, int r, int c, int got, int gold)
{
  if (bad) {
    printf("  %s FAIL %d / %d differ; first [%d][%d]: got %d, golden %d\n", what, bad, total, r, c, got, gold);
    fails++;
  }
}

// row-major [rows][cols] with stride ld against a row-major golden
__attribute__((unused)) static void check(const char *what, const int8_t *got, int ld, const int8_t *gold, int rows, int cols)
{
  int bad = 0, fr = 0, fc_ = 0;
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++)
      if (got[r * ld + c] != gold[r * cols + c] && !bad++) { fr = r; fc_ = c; }
  report(what, bad, rows * cols, fr, fc_, got[fr * ld + fc_], gold[fr * cols + fc_]);
}

// [rows][K] in pidx layout (row length Kp) against a golden with element (r, k) at r*rs + k*ks
__attribute__((unused)) static void check_p(const char *what, const int8_t *got, int Kp, const int8_t *gold, int rows, int K, int rs, int ks)
{
  int bad = 0, fr = 0, fc_ = 0;
  for (int r = 0; r < rows; r++)
    for (int k = 0; k < K; k++)
      if (got[pidx(r, k, Kp)] != gold[r * rs + k * ks] && !bad++) { fr = r; fc_ = k; }
  report(what, bad, rows * K, fr, fc_, got[pidx(fr, fc_, Kp)], gold[fr * rs + fc_ * ks]);
}

#endif
