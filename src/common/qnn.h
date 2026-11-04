// Integer helpers shared by the block and whole-model programs (src/common/vit.h): TFLite
// requantization (both rounding schemes LiteRT uses), the tables of int8 ADD/SUB, and an integer
// emulation of the float32 arithmetic of LiteRT's x86 SOFTMAX for rows whose integer result could
// round differently. Everything here is integer code: scripts/check_no_fp.sh still passes.
#ifndef QNN_H
#define QNN_H
#include <stdint.h>
#ifndef SCALAR
#include "xqmac.h"
#endif

// ---- requantization: clamp(MultiplyByQuantizedMultiplier(acc) + zo) ----

typedef struct { int32_t m0, sh, zo; uint64_t p; } rq_t;
#define RQ(m0, sh, zo) { (m0), (sh), (zo), (uint32_t)(m0) | ((uint64_t)((sh) & 0x3f) << 32) | \
                                          ((uint64_t)((zo) & 0xff) << 40) }

static inline int32_t srdhm(int32_t a, int32_t b)    // b = M0 > 0, so no saturation case
{
  int64_t ab = (int64_t)a * b;
  int64_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
  return (int32_t)((ab + nudge) / ((int64_t)1 << 31));
}

static inline int32_t rdpot(int32_t x, int e)        // gemmlowp RoundingDivideByPOT
{
  int32_t mask = (1 << e) - 1;
  int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> e) + ((x & mask) > threshold ? 1 : 0);
}

// gemmlowp double rounding: SRDHM, then a right shift with ties away from zero (MUL, MEAN, BMM)
static inline int32_t mbqm(int32_t x, int32_t m0, int sh)
{
  return rdpot(srdhm(x * (1 << (sh > 0 ? sh : 0)), m0), sh > 0 ? 0 : -sh);
}

// TFLITE_SINGLE_ROUNDING: one rounding of x * m0 / 2^(31 - sh), ties toward +inf (ADD, SUB)
static inline int32_t mbqm_single(int32_t x, int32_t m0, int sh)
{
  int ts = 31 - sh;
  return (int32_t)(((int64_t)x * m0 + ((int64_t)1 << (ts - 1))) >> ts);
}

static inline int32_t clamp8(int32_t y)
{
  return y < -128 ? -128 : y > 127 ? 127 : y;
}

#ifdef SCALAR
static inline int32_t rqa(int32_t acc, int32_t m0, int sh, int32_t zo, uint64_t p)
{
  (void)p;
  return clamp8(mbqm(acc, m0, sh) + zo);
}
#define CLAMP8(v) clamp8(v)
#else
static inline int32_t rqa(int32_t acc, int32_t m0, int sh, int32_t zo, uint64_t p)
{
  (void)m0; (void)sh; (void)zo;
  return rq_a(acc, p);
}
// rq with M0 = 2^30, shift = +1, zp = 0 is the identity followed by the int8 clamp
#define RQ_CLAMP (((uint64_t)1 << 30) | ((uint64_t)1 << 32))
#define CLAMP8(v) rq((v), RQ_CLAMP)
#endif
// r is a local copy of an rq_t (stores to int8 buffers would otherwise force reloads)
#define RQA(acc, r) rqa((acc), (r).m0, (r).sh, (r).zo, (r).p)

// ---- int8 ADD / SUB ----
// LiteRT: a' = single((a - za) << 20, ma), b' = single((b - zb) << 20, mb),
//         y  = clamp(single(a' +- b', mo, eo) + zo)  with  single(s, mo, eo) = (s*mo + 2^(ts-1)) >> ts,
// ts = 31 - eo. The output product distributes over the sum, so a 256-entry int64 table per input
// holds a'*mo (plus the rounding term and zo << ts in the first one), and each output is
// clamp((A[a] + B[b]) >> ts): two loads, an add, a shift and a clamp, and exactly LiteRT's result.
static inline int32_t addsub_stage(int q, int32_t zp, int32_t m0, int sh)
{
  return mbqm_single((q - zp) * (1 << 20), m0, sh);
}

static void addsub_table(int64_t *t, int32_t zp, int32_t m0, int sh, int64_t mo, int64_t bias)
{
  for (int q = -128; q < 128; q++)
    t[q & 0xff] = (int64_t)addsub_stage(q, zp, m0, sh) * mo + bias;
}

static inline int64_t add_bias(int ts, int32_t zo)
{
  return ((int64_t)1 << (ts - 1)) + ((int64_t)zo << ts);
}

#define ADDOUT(s, ts) CLAMP8((int32_t)((s) >> (ts)))

// ---- float32 emulation for the exact softmax path ----
// Only what SOFTMAX needs: non-negative finite floats, round to nearest even, as IEEE bit patterns.

static inline int msb64(uint64_t m)                  // m > 0
{
  return 63 - __builtin_clzll(m);
}

// float bits of round(m * 2^e), m > 0
static uint32_t f32_round(uint64_t m, int e)
{
  int p = msb64(m);
  int drop = p - 23;                                  // keep 24 significant bits
  if (e + drop < -149) drop = -149 - e;               // subnormal: granularity 2^-149
  uint64_t q;
  if (drop <= 0) {
    q = m << -drop;
  } else if (drop > 63) {
    q = 0;
  } else {
    uint64_t rem = m & (((uint64_t)1 << drop) - 1), half = (uint64_t)1 << (drop - 1);
    q = m >> drop;
    if (rem > half || (rem == half && (q & 1))) q++;
  }
  if (q < ((uint64_t)1 << 23))                        // subnormal (or zero)
    return (uint32_t)q;
  int E = e + drop + 150;                             // value = q * 2^(e+drop), q in [2^23, 2^24]
  if (q >> 24) { q >>= 1; E++; }
  return ((uint32_t)E << 23) | ((uint32_t)q & 0x7fffff);
}

static inline uint64_t f32_unpack(uint32_t f, int *e)     // f = m * 2^e
{
  uint32_t E = f >> 23, m = f & 0x7fffff;
  if (E == 0) { *e = -149; return m; }
  *e = (int)E - 150;
  return m | 0x800000;
}

static uint32_t f32_add(uint32_t a, uint32_t b)
{
  if (a < b) { uint32_t t = a; a = b; b = t; }        // a >= b as values (both non-negative)
  if (b == 0) return a;
  int ea, eb;
  uint64_t ma = f32_unpack(a, &ea), mb = f32_unpack(b, &eb);
  int d = ea - eb, s = d < 38 ? d : 38;               // align b to a, keeping 38 guard bits
  uint64_t m = ma << s;
  if (d - s >= 63)
    m |= 1;                                           // b only decides ties: sticky bit
  else
    m += (mb >> (d - s)) | ((mb & (((uint64_t)1 << (d - s)) - 1)) != 0);
  return f32_round(m, ea - s);
}

static uint32_t f32_mul(uint32_t a, uint32_t b)
{
  if (a == 0 || b == 0) return 0;
  int ea, eb;
  uint64_t ma = f32_unpack(a, &ea), mb = f32_unpack(b, &eb);
  return f32_round(ma * mb, ea + eb);
}

static uint32_t f32_div(uint32_t a, uint32_t b)          // b > 0
{
  if (a == 0) return 0;
  int ea, eb;
  uint64_t ma = f32_unpack(a, &ea), mb = f32_unpack(b, &eb);
  int sa = 63 - msb64(ma), sb = 23 - msb64(mb);           // normalize: ma to bit 63, mb to bit 23
  ma <<= sa; mb <<= sb;
  uint64_t q = (ma >> 1) / mb, r = (ma >> 1) % mb;        // quotient has >= 38 bits
  return f32_round((q << 1) | (r != 0), ea - sa - (eb - sb));
}

static inline int32_t f32_round_half_up(uint32_t f)       // std::round of a non-negative float
{
  int e;
  uint64_t m = f32_unpack(f, &e);
  if (e >= 0) return (int32_t)(m << e);
  if (-e > 40) return 0;
  return (int32_t)((m + ((uint64_t)1 << (-e - 1))) >> -e);
}

// One SOFTMAX row exactly as LiteRT's x86 kernel computes it (optimized_ops::Softmax): float32
// sum of the table entries in index order, inv = 1 / (sum * scale_out), y = round(T[x] * inv) + zo.
// tf = float table already offset by 255 - max (the caller's row maximum).
static void softmax_row_exact(const uint32_t *tf, const int8_t *x, int n, uint32_t scale_out,
                              int32_t zo, int8_t *y)
{
  uint32_t sum = 0;
  for (int j = 0; j < n; j++)
    sum = f32_add(sum, tf[x[j]]);
  uint32_t inv = f32_div(0x3f800000u, f32_mul(sum, scale_out));
  for (int j = 0; j < n; j++) {
    int32_t v = f32_round_half_up(f32_mul(tf[x[j]], inv)) + zo;
    y[j] = (int8_t)(v > 127 ? 127 : v);
  }
}

#endif
