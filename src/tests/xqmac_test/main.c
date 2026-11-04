// Checks the Xqmac instructions (qmac8, dot8, rq, rq.a) implemented in Spike against the spec
// in DESIGN.md D4/D5: hand-computed directed cases, then random cases against a C reference
// model written independently of the Spike code (different formulation of each rounding step).
//   make TEST=xqmac_test run EXT=xqmac
#include "runtime.h"
#include "xqmac.h"

static int fails, checks;

static void check(const char *what, long got, long want)
{
  checks++;
  if (got != want && fails++ < 20)
    printf("FAIL %s: got %ld, want %ld\n", what, got, want);
}

// ---- reference model ----
static int32_t ref_dot8(uint64_t a, uint64_t b)
{
  int8_t x[8], y[8];
  memcpy(x, &a, 8);                       // little-endian: x[i] = a.byte[i]
  memcpy(y, &b, 8);
  int64_t s = 0;
  for (int i = 0; i < 8; i++)
    s += x[i] * y[i];
  return (int32_t)s;                      // |s| <= 2^17, never wraps
}

// round(v / 2^r) with ties away from zero (RoundingDivideByPOT), written on magnitudes
static int64_t div_pot_away(int64_t v, int r)
{
  int64_t q = ((v < 0 ? -v : v) + ((int64_t)1 << r >> 1)) >> r;
  return v < 0 ? -q : q;
}

static int32_t ref_rq(int32_t acc, int32_t m0, int sh, int zp, int ties_away)
{
  int32_t a = sh > 0 ? (int32_t)((uint32_t)acc << sh) : acc;
  // SRDHM: gemmlowp's nudge (1 - 2^30 for negative products, then truncation) makes exact
  // halves round toward +inf for both signs, i.e. floor((a*m0 + 2^30) / 2^31)
  int64_t v = ((int64_t)a * m0 + ((int64_t)1 << 30)) >> 31;
  if (v > INT32_MAX) v = INT32_MAX;                       // only a = m0 = INT32_MIN
  int r = sh < 0 ? -sh : 0;
  if (r)
    v = ties_away ? div_pot_away(v, r) : (v + ((int64_t)1 << (r - 1))) >> r;  // >> is floor
  v += zp;
  return v < -128 ? -128 : v > 127 ? 127 : (int32_t)v;
}

// ---- directed cases ----
static void directed(void)
{
  const uint64_t m128 = 0x8080808080808080ull, p127 = 0x7f7f7f7f7f7f7f7full;
  check("dot8 -128*-128", dot8(m128, m128), 131072);
  check("dot8 -128*127", dot8(m128, p127), -130048);
  check("dot8 byte0/1 order", dot8(0x0102, 0x0304), 2 * 4 + 1 * 3);
  check("dot8 byte7 sign", dot8(0x0100000000000000ull, 0xff00000000000000ull), -1);
  check("dot8 zero", dot8(0, m128), 0);
  check("qmac8 accumulate", qmac8(1000, m128, m128), 132072);
  check("qmac8 wraps at 2^31", qmac8(INT32_MAX, 1, 1), INT32_MIN);

  // qmac8 must ignore rd[63:32] and return a sign-extended 32-bit result
  int64_t r = 0xdeadbeef00000005ll;
  __asm__(".insn r 0x0B, 0, 0, %0, %1, %2" : "+r"(r) : "r"(1ull), "r"(0xf9ull));  // 5 + 1*(-7)
  check("qmac8 ignores rd[63:32]", r, -2);

  // M0 = 2^31-1 makes SRDHM the identity for small acc, isolating the final shift
  const int32_t ONE = INT32_MAX;
  check("rq    2.5 -> 3", rq(5, rq_param(ONE, -1, 0)), 3);
  check("rq.a  2.5 -> 3", rq_a(5, rq_param(ONE, -1, 0)), 3);
  check("rq   -2.5 -> -2", rq(-5, rq_param(ONE, -1, 0)), -2);
  check("rq.a -2.5 -> -3", rq_a(-5, rq_param(ONE, -1, 0)), -3);
  check("rq   -1.5 -> -1", rq(-6, rq_param(ONE, -2, 0)), -1);
  check("rq.a -1.5 -> -2", rq_a(-6, rq_param(ONE, -2, 0)), -2);
  check("rq   -1.75 -> -2", rq(-7, rq_param(ONE, -2, 0)), -2);
  check("rq    1.5 -> 2", rq(6, rq_param(ONE, -2, 0)), 2);
  check("rq SRDHM 3*0.5 -> 2", rq(3, rq_param(1 << 30, 0, 0)), 2);
  check("rq SRDHM -3*0.5 -> -1", rq(-3, rq_param(1 << 30, 0, 0)), -1);   // SRDHM ties go up
  check("rq left shift 3<<2 * 0.5", rq(3, rq_param(1 << 30, 2, 0)), 6);
  check("rq saturate +", rq(1000, rq_param(ONE, 0, 0)), 127);
  check("rq saturate -", rq(-1000, rq_param(ONE, 0, 0)), -128);
  check("rq zp then clamp", rq(100, rq_param(ONE, 0, 50)), 127);
  check("rq zp -128", rq(10, rq_param(ONE, 0, -128)), -118);
  check("rq zp -128 clamp", rq(-10, rq_param(ONE, 0, -128)), -128);
  // SRDHM saturates to INT32_MAX; the rounding add must not overflow 32 bits: 128 - 100
  check("rq SRDHM saturation", rq(INT32_MIN, rq_param(INT32_MIN, -24, -100)), 28);
  check("rq.a SRDHM saturation", rq_a(INT32_MIN, rq_param(INT32_MIN, -24, -100)), 28);
  // reserved bits are ignored
  uint64_t p = rq_param(0x5a5a5a5a, -9, 3);
  check("rq reserved bits", rq(123456, p | 0xffff000000000000ull | (3ull << 38)), rq(123456, p));
}

// ---- random cases ----
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t next(void)                // xorshift64*
{
  rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
  return rng * 0x2545f4914f6cdd1dull;
}

static void randomized(int n)
{
  for (int i = 0; i < n; i++) {
    uint64_t a = next(), b = next();
    int32_t acc = (int32_t)next();
    check("random dot8", dot8(a, b), ref_dot8(a, b));
    check("random qmac8", qmac8(acc, a, b), (int32_t)((uint32_t)acc + (uint32_t)ref_dot8(a, b)));

    // half the cases use model-like values (|acc| < 2^24, shift -15..-6), half the full range
    uint64_t s = next();
    int32_t m0 = (int32_t)next();
    int sh, zp = (int8_t)(s >> 8);
    if (s & 1) {
      acc >>= 8;
      m0 = (int32_t)(0x40000000u | ((uint32_t)m0 & 0x3fffffffu));   // [0.5, 1) like TFLite
      sh = -6 - (int)((s >> 16) % 10);
    } else {
      sh = (int)((s >> 16) & 0x3f) - 32;                             // -32..31
    }
    uint64_t prm = rq_param(m0, sh, zp);
    check("random rq", rq(acc, prm), ref_rq(acc, m0, sh, zp, 0));
    check("random rq.a", rq_a(acc, prm), ref_rq(acc, m0, sh, zp, 1));
  }
}

int main(void)
{
  directed();
  printf("directed: %d checks, %d failures\n", checks, fails);
  int d = checks;
  randomized(100000);
  printf("random:   %d checks, %d failures\n", checks - d, fails);
  printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
