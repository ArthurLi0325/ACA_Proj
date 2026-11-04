// Checks the X2 outer-product instructions (qop4, qop8, qacc.col, qacc.row, qacc.rq, qacc.rqa,
// DESIGN.md D10) implemented in Spike: directed cases, then random sequences against a C model of
// the 8x8 accumulator array. The accumulators can only be read through qacc.rq / qacc.rqa, so each
// one is read with several parameter words (identity and scaled by 2^-8, 2^-16, 2^-24, which
// together expose all 32 bits) plus one random parameter word for the rounding modes.
//   make -B TEST=xqop_test run EXT=xqmac          (needs patches/spike-xqop.patch)
#include "runtime.h"
#include "xqmac.h"
#include "xqop.h"

static int fails, checks;

static void check(const char *what, int n, long got, long want)
{
  checks++;
  if (got != want && fails++ < 20)
    printf("FAIL %s [%d]: got %ld, want %ld\n", what, n, got, want);
}

// ---- reference model ----
static int32_t ref[64];                    // ACC[i][j] = ref[8*i + j]

static int32_t s8(uint64_t x, int i) { return (int8_t)(x >> (8 * i)); }
static int32_t wrap(int64_t v) { return (int32_t)(uint32_t)v; }

static void ref_qop4(uint64_t a, uint64_t b)
{
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      ref[8 * i + j] = wrap((int64_t)ref[8 * i + j] + s8(a, 2 * i) * s8(b, 2 * j) +
                            s8(a, 2 * i + 1) * s8(b, 2 * j + 1));
}

static void ref_qop8(uint64_t a, uint64_t b)
{
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      ref[8 * i + j] = wrap((int64_t)ref[8 * i + j] + s8(a, i) * s8(b, j));
}

static void ref_col(uint64_t v, int c)
{
  for (int i = 0; i < 8; i++) {
    ref[8 * i + 2 * c] = (int32_t)v;
    ref[8 * i + 2 * c + 1] = (int32_t)(v >> 32);
  }
}

static void ref_row(uint64_t v, int c)
{
  for (int j = 0; j < 8; j++) {
    ref[8 * (2 * c) + j] = wrap((int64_t)ref[8 * (2 * c) + j] + (int32_t)v);
    ref[8 * (2 * c + 1) + j] = wrap((int64_t)ref[8 * (2 * c + 1) + j] + (int32_t)(v >> 32));
  }
}

// D5 requantization, written independently of Spike (same model as xqmac_test)
static int64_t div_pot_away(int64_t v, int r)
{
  int64_t q = ((v < 0 ? -v : v) + ((int64_t)1 << r >> 1)) >> r;
  return v < 0 ? -q : q;
}

static int32_t ref_rq(int32_t acc, uint64_t p, int ties_away)
{
  int32_t m0 = (int32_t)p, zp = (int8_t)(p >> 40);
  int sh = (int)((int64_t)(p << 26) >> 58);
  int32_t a = sh > 0 ? (int32_t)((uint32_t)acc << sh) : acc;
  int64_t v = ((int64_t)a * m0 + ((int64_t)1 << 30)) >> 31;
  if (v > INT32_MAX) v = INT32_MAX;
  int r = sh < 0 ? -sh : 0;
  if (r)
    v = ties_away ? div_pot_away(v, r) : (v + ((int64_t)1 << (r - 1))) >> r;
  v += zp;
  return v < -128 ? -128 : v > 127 ? 127 : (int32_t)v;
}

// ---- reading the hardware accumulators (the index is an immediate, so unroll all 64) ----
#define RD(n)  out[n] = (int32_t)(away ? QACC_RQA(p, n) : QACC_RQ(p, n));
#define RD8(r) RD(8 * (r) + 0) RD(8 * (r) + 1) RD(8 * (r) + 2) RD(8 * (r) + 3) \
               RD(8 * (r) + 4) RD(8 * (r) + 5) RD(8 * (r) + 6) RD(8 * (r) + 7)
__attribute__((noinline))
static void read_all(uint64_t p, int away, int32_t out[64])
{
  RD8(0) RD8(1) RD8(2) RD8(3) RD8(4) RD8(5) RD8(6) RD8(7)
}

#define QCOL(v, c) do { uint64_t v_ = (v); QACC_COL(v_, c); ref_col(v_, c); } while (0)
#define QROW(v, c) do { uint64_t v_ = (v); QACC_ROW(v_, c); ref_row(v_, c); } while (0)

static uint64_t pair(int32_t lo, int32_t hi) { return (uint32_t)lo | (uint64_t)(uint32_t)hi << 32; }

static void clear(void)
{
  for (int c = 0; c < 4; c++) {
    switch (c) {
    case 0: QCOL(0, 0); break;
    case 1: QCOL(0, 1); break;
    case 2: QCOL(0, 2); break;
    case 3: QCOL(0, 3); break;
    }
  }
}

static uint64_t rng = 0x2545f4914f6cdd1dull;
static uint64_t next(void)                // xorshift64*
{
  rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
  return rng * 0x2545f4914f6cdd1dull;
}

// compares every accumulator with the model through 4 fixed + 1 random parameter word, both modes
static void compare(const char *what)
{
  static const uint64_t fixed[4] = {
    (1u << 30) | (1ull << 32),             // x * 0.5 << 1 = x (clamped to int8)
    (1u << 30) | (0x39ull << 32),          // x / 2^8   (shift -7)
    (1u << 30) | (0x31ull << 32),          // x / 2^16  (shift -15)
    (1u << 30) | (0x29ull << 32),          // x / 2^24  (shift -23)
  };
  int32_t got[64];
  for (int k = 0; k < 5; k++) {
    uint64_t p = k < 4 ? fixed[k] : next() & 0xffff3fffffffffffull;
    for (int away = 0; away < 2; away++) {
      read_all(p, away, got);
      for (int n = 0; n < 64; n++)
        check(what, n, got[n], ref_rq(ref[n], p, away));
    }
  }
}

// ---- directed cases ----
static const uint64_t ID = (1u << 30) | (1ull << 32);          // identity readout

static void directed(void)
{
  int32_t got[64];
  clear();
  QCOL(pair(5, -7), 1);                                         // ACC[i][2] = 5, ACC[i][3] = -7
  read_all(ID, 0, got);
  check("qacc.col lo", 7 * 8 + 2, got[7 * 8 + 2], 5);
  check("qacc.col hi", 0 * 8 + 3, got[0 * 8 + 3], -7);
  check("qacc.col leaves col 4", 0 * 8 + 4, got[0 * 8 + 4], 0);
  QROW(pair(-2, 3), 2);                                         // rows 4, 5 += -2, 3
  read_all(ID, 0, got);
  check("qacc.row adds", 4 * 8 + 2, got[4 * 8 + 2], 3);
  check("qacc.row hi row", 5 * 8 + 3, got[5 * 8 + 3], -4);
  check("qacc.row leaves row 6", 6 * 8 + 0, got[6 * 8 + 0], 0);

  clear();
  qop4(0x0201, 0x0403); ref_qop4(0x0201, 0x0403);              // row 0 = (1, 2), col 0 = (3, 4)
  read_all(ID, 0, got);
  check("qop4 byte pairs", 0, got[0], 1 * 3 + 2 * 4);
  check("qop4 other rows untouched", 8, got[8], 0);
  clear();
  qop4(0xffffffffffffffffull, 0x0101010101010101ull);          // every lane: 2 * (-1 * 1)
  ref_qop4(0xffffffffffffffffull, 0x0101010101010101ull);
  read_all(ID, 0, got);
  check("qop4 lane (3,3)", 27, got[27], -2);
  check("qop4 leaves (4,4)", 36, got[36], 0);
  check("qop4 leaves (0,4)", 4, got[4], 0);

  clear();
  qop8(0x8000000000000000ull, 0x80);                            // a.b[7] = b.b[0] = -128
  ref_qop8(0x8000000000000000ull, 0x80);
  read_all((1u << 30) | (0x39ull << 32), 0, got);               // 16384 / 256 = 64
  check("qop8 -128*-128 at (7,0)", 56, got[56], 64);
  read_all(ID, 0, got);
  check("qop8 (0,0) zero", 0, got[0], 0);

  QCOL(pair(INT32_MAX, 0), 0);                                  // accumulators wrap at 2^31
  qop8(1, 1); ref_qop8(1, 1);
  compare("wrap");

  clear();
  QCOL(pair(5, -5), 0);
  const uint64_t HALF = 0x7fffffffull | (0x3full << 32);        // M0 = 2^31-1, shift -1
  read_all(HALF, 0, got);
  check("qacc.rq   2.5 -> 3", 0, got[0], 3);
  check("qacc.rq  -2.5 -> -2", 1, got[1], -2);
  read_all(HALF, 1, got);
  check("qacc.rqa  2.5 -> 3", 0, got[0], 3);
  check("qacc.rqa -2.5 -> -3", 1, got[1], -3);

  clear();
  QCOL(pair(11, 22), 0); QCOL(pair(33, 44), 1); QCOL(pair(55, 66), 2); QCOL(pair(77, 88), 3);
  QROW(pair(100, 0), 2);                                        // index 32+ uses funct7[0]
  read_all(ID, 0, got);
  check("readout index 32 (row 4)", 32, got[32], 111);
  check("readout index 39", 39, got[39], 127);                  // 188 clamps
  check("readout index 47 (row 5)", 47, got[47], 88);
  compare("directed");
}

// ---- random sequences ----
static void randomized(int n)
{
  for (int it = 0; it < n; it++) {
    int32_t range = (it & 1) ? 0xffffff : -1;                   // model-like or full range
    QCOL(next() & pair(range, range), 0); QCOL(next() & pair(range, range), 1);
    QCOL(next() & pair(range, range), 2); QCOL(next() & pair(range, range), 3);
    if (it & 2) {
      QROW(next() & pair(range, range), 0); QROW(next() & pair(range, range), 1);
      QROW(next() & pair(range, range), 2); QROW(next() & pair(range, range), 3);
    }
    int ops = 1 + (int)(next() % 64);
    for (int k = 0; k < ops; k++) {
      uint64_t a = next(), b = next();
      if (next() & 1) { qop4(a, b); ref_qop4(a, b); }
      else            { qop8(a, b); ref_qop8(a, b); }
    }
    compare("random");
  }
}

int main(void)
{
  directed();
  printf("directed: %d checks, %d failures\n", checks, fails);
  int d = checks;
  randomized(2000);
  printf("random:   %d checks, %d failures\n", checks - d, fails);
  printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
