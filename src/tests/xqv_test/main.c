// Checks the X3 row-vector instructions vqdot4.vx and vqdot8.vx (DESIGN.md D12) implemented in
// Spike: directed edge cases, then random operands against a C model, at vl = 8 and vl = 5 (the
// elements past vl must stay unchanged).
//   make -B TEST=xqv_test run EXT=xqmac VEC=1
#include "runtime.h"


static int fails, checks;

static int32_t s8(uint64_t x, int i) { return (int8_t)(x >> (8 * i)); }

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t rnd(void)
{
  rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
  return rng;
}

static int32_t acc[8], w4[8], out[8];
static uint64_t w8[8];

// vd = v8 (acc), vs2 = v16 (w4) or v16-v17 (w8); result to out
static void run(int op, uint64_t a, int vl)
{
  if (op == 4)
    __asm__ volatile(
      "vsetivli zero, 8, e32, m1, ta, mu\n"
      "vle32.v v8, (%[acc])\n"
      "vle32.v v16, (%[w])\n"
      "vsetvli zero, %[vl], e32, m1, tu, mu\n"
      ".insn r 0x5B, 0, 0, x8, %[a], x16\n"
      "vsetivli zero, 8, e32, m1, ta, mu\n"
      "vse32.v v8, (%[out])\n"
      :: [acc] "r"(acc), [w] "r"(w4), [a] "r"(a), [vl] "r"(vl), [out] "r"(out)
      : "memory", "v8", "v16");
  else
    __asm__ volatile(
      "vsetivli zero, 8, e32, m1, ta, mu\n"
      "vle32.v v8, (%[acc])\n"
      "vl2re32.v v16, (%[w])\n"
      "vsetvli zero, %[vl], e32, m1, tu, mu\n"
      ".insn r 0x5B, 1, 0, x8, %[a], x16\n"
      "vsetivli zero, 8, e32, m1, ta, mu\n"
      "vse32.v v8, (%[out])\n"
      :: [acc] "r"(acc), [w] "r"(w8), [a] "r"(a), [vl] "r"(vl), [out] "r"(out)
      : "memory", "v8", "v16", "v17");
}

static void one(int op, uint64_t a, int vl)
{
  run(op, a, vl);
  for (int i = 0; i < 8; i++) {
    int32_t want = acc[i];
    if (i < vl) {
      int64_t s = 0;
      for (int t = 0; t < op; t++)
        s += s8(a, t) * (op == 4 ? s8((uint32_t)w4[i], t) : s8(w8[i], t));
      want = (int32_t)(uint32_t)((int64_t)acc[i] + s);
    }
    checks++;
    if (out[i] != want && fails++ < 20)
      printf("FAIL vqdot%d lane %d (vl %d): got %d, want %d\n", op, i, vl, out[i], want);
  }
}

static void fill(uint64_t pw, int32_t pacc)
{
  for (int i = 0; i < 8; i++) {
    acc[i] = pacc;
    w4[i] = (int32_t)pw;
    w8[i] = pw;
  }
}

int main(void)
{
  // directed: extremes of the int8 products and 32-bit wrap-around of the lanes
  const uint64_t ext[] = { 0, 0x8080808080808080ull, 0x7f7f7f7f7f7f7f7full, 0xffffffffffffffffull,
                           0x0102030405060708ull, 0x807f807f807f807full };
  const int32_t accs[] = { 0, 1, -1, 0x7fffffff, (int32_t)0x80000000, 123456789 };
  for (unsigned i = 0; i < sizeof ext / 8; i++)
    for (unsigned j = 0; j < sizeof ext / 8; j++)
      for (unsigned k = 0; k < sizeof accs / 4; k++) {
        fill(ext[j], accs[k]);
        one(4, ext[i], 8);
        one(8, ext[i], 8);
      }
  // random operands, full and partial vl
  for (int n = 0; n < 20000; n++) {
    for (int i = 0; i < 8; i++) {
      acc[i] = (int32_t)rnd();
      w4[i] = (int32_t)rnd();
      w8[i] = rnd();
    }
    uint64_t a = rnd();
    int vl = (n & 3) == 3 ? 5 : 8;
    one(4, a, vl);
    one(8, a, vl);
  }
  if (fails)
    printf("FAIL: %d of %d lane checks\n", fails, checks);
  else
    printf("PASS: %d lane checks (vqdot4.vx, vqdot8.vx)\n", checks);
  return fails ? 1 : 0;
}
