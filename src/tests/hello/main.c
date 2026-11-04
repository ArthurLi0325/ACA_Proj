// Environment smoke test: console output, counters, and a small int8 dot product.
#include "runtime.h"

#define N 1024

static int8_t a[N], w[N];

__attribute__((noinline))
static int32_t dot_i8(const int8_t *x, const int8_t *y, int n)
{
  int32_t acc = 0;
  for (int i = 0; i < n; i++)
    acc += x[i] * y[i];
  return acc;
}

int main(void)
{
  printf("Hello from bare-metal RV64IMC!\n");

  for (int i = 0; i < N; i++) {
    a[i] = (int8_t)(i * 7 - 50);
    w[i] = (int8_t)(31 - i * 3);
  }

  uint64_t i0 = rdinstret(), c0 = rdcycle();
  MARK(1);
  int32_t acc = dot_i8(a, w, N);
  MARK(2);
  uint64_t i1 = rdinstret(), c1 = rdcycle();

  // Expected value computed on the host with the same wrap-around inputs.
  const int32_t expected = -121856;
  printf("dot8(%d) = %d (expected %d)\n", N, acc, expected);
  printf("instret = %lu, cycles = %lu, instr/MAC = %lu.%02lu\n",
         i1 - i0, c1 - c0, (i1 - i0) / N, ((i1 - i0) % N) * 100 / N);

  return acc == expected ? 0 : 1;
}
