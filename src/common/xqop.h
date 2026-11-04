// X2: outer-product MAC with an internal 8x8 int32 accumulator array (custom-1),
// see DESIGN.md D10 and the top of riscv/xqmac.cc (patches/spike-xqop.patch).
// The accumulators are state the compiler cannot see, so every wrapper is volatile (kept in order).
#ifndef XQOP_H
#define XQOP_H

#include <stdint.h>

// ACC[i][j] += sum_{t<2} a.b[2i+t] * b.b[2j+t], i, j < 4
static inline void qop4(uint64_t a, uint64_t b)
{
  __asm__ volatile(".insn r 0x2B, 0, 0, x0, %0, %1" :: "r"(a), "r"(b));
}

// ACC[i][j] += a.b[i] * b.b[j], i, j < 8
static inline void qop8(uint64_t a, uint64_t b)
{
  __asm__ volatile(".insn r 0x2B, 1, 0, x0, %0, %1" :: "r"(a), "r"(b));
}

// ACC[i][2c] = v[31:0], ACC[i][2c+1] = v[63:32] for every row i (c = 0..3)
#define QACC_COL(v, c) __asm__ volatile(".insn r 0x2B, 2, 0, x0, %0, x%1" :: "r"(v), "i"(c))

// ACC[2c][j] += v[31:0], ACC[2c+1][j] += v[63:32] for every column j (c = 0..3)
#define QACC_ROW(v, c) __asm__ volatile(".insn r 0x2B, 5, 0, x0, %0, x%1" :: "r"(v), "i"(c))

// rq / rq.a of ACC[n] (n = 0..63, row-major) with the packed parameter word p (DESIGN.md D5)
#define QACC_RQ(p, n) ({ int64_t r_;                                                   \
    __asm__ volatile(".insn r 0x2B, 3, %2, %0, %1, x%3"                                 \
                     : "=r"(r_) : "r"(p), "i"((n) >> 5), "i"((n) & 31)); r_; })
#define QACC_RQA(p, n) ({ int64_t r_;                                                  \
    __asm__ volatile(".insn r 0x2B, 4, %2, %0, %1, x%3"                                 \
                     : "=r"(r_) : "r"(p), "i"((n) >> 5), "i"((n) & 31)); r_; })

#endif
