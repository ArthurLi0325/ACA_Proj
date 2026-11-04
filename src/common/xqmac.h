// C wrappers for the Xqmac custom instructions (DESIGN.md D4-D6), emitted with .insn so no
// assembler changes are needed. Programs using them must run on Spike with --extension=xqmac
// (make ... EXT=xqmac).
#ifndef XQMAC_H
#define XQMAC_H

#include <stdint.h>

// The results are 32-bit values sign-extended to 64 bits. qmac8/dot8 return int64_t so that
// GCC knows the register is already sign-extended and does not add a sext.w after every one;
// keep accumulators in int64_t variables and truncate to int32_t once at the end.

// sext32(acc[31:0] + sum_{i<8} int8(a.byte[i]) * int8(b.byte[i])), 32-bit wrap-around.
static inline int64_t qmac8(int64_t acc, uint64_t a, uint64_t b)
{
  __asm__(".insn r 0x0B, 0, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b));
  return acc;
}

// sum_{i<8} int8(a.byte[i]) * int8(b.byte[i]).
static inline int64_t dot8(uint64_t a, uint64_t b)
{
  int64_t r;
  __asm__(".insn r 0x0B, 1, 0, %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

// Parameter word for rq / rq.a: [31:0] M0, [37:32] shift (TFLite convention, > 0 = left),
// [47:40] output zero point.
static inline uint64_t rq_param(int32_t m0, int shift, int32_t zp)
{
  return (uint32_t)m0 | ((uint64_t)(shift & 0x3f) << 32) | ((uint64_t)(zp & 0xff) << 40);
}

// clamp(requant(acc) + zp, -128, 127); rq rounds ties toward +inf (FC), rq_a away from 0 (BMM).
static inline int32_t rq(int32_t acc, uint64_t param)
{
  int32_t r;
  __asm__(".insn r 0x0B, 4, 0, %0, %1, %2" : "=r"(r) : "r"(acc), "r"(param));
  return r;
}

static inline int32_t rq_a(int32_t acc, uint64_t param)
{
  int32_t r;
  __asm__(".insn r 0x0B, 5, 0, %0, %1, %2" : "=r"(r) : "r"(acc), "r"(param));
  return r;
}

#endif
