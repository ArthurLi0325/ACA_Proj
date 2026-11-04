// Bare-metal runtime: HTIF console/exit, tiny printf, performance counters.
#ifndef RUNTIME_H
#define RUNTIME_H

#include <stdint.h>
#include <stddef.h>

void exit(int code) __attribute__((noreturn));
void putchar_htif(char c);
int  printf(const char *fmt, ...);

void *memcpy(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
int   memcmp(const void *a, const void *b, size_t n);

static inline uint64_t rdcycle(void)   { uint64_t x; __asm__ volatile("rdcycle %0"   : "=r"(x)); return x; }
static inline uint64_t rdinstret(void) { uint64_t x; __asm__ volatile("rdinstret %0" : "=r"(x)); return x; }

// Region markers: a uniquely-encoded no-op (slti x0, x0, id) that trace and
// histogram scripts can search for to attribute instructions to operators.
#define MARK(id) __asm__ volatile("slti x0, x0, %0" :: "i"(id))

#endif
