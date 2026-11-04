#include <stdarg.h>
#include "runtime.h"

// ---- HTIF (Spike host-target interface) ----
volatile uint64_t tohost   __attribute__((section(".tohost"), aligned(8)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(8)));

// Spike only polls tohost every few thousand instructions, so every HTIF call
// busy-waits. Console output is therefore buffered and sent as one
// write(1, buf, len) syscall per line to keep profiles clean.
#define SYS_WRITE 64
static char     con_buf[256];
static unsigned con_len;
static volatile uint64_t magic_mem[8] __attribute__((aligned(64)));

static void htif_syscall(uint64_t n, uint64_t a0, uint64_t a1, uint64_t a2)
{
  magic_mem[0] = n; magic_mem[1] = a0; magic_mem[2] = a1; magic_mem[3] = a2;
  __asm__ volatile("fence" ::: "memory");
  tohost = (uintptr_t)magic_mem;            // device 0, cmd 0, pointer => syscall
  while (fromhost == 0) ;
  fromhost = 0;
}

static void con_flush(void)
{
  if (con_len) htif_syscall(SYS_WRITE, 1, (uintptr_t)con_buf, con_len);
  con_len = 0;
}

void putchar_htif(char c)
{
  con_buf[con_len++] = c;
  if (c == '\n' || con_len == sizeof(con_buf)) con_flush();
}

void exit(int code)
{
  con_flush();
  // device 0, command 0: exit with (code << 1) | 1; non-zero code => FAILED
  tohost = ((uint64_t)code << 1) | 1;
  for (;;) ;
}

void trap_handler(uint64_t mcause, uint64_t mepc, uint64_t mtval)
{
  printf("\n*** TRAP mcause=%lu mepc=0x%lx mtval=0x%lx\n", mcause, mepc, mtval);
  exit(3);
}

// ---- libc pieces gcc may emit calls to ----
void *memcpy(void *dst, const void *src, size_t n)
{
  uint8_t *d = dst; const uint8_t *s = src;
  while (n--) *d++ = *s++;
  return dst;
}

void *memset(void *dst, int c, size_t n)
{
  uint8_t *d = dst;
  while (n--) *d++ = (uint8_t)c;
  return dst;
}

int memcmp(const void *a, const void *b, size_t n)
{
  const uint8_t *x = a, *y = b;
  for (; n; n--, x++, y++)
    if (*x != *y) return *x - *y;
  return 0;
}

// ---- tiny printf: %d %i %u %x %c %s %p %%, with 'l' and width/0-pad ----
static void print_num(uint64_t v, int base, int neg, int width, char pad)
{
  char buf[24]; int i = 0;
  do { int d = v % base; buf[i++] = d < 10 ? '0' + d : 'a' + d - 10; v /= base; } while (v);
  if (neg) buf[i++] = '-';
  while (i < width--) putchar_htif(pad);
  while (i) putchar_htif(buf[--i]);
}

int printf(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  for (; *fmt; fmt++) {
    if (*fmt != '%') { putchar_htif(*fmt); continue; }
    fmt++;
    char pad = ' '; int width = 0, lng = 0;
    if (*fmt == '0') { pad = '0'; fmt++; }
    while (*fmt >= '0' && *fmt <= '9') width = width * 10 + (*fmt++ - '0');
    while (*fmt == 'l') { lng = 1; fmt++; }
    switch (*fmt) {
    case 'd': case 'i': {
      int64_t v = lng ? va_arg(ap, int64_t) : va_arg(ap, int);
      print_num(v < 0 ? -(uint64_t)v : (uint64_t)v, 10, v < 0, width, pad);
      break;
    }
    case 'u': print_num(lng ? va_arg(ap, uint64_t) : va_arg(ap, unsigned), 10, 0, width, pad); break;
    case 'x': print_num(lng ? va_arg(ap, uint64_t) : va_arg(ap, unsigned), 16, 0, width, pad); break;
    case 'p': print_num((uintptr_t)va_arg(ap, void *), 16, 0, width, pad); break;
    case 'c': putchar_htif((char)va_arg(ap, int)); break;
    case 's': { const char *s = va_arg(ap, const char *); while (*s) putchar_htif(*s++); break; }
    case '%': putchar_htif('%'); break;
    default: putchar_htif('%'); putchar_htif(*fmt); break;
    }
  }
  va_end(ap);
  return 0;
}
