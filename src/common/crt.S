# Minimal bare-metal startup: set up gp/sp, clear .bss, install trap handler,
# call main(), then report the return code through HTIF.

  .section .text.init
  .globl _start
_start:
  .option push
  .option norelax
  la    gp, __global_pointer$
  .option pop
  la    sp, __stack_top

  la    t0, trap_entry
  csrw  mtvec, t0
#ifdef __riscv_vector
  li    t0, 1 << 9      # mstatus.VS = Initial: the X3 builds use the vector unit
  csrs  mstatus, t0
#endif

  la    t0, __bss_start
  la    t1, __bss_end
1:
  bgeu  t0, t1, 2f
  sd    zero, 0(t0)
  addi  t0, t0, 8
  j     1b
2:
  call  main
  call  exit          # exit(main's return value)
3:
  j     3b

# Any trap (e.g. an unimplemented custom instruction) lands here.
  .align 4
trap_entry:
  csrr  a0, mcause
  csrr  a1, mepc
  csrr  a2, mtval
  call  trap_handler
4:
  j     4b
