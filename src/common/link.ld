/* Bare-metal linker script for Spike and CV-Wally (RAM at 0x80000000). */
OUTPUT_ARCH("riscv")
ENTRY(_start)

__stack_size = 0x100000;   /* 1 MiB */

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 1024M
}

SECTIONS
{
  .text.init : { *(.text.init) } > RAM

  /* HTIF mailbox used by Spike (must be 8-byte aligned, own section). */
  .tohost ALIGN(0x1000) : { *(.tohost) } > RAM

  .text ALIGN(16) : { *(.text .text.*) } > RAM

  .rodata ALIGN(16) : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM

  .data ALIGN(16) : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
    *(.data .data.*)
  } > RAM

  .bss ALIGN(16) (NOLOAD) : {
    __bss_start = .;
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(16);
    __bss_end = .;
  } > RAM

  .stack ALIGN(16) (NOLOAD) : {
    . += __stack_size;
    __stack_top = .;
  } > RAM

  _end = .;
}
