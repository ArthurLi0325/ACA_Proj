// Emits an R-type instruction in the custom-0 opcode space (0x0B) via `.insn`.
// Stock Spike must trap with mcause=2 (illegal instruction); once the custom
// extension is implemented in our Spike, this becomes its first smoke test.
#include "runtime.h"

int main(void)
{
  uint64_t rs1 = 0x0102030405060708ULL, rs2 = 0x0101010101010101ULL, rd;
  printf("issuing custom-0 instruction...\n");
  __asm__ volatile(".insn r 0x0B, 0, 0, %0, %1, %2" : "=r"(rd) : "r"(rs1), "r"(rs2));
  printf("rd = 0x%lx\n", rd);
  return 0;
}
