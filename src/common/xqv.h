// X3: row-vector dot products on the vector registers (custom-2, DESIGN.md D12; Spike
// patches/spike-xqv.patch). The instructions name fixed vector registers in their encoding, so
// they are used from hand-written assembly (src/tests/fc_x3/kern.S) or with fixed registers here.
//   vqdot4.vx vd, rs1, vs2   vd[i] += sum_{t<4} rs1.b[t] * vs2.e32[i].b[t]   (SEW = 32, i < vl)
//   vqdot8.vx vd, rs1, vs2   vd[i] += sum_{t<8} rs1.b[t] * vs2.e64[i].b[t]   (vs2 group of 2)
// Build with MARCH=rv64imc_zicsr_zicntr_zve64x_zvl256b, run with ISA=rv64imc_zicntr_zve64x_zvl256b.
#ifndef XQV_H
#define XQV_H

// assembler macros: vd and vs2 are vector register numbers, rs1 an integer register name
#define VQDOT4(vd, rs1, vs2) .insn r 0x5B, 0, 0, x##vd, rs1, x##vs2
#define VQDOT8(vd, rs1, vs2) .insn r 0x5B, 1, 0, x##vd, rs1, x##vs2

#endif
