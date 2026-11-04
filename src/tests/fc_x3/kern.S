# X3 tile kernels (DESIGN.md D12): one 8x8 output tile, one output row per vector register v0..v7
# (8 int32 lanes = 8 output columns, VLEN = 256), operands in the fc3.h layout. QDOT8 selects
# vqdot8.vx (8 bytes per lane per instruction), otherwise vqdot4.vx (4 bytes).
# Requantization uses standard RVV fixed-point instructions with vxrm = rnu (set in crt.S):
#   vsmul  = gemmlowp SRDHM (round half up, saturates only for INT32_MIN * INT32_MIN)
#   vssra  = rounding right shift, ties toward +inf (rq); rq.a first adds v >> 31 (ties away from 0)
#   vnclip = clamp to int16, then to int8 (after the output zero point is added)
# Only right shifts are supported: every FC / BATCH_MATMUL shift in DeiT-Tiny is negative.
#include "xqv.h"

#ifdef QDOT8
#define QD 8
#else
#define QD 4
#endif

  .text

# K loop: a0 = A tile (8 rows), a1 = W/B tile (8 columns), a2 = end of A tile; v0..v7 += A x W^T.
# Uses t0-t6, a3.
#if QD == 8
#define KLOOP                                                          \
1:                                                                     ;\
  vl2re32.v v16, (a1)                                                  ;\
  addi  a1, a1, 64                                                     ;\
  ld    t0, 0(a0)  ; ld t1, 8(a0)  ; ld t2, 16(a0) ; ld t3, 24(a0)      ;\
  ld    t4, 32(a0) ; ld t5, 40(a0) ; ld t6, 48(a0) ; ld a3, 56(a0)      ;\
  VQDOT8(0, t0, 16) ; VQDOT8(1, t1, 16) ; VQDOT8(2, t2, 16) ; VQDOT8(3, t3, 16) ;\
  VQDOT8(4, t4, 16) ; VQDOT8(5, t5, 16) ; VQDOT8(6, t6, 16) ; VQDOT8(7, a3, 16) ;\
  addi  a0, a0, 64                                                     ;\
  bne   a0, a2, 1b
#else
#define KLOOP                                                          \
1:                                                                     ;\
  vle32.v v16, (a1)                                                    ;\
  addi  t0, a1, 32                                                     ;\
  vle32.v v17, (t0)                                                    ;\
  addi  a1, a1, 64                                                     ;\
  lw    t0, 0(a0)  ; lw t1, 4(a0)  ; lw t2, 8(a0)  ; lw t3, 12(a0)      ;\
  lw    t4, 16(a0) ; lw t5, 20(a0) ; lw t6, 24(a0) ; lw a3, 28(a0)      ;\
  VQDOT4(0, t0, 16) ; VQDOT4(1, t1, 16) ; VQDOT4(2, t2, 16) ; VQDOT4(3, t3, 16) ;\
  VQDOT4(4, t4, 16) ; VQDOT4(5, t5, 16) ; VQDOT4(6, t6, 16) ; VQDOT4(7, a3, 16) ;\
  lw    t0, 32(a0) ; lw t1, 36(a0) ; lw t2, 40(a0) ; lw t3, 44(a0)      ;\
  lw    t4, 48(a0) ; lw t5, 52(a0) ; lw t6, 56(a0) ; lw a3, 60(a0)      ;\
  VQDOT4(0, t0, 17) ; VQDOT4(1, t1, 17) ; VQDOT4(2, t2, 17) ; VQDOT4(3, t3, 17) ;\
  VQDOT4(4, t4, 17) ; VQDOT4(5, t5, 17) ; VQDOT4(6, t6, 17) ; VQDOT4(7, a3, 17) ;\
  addi  a0, a0, 64                                                     ;\
  bne   a0, a2, 1b
#endif

# v0..v7 (int32, zero point added) -> int8 rows; row r stored as 2 words at o + r*ldo and
# o + r*ldo + stride (stride 4 = 8 contiguous bytes). a6 = o, a7 = ldo, t0 = stride.
#define STORE                                                          \
  vsetivli zero, 8, e16, mf2, ta, ma                                   ;\
  vnclip.wi v0, v0, 0 ; vnclip.wi v1, v1, 0 ; vnclip.wi v2, v2, 0 ; vnclip.wi v3, v3, 0 ;\
  vnclip.wi v4, v4, 0 ; vnclip.wi v5, v5, 0 ; vnclip.wi v6, v6, 0 ; vnclip.wi v7, v7, 0 ;\
  vsetivli zero, 8, e8, mf4, ta, ma                                    ;\
  vnclip.wi v0, v0, 0 ; vnclip.wi v1, v1, 0 ; vnclip.wi v2, v2, 0 ; vnclip.wi v3, v3, 0 ;\
  vnclip.wi v4, v4, 0 ; vnclip.wi v5, v5, 0 ; vnclip.wi v6, v6, 0 ; vnclip.wi v7, v7, 0 ;\
  vsetivli zero, 2, e32, m1, ta, ma                                    ;\
  vsse32.v v0, (a6), t0 ; add a6, a6, a7                               ;\
  vsse32.v v1, (a6), t0 ; add a6, a6, a7                               ;\
  vsse32.v v2, (a6), t0 ; add a6, a6, a7                               ;\
  vsse32.v v3, (a6), t0 ; add a6, a6, a7                               ;\
  vsse32.v v4, (a6), t0 ; add a6, a6, a7                               ;\
  vsse32.v v5, (a6), t0 ; add a6, a6, a7                               ;\
  vsse32.v v6, (a6), t0 ; add a6, a6, a7                               ;\
  vsse32.v v7, (a6), t0

# void fc3_tile(const int8_t *a, const int8_t *w, const int8_t *aend, const int32_t *bias,
#               const uint64_t *rqp, int32_t zo, int8_t *o, long ldo)
# rqp[n]: [31:0] M0, [37:32] shift (< 0), [47:40] zero point (DESIGN.md D5)
  .globl fc3_tile
  .type fc3_tile, @function
fc3_tile:
  vsetivli zero, 8, e32, m1, ta, ma
  vle32.v v24, (a3)                    # bias of the 8 columns
  li    t0, 8
  vlse32.v v25, (a4), t0               # M0
  addi  t1, a4, 4
  vlse32.v v26, (t1), t0               # rqp[63:32]: -(low 5 bits) = right shift
  vrsub.vx v26, v26, zero
  vmv.v.v v0, v24 ; vmv.v.v v1, v24 ; vmv.v.v v2, v24 ; vmv.v.v v3, v24
  vmv.v.v v4, v24 ; vmv.v.v v5, v24 ; vmv.v.v v6, v24 ; vmv.v.v v7, v24
  KLOOP
#define RQ_FC(r) vsmul.vv v##r, v##r, v25 ; vssra.vv v##r, v##r, v26 ; vadd.vx v##r, v##r, a5
  RQ_FC(0) ; RQ_FC(1) ; RQ_FC(2) ; RQ_FC(3)
  RQ_FC(4) ; RQ_FC(5) ; RQ_FC(6) ; RQ_FC(7)
  li    t0, 4
  STORE
  ret
  .size fc3_tile, . - fc3_tile

# void bmm3_tile(const int8_t *a, const int8_t *b, const int8_t *aend, const int32_t *rowc,
#                const int32_t *colc, const int32_t prm[4], int8_t *o, long ldo)
# acc(r, j) starts at rowc[r] + colc[j] (zero-point terms); prm = {M0, right shift, zo, stride};
# requantization is rq.a (ties away from zero).
  .globl bmm3_tile
  .type bmm3_tile, @function
bmm3_tile:
  vsetivli zero, 8, e32, m1, ta, ma
  vle32.v v24, (a4)
#define INIT_BMM(r) lw t0, 4 * r(a3) ; vadd.vx v##r, v24, t0
  INIT_BMM(0) ; INIT_BMM(1) ; INIT_BMM(2) ; INIT_BMM(3)
  INIT_BMM(4) ; INIT_BMM(5) ; INIT_BMM(6) ; INIT_BMM(7)
  KLOOP
  lw    t1, 0(a5)
  lw    t2, 4(a5)
  lw    t3, 8(a5)
#define RQ_BMM(r) vsmul.vx v##r, v##r, t1 ; vsra.vi v28, v##r, 31 ; vadd.vv v##r, v##r, v28 ;\
                  vssra.vx v##r, v##r, t2 ; vadd.vx v##r, v##r, t3
  RQ_BMM(0) ; RQ_BMM(1) ; RQ_BMM(2) ; RQ_BMM(3)
  RQ_BMM(4) ; RQ_BMM(5) ; RQ_BMM(6) ; RQ_BMM(7)
  lw    t0, 12(a5)
  STORE
  ret
  .size bmm3_tile, . - bmm3_tile
