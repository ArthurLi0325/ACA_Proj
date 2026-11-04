# Design Decisions

Each decision: options → trade-offs → choice → supporting measurement.

## D1. Model and evaluation scope
**DeiT-Tiny** (timm `deit_tiny_patch16_224.fb_in1k`, from `litert-community`), 5.65 M weights,
1.254 GMAC per inference.

| Op | Count | MACs | % MAC |
|---|---:|---:|---:|
| FULLY_CONNECTED | 49 | 1,045,949,952 | 83.43% |
| BATCH_MATMUL (QKᵀ, AV) | 24 | 178,831,872 | 14.26% |
| CONV_2D (16×16/16 patch embedding) | 1 | 28,901,376 | 2.31% |

99.97% of MACs are GEMM-shaped, which maps directly onto a packed dot-product
instruction. All 12 encoder blocks are identical, so one block is a natural
evaluation unit, and full-model results can be validated against it.

Alternatives considered: ResNet-18 (ready-made INT8 model, but 1.8 GMAC and not a
Transformer), MobileNetV3-Large (0.22 GMAC, but depthwise conv maps poorly onto a
reduction-style 8-wide MAC).

## D2. Quantization scheme (per-channel / zero-point handling)
litert-community provides DeiT-Tiny only in float32, so we quantize it ourselves with
`ai-edge-quantizer` (`STATIC_WI8_AI8`, 128 calibration images; `scripts/get_model.sh`):
weights INT8 symmetric per-channel, activations INT8 asymmetric per-tensor, int32 bias.

Accuracy on 500 mini-imagenet validation images (100 ImageNet classes, so higher than
the 72.2% full ImageNet top-1):

| Model | Top-1 | Agreement with float |
|---|---:|---:|
| float32 | 85.2% | — |
| **W8A8** | **82.6%** | 91.6% |
| W8A16 | 83.2% | 97.2% |
| W8A8, random-noise calibration | 7.4% | 7.8% |

All compute ops (FC, BATCH_MATMUL, CONV_2D, SOFTMAX, GELU, and the LayerNorm
decomposition MEAN/SUB/MUL/RSQRT/ADD) are INT8. The only float ops are 36 SQUEEZE
ops, each wrapped in DEQUANTIZE→SQUEEZE→QUANTIZE when the fused QKV tensor is split
into Q, K, V. 35 of these change scale/zero-point, so they are implemented as integer
int8→int8 requantization with a fixed-point multiplier computed offline; no floating
point at run time. (Possible ±1 LSB mismatch vs. the float reference path, to be checked.)

**Decision: fold the activation zero point into the FC/conv bias.** Weights are
symmetric (zp_w = 0), so `Σ(a − zp_a)·w + b = Σ a·w + (b − zp_a·Σw)`; the bracket is computed
offline by `scripts/export_block.py` and stored as the int32 bias, so the FC inner loop is a pure
signed int8 × signed int8 dot product. Folded biases of block 0 stay within int32. BATCH_MATMUL
(QKᵀ, AV) has activations on both sides, so its zero points cannot be folded offline.

Observation from the export check (EXPERIMENTS.md E1): with the same M0/shift, LiteRT's
FULLY_CONNECTED matches a requantization whose final right shift rounds ties toward +∞
(100% bit-exact), while BATCH_MATMUL and the SQUEEZE requantizations match gemmlowp's
round-half-away-from-zero (100%). Using the other rule gives ±1 LSB on 0.1–0.5% of outputs.
Input to D5 (requant rounding semantics).

## D3. Data layout and software optimization (B1)
**Choice: 2×4 register blocking + 4× unrolling of K**, weights kept row-major
[N][K] (K contiguous). Each `tile_2x4` call computes 2 rows × 4 output channels with 8 int32
accumulators in registers: every loaded activation is used 4×, every weight 2×, so the inner
loop needs 6 loads per 8 MACs instead of 16. Unrolling K by 4 with constant load offsets pays the
6 pointer increments and the branch once per 32 MACs (95 instructions / 32 MACs = 2.97 in the
loop). GCC's pre-RA scheduler hoisted all 24 loads and spilled accumulators, so `tile_2x4` is
compiled with `optimize("no-schedule-insns")`.

Alternatives: unrolling only (≈4.4 instr/MAC predicted, still 2 loads/MAC); larger tiles (4×4
needs 16 accumulators + 8 pointers + loaded values, more than the 31 integer registers); SWAR
with `ld` + byte extraction (2 instructions per extracted byte on RV64IMC, no gain). Weight
interleaving would save 3 of the 6 pointer increments (~0.1 instr/MAC) but breaks the K-contiguous
layout a packed-MAC instruction needs.

Result on block-0 QKV (EXPERIMENTS.md B0-QKV / B1-QKV): 7.284 → 3.277 instr/MAC (2.22× fewer
instructions), ld+st/MAC 2.06 → 0.83. B1 is now close to the scalar floor: every MAC still needs
one `mulw` and one `addw` (61% of B1's instructions), so going below 2 instr/MAC requires doing
several MACs per instruction. Requantization grew from 2.5% to 5.6% of the FC.

## D4. Xqmac semantics (operands, accumulator width, signedness)
**Choice: accumulate into rd as the primary instruction, with a non-accumulating
dot product implemented alongside for comparison.**

- `qmac8 rd, rs1, rs2` (primary): `rd = sext32(rd[31:0] + Σ_{i=0..7} sext(rs1.byte[i]) · sext(rs2.byte[i]))`
- `dot8 rd, rs1, rs2` (comparison): `rd = sext32(Σ_{i=0..7} sext(rs1.byte[i]) · sext(rs2.byte[i]))`, accumulated with `addw`

Operands are 8 packed signed int8 values per 64-bit register (byte i = bits 8i+7..8i). Signed ×
signed is enough: FC is purely signed after zero-point folding (D2); BATCH_MATMUL uses
`Σ(a−za)(b−zb) = Σab − zb·Σa − za·Σb + K·za·zb`, where the row sums are computed once per matrix.
The accumulator is 32 bits with wrap-around like `addw`: the worst case in DeiT-Tiny is FC2,
|acc| ≤ 768·2^14 + max|bias| = 13.5M, 160× below 2^31 (int16 is too small: one product reaches 2^14).

Predicted with B1's 2×4 tiling (6 `ld` per 64 MACs): qmac8 21 instructions / 64 MACs ≈ 0.33
instr/MAC (≈10× fewer than B1); dot8 + addw 29 / 64 ≈ 0.45 (≈7×).
Trade-off: qmac8 reads three registers, but CV-Wally's `regfile.sv` has two read ports (a1/rd1,
a2/rd2), so RTL needs a third port or an extra read cycle; dot8 fits the existing ports. Both are
measured in Spike; the RTL choice is deferred to the final phase, when area and timing are known.
Alternatives not taken: R4-type with rs3 as accumulator (same port cost, more encoding space);
a hidden accumulator file (new architectural state, clear/read-out instructions, context save).

Measured in Spike (EXPERIMENTS.md X1), same 2×4 tiling with K unrolled 2× (16 bytes per
iteration): the inner loop is 35 instructions per 128 MACs (12 `ld`, 16 `qmac8`, 6 `addi`, 1 `bne`).
QKV goes from 3.277 (B1) to 0.583 instr/MAC with `qmac8` (5.6× fewer) and 0.704 with `dot8` + `add`
(4.7×), so accumulating in rd saves 17% of the instructions. All four FC layers stay bit-exact.
The same instructions cover BATCH_MATMUL with the row/column-sum correction above (row sums are
`qmac8` against 0x0101010101010101): QKᵀ 4.013 → 0.580 and AV 3.432 → 0.388 instr/MAC, bit-exact
with `rq.a` (EXPERIMENTS.md BMM-QK, BMM-AV). AV needs P padded to a multiple of 8 bytes per row and
V transposed (0.79M instructions of packing, 27% extra), which the producing ops can absorb.

## D5. Requantization instruction semantics (parameters, rounding, saturation)
**Choice: one fused instruction, parameters packed in rs2, TFLite-exact rounding
with the tie rule selected by funct3.**

`rq rd, rs1, rs2` / `rq.a rd, rs1, rs2`:
```
acc   = rs1[31:0]                      (int32)
M0    = rs2[31:0]                      (int32, Q31 multiplier)
sh    = sext(rs2[37:32])               (6-bit signed shift; > 0 = left)
zp    = sext(rs2[47:40])               (int8 output zero point)
v     = SRDHM(acc << max(sh, 0), M0)   (gemmlowp SaturatingRoundingDoublingHighMul)
r     = max(-sh, 0)
rq  : v = r ? (v + 2^(r-1)) >>arith r : v          ties toward +inf (LiteRT FULLY_CONNECTED)
rq.a: v = RoundingDivideByPOT(v, r)                 ties away from 0 (BATCH_MATMUL, SQUEEZE)
rd    = sext64(clamp(v + zp, -128, 127))
```
rs2[39:38] and rs2[63:48] are reserved (zero). The exporter packs one 64-bit parameter word per
output channel (per-channel FC) or per tensor (BMM, requant), so one `ld` per channel serves both
rows of a 2×4 tile. The software version costs 35.5 instructions per output; `rq` makes it 1
(plus the amortized parameter load and the `sb`). FC/conv (−15..−7 over 21,928 channels) and BMM
(−9..−6) only shift right; the SQUEEZE requantizations (block 0: ops 23, 26, 29) use shift = +1, so the
left-shift path is exercised by DeiT-Tiny too. The left shift wraps in 32 bits; with SQUEEZE inputs of at most ±255 (after subtracting the input zero point) it never overflows. SQUEEZE requantization subtracts its
input zero point in software first. Two tie rules keep FC, BMM and SQUEEZE 100% bit-exact with
LiteRT (DESIGN D2 observation); the hardware cost is one multiplexer on the rounding increment.
Alternatives not taken: zero point in a CSR (extra architectural state, no instruction saved);
fusing only SRDHM + shift (~5–6 instructions per output left); single rounding (simplest
hardware, but ~0.1% of FC outputs off by 1 LSB).
Note: SRDHM itself rounds exact halves toward +∞ for both signs (gemmlowp's negative nudge is
1 − 2^30 followed by a truncating division); the tie rule selected by funct3 only applies to the
final right shift.

Measured (EXPERIMENTS.md X1, X1+RQ): with `qmac8` and software requantization, requant + SRDHM are
31.7% of the QKV instructions (D8 predicted 29–36%). With `rq` the whole requantization is one
instruction per output and QKV drops from 0.583 to 0.363 instr/MAC (1.61×; 9.0× fewer than B1, 20×
fewer than B0), still bit-exact on all four FC layers.

## D6. Encoding (opcode / funct allocation)
**Choice: all Xqmac instructions are R-type in custom-0 (`0x0B`), funct7 = 0,
selected by funct3; custom-1 (`0x2B`) stays free for the final-phase extension (X2).**

| funct3 | Mnemonic | Semantics | MATCH | MASK |
|---|---|---|---|---|
| 000 | `qmac8` | rd = sext32(rd[31:0] + Σ_{i<8} s8(rs1.b[i])·s8(rs2.b[i])) | `0x0000000b` | `0xfe00707f` |
| 001 | `dot8` | rd = sext32(Σ_{i<8} s8(rs1.b[i])·s8(rs2.b[i])) | `0x0000100b` | `0xfe00707f` |
| 010 | (reserved `qmac4h`) | W8A16: rd += Σ_{i<4} s16(rs1.h[i])·s8(rs2.b[i]) | `0x0000200b` | `0xfe00707f` |
| 011 | reserved | — | | |
| 100 | `rq` | requantize, ties toward +inf (D5) | `0x0000400b` | `0xfe00707f` |
| 101 | `rq.a` | requantize, ties away from 0 (D5) | `0x0000500b` | `0xfe00707f` |
| 110–111 | reserved | — | | |

In C: `.insn r 0x0B, <funct3>, 0, rd, rs1, rs2`. All are 32-bit instructions (no compressed forms).
Wrappers: `src/common/xqmac.h`. Spike implementation: `patches/spike-xqmac.patch` (adds
`riscv/xqmac.cc`, enabled with `--extension=xqmac`). It is linked into the spike binary rather than
built as a `libcustomext.so` plugin, because loading any customext library aborts on this Spike
version ("Plugin imsic_mmio already registered").

## D7. Profiling methodology
Spike PC histogram (`-g`) + objdump → `scripts/insn_mix.py`. See README.

## D8. Three largest bottlenecks (B0/B1 profiling)
**Choice**, from block-0 QKV (EXPERIMENTS.md B0-QKV, B1-QKV):

1. **One scalar multiply and one add per MAC.** 27.5% of B0, 61% of B1; no scalar code goes below
   2 instr/MAC → packed MAC (`qmac8`, D4).
2. **Byte-granular loads.** 2 `lb`/MAC in B0 (28%), 0.75 in B1 (23%); register blocking reuses data but
   each load still brings 1 byte of a 64-bit datapath → 64-bit packed operands for `qmac8`.
3. **Requantization.** A fixed 35.5 instructions per output: 2.5% of B0, 5.6% of B1, and an estimated
   29–36% (K = 192) or 9–12% (K = 768) once the MAC loop uses `qmac8` → requantization instruction (D5).

Pointer increments + loop branches were 42% of B0 but only ~8.5% of B1, so software removes most of
it; it is reported with B1 and is a candidate for the final extension (hardware loops,
post-increment loads). Cache misses and load-use stalls cannot be measured in Spike. The share of
SOFTMAX, LayerNorm and GELU, unknown when this was decided, is now measured in D9.

## D9. Non-MAC operators in integer arithmetic, and the whole block
*D11 corrects three rows of this table (MEAN, ADD/SUB, SOFTMAX) and gives the final formulations. The numbers below are for the formulations in this table.*

**Implementation (no new instructions):** every op of block 0 (TFLite
ops 10..54) runs integer-only on RV64IMC and the block output matches LiteRT bit-exactly, in both the
scalar baseline and the Xqmac build (`src/tests/block`, EXPERIMENTS.md BLOCK-S / BLOCK-X). LiteRT's x86
kernels use float for MEAN, SOFTMAX and the GELU table, so those were rewritten in integer form;
`export_block.py` checks each formulation against the LiteRT tensors ("elementwise / table ops").

| Op | LiteRT x86 | Integer form used here |
|---|---|---|
| MEAN | float | `rq.a(Σx − N·zp)`, multiplier s_in / (s_out·N) |
| ADD / SUB | integer: each input `(q − zp) << 20`, rescaled to 2·max(s_a, s_b), then added and requantized | the input stage depends on one int8 value, so it is a 256-entry int32 table per input (built once); output with `rq.a` |
| MUL | integer | `rq.a((a − za)(b − zb))` |
| RSQRT | integer Newton–Raphson | 256-entry int8 table (one lookup per row) |
| GELU (exact, erf) | float, tabulated | the same 256-entry int8 table, generated offline |
| MUL by 1/√64 (op 34) | integer | 256-entry table, folded into softmax |
| SOFTMAX | float exp table, float sum, 1/(sum·scale) | table E[i] = round(exp(−s·β·(255 − i))·2²⁴); per row S = Σe, R = ⌊2⁴⁶/S⌋ (one division), out = ((e·R·256 + 2⁴⁵) >> 46) + zo |

Where LiteRT computes in float (MEAN, SOFTMAX), bit-exactness is verified on the exported input, not
proven for every input: a value within float rounding error of a .5 boundary could differ by 1 LSB.
LayerNorm is fused per row (one pass for mean, one for x − mean and the squares, one for the rest).
Layout choices remove all data movement ops: the Q/K requantization step also writes V transposed as
[64][200] (so AV needs no packing, BMM-AV's 0.79M packing instructions disappear), softmax works in
place on [197][200] rows with zero padding, and AV writes into [197][192] with a row stride of 192 (the
op 38 transpose).

**Measured (WSL Spike, image 0000.npy):** the whole block drops from 355,093,282
to 46,409,757 instructions (7.65×). FC + BMM go 9.05× faster and fall from 94.2% to 79.7% of the
block; the non-MAC operators go only 2.17× faster (LayerNorm 3.9× from `rq.a`, softmax and GELU
unchanged) and rise from 5.8% to 20.3%: softmax 8.0% (32 instr per score), LayerNorm ×2 6.0%
(37 instr per element), residual ADD ×2 2.9%, GELU 2.0%, Q/K requant 1.2%. Even with free MACs the
block could get at most 4.9× faster than X1 (Amdahl), so the X2 choice should weigh the remaining MAC
overhead (pointer increments, loads, per-tile calls) against softmax / LayerNorm (byte loads and
stores, no min/max, 64-bit multiply per score). Which one X2 targets is the student's decision.

## D10. X2: outer-product MAC with internal accumulators (custom-1)
**Choice:** after D9, `qmac8` itself was the largest single cost (12.8M of the 37.8M
instructions left once the FC/BMM kernels use 4×4 tiles in software, EXPERIMENTS.md A-BLOCK), so X2
raises the MACs per instruction. A post-increment load was rejected: after the 4×4 tiles it would
remove only ≈ 2% and it needs a second register-file write port. Softmax/LayerNorm helpers are left
for later. **Both sizes, 4×4 and 8×8, are built; the RTL is parameterized and DC area decides**
(performance per area for the final report).

| funct3 | Mnemonic | Semantics |
|---|---|---|
| 000 | `qop4 rs1, rs2` | ACC[i][j] += Σ_{t<2} s8(rs1.b[2i+t])·s8(rs2.b[2j+t]), i, j < 4 (32 MACs) |
| 001 | `qop8 rs1, rs2` | ACC[i][j] += s8(rs1.b[i])·s8(rs2.b[j]), i, j < 8 (64 MACs) |
| 010 | `qacc.col rs1, c` | ACC[i][2c] = rs1[31:0], ACC[i][2c+1] = rs1[63:32] for every row i (bias) |
| 101 | `qacc.row rs1, c` | ACC[2c][j] += rs1[31:0], ACC[2c+1][j] += rs1[63:32] for every column j (BMM zero-point terms) |
| 011 | `qacc.rq rd, rs1, n` | rd = rq(ACC[n], rs1), the D5 requantization, ties toward +inf |
| 100 | `qacc.rqa rd, rs1, n` | same, ties away from zero |

ACC is an internal 8×8 array of int32 (row-major index n = 8i + j); the 4×4 version uses its top-left
quarter. `c` is the rs2 field and `n` is {funct7[0], rs2 field}, so readout needs no index register.
Each instruction reads at most 2 registers and writes at most 1, so Wally's register file is unchanged.
Operands are 8 bytes of a K-interleaved layout: qop4 [rows/4][K/2][4][2], qop8 [rows/8][K][8], rows
padded to a multiple of 4 / 8 with zeros. Weights are packed offline; every activation producer in the
block writes one byte at a time, so it can write the packed layout directly.

Prototype (Spike `patches/spike-xqop.patch` on top of `spike-xqmac.patch`, `src/common/xqop.h`,
`src/tests/fc_x2`): all four FC layers bit-exact. The four layers take 29.8M instructions with X1 + `rq`,
23.1M with 4×4 software tiles, **12.0M with qop4** and **6.3M with qop8** (QKV 0.363 → 0.281 → 0.143 →
0.075 instr/MAC). qop8's inner loop is 15 instructions per 256 MACs, 8 of them loads, so loads limit
any larger tile. Hardware: qop4 has 32 int8 multipliers and 16×32-bit accumulators; qop8 has 64
multipliers and 64×32 bits of accumulators, about the size of the integer register file (31×64), but
each lane is one product plus one add (qop4: two products plus a 3-input add).

**Whole block on the prototype (`src/tests/block_x2`, bit-exact, WSL Spike):** every producer writes its
FC/BMM operand directly in the packed layout and counts the row sums the BMM zero-point terms need, so
there is no separate packing step. BMM: QKᵀ 0.208 / 0.112 and AV 0.145 / 0.075 instr/MAC (qop4 /
qop8; A: 0.415 / 0.286).

| Version | Block instructions | vs scalar | MAC part (FC + BMM) | Non-MAC ops |
|---|---|---|---|---|
| Scalar (BLOCK-S) | 355.09M | 1× | 94% | 6% |
| X1: `qmac8` + `rq` (BLOCK-X) | 46.41M | 7.65× | 80% | 20% |
| A: X1 + 4×4 software tiles | 37.80M | 9.39× | 75% | 25% |
| X2 qop4 | 24.69M | 14.4× | 59% (14.6M) | 41% (10.1M) |
| X2 qop8 | 17.30M | 20.5× | 44% (7.7M) | 56% (9.7M) |

With qop8, softmax (3.8M, 22%) is the largest single stage and the non-MAC operators are the majority,
so the next step after X2 would be softmax / LayerNorm, not more MAC width.

## D11. Whole model, and the final LiteRT-exact formulations
**Scope:** the course asks for an ISA evaluation in Spike
with a full-workload benchmark, and no PPA. The full workload is the whole DeiT-Tiny inference:
int8 image in, 1000 int8 logits and the predicted class out.

**Graph.** All 561 TFLite ops run in integer arithmetic (`src/tests/model`, `model_x2`).

| Ops | What | Implementation |
|---|---|---|
| 0–1 | TRANSPOSE + CONV_2D 16×16, stride 16 | FC over the 196 patches. im2col reads the NCHW image directly. Weights are reordered offline to (c, kh, kw). |
| 2–4 | reshape / transpose | layout only, no code |
| 5–9 | class token, 2 QUANTIZE, CONCATENATION, ADD position embedding | `rq.a` requantization, then a residual ADD with a constant |
| 10–549 | 12 encoder blocks, 45 ops each | `vit_block()` in `src/common/vit.h` (one code path for block, block_x2, model, model_x2) |
| 550–558 | final LayerNorm | on the class-token row only. LayerNorm is per row, and op 559 keeps only that row. |
| 559–560 | STRIDED_SLICE, FC 192 → 1000 | the head FC with M = 1 |

**Corrections found in blocks 1–11.** Block 0 was bit-exact with the D9 formulations. Other blocks
were not. Each fix below makes all 398 checked operators match LiteRT on 100 % of their outputs.

1. **MEAN.** LiteRT does not use s_in / (s_out·N). It quantizes s_in / s_out to (M, e). Then it
   divides by N in fixed point: sh = min(⌊log2 N⌋, 32, 31 + e), M' = ⌊(M << sh) / N⌋, e' = e − sh.
   The shift can become positive. Blocks 2–10 differed by 1 LSB before this fix.
2. **ADD / SUB.** The x86 build uses TFLITE_SINGLE_ROUNDING. Each stage is (x·m + 2^(t−1)) >> t with
   t = 31 − shift. It is not the gemmlowp double rounding. One output of op 275 differed. In C, the
   output multiplier is distributed into two 64-bit tables per ADD:
   A[q] = stage_a(q)·m_out + 2^(t−1) + (zp_out << t) and B[q] = ±stage_b(q)·m_out.
   Then y = clamp((A[a] + B[b]) >> t). This is 2 loads, 1 add, 1 shift and 1 clamp per element. The
   clamp is `rq` with an identity parameter word.
3. **SOFTMAX.** LiteRT computes it in float32: T[255 − v] = expf(−s·v), a sequential float32 sum,
   inv = 1 / (sum·s_out), out = round(T·inv) + zp_out. The D9 integer form gave the same result except
   near ties (99.997 %). The C code now uses an integer fast path with a guard:
   * e = round(T·2^24) from a table. The row maximum gives e = 2^24, so sum ≥ 2^24.
   * R = ⌊2^54 / sum⌋ (one division per row) and q = e·R.
   * The fast path rounds q with a margin: hi = (q + (q >> 15) + 2^45 + 2^30) >> 46 and
     lo = (q − (q >> 15) + 2^45 − 2^30) >> 46. These bound the float32 error.
   * If hi = lo, the result is certain. Otherwise the row runs an exact integer emulation of the
     float32 operations (add, multiply, divide with round-to-nearest-even, `qnn.h`). This happens
     rarely: 0 rows for most inputs.

   The scale table (op 34, ×1/√64) is monotonic. The setup checks this. So the row maximum is taken
   on the raw scores.

With these, the program needs no floating-point instruction. The integer operators follow LiteRT's
arithmetic step by step, and softmax reproduces LiteRT's float32 steps exactly (given the same exp
table, which is exported). Bit-exactness is checked on the exported image; it was not run on a whole
dataset.

**Measured (synthetic input, `make bench`):** the whole model takes 4,321.7M instructions scalar, 575.1M with X1 (7.51×),
470.0M with X1 4×4 tiles (9.20×), 312.0M with qop4 (13.85×) and 221.9M with qop8 (19.48×). All
variants predict class 904, the same as LiteRT. With qop8, the MAC layers are 42 % of the
instructions, softmax is 24 % and LayerNorm 15 %. So the next instruction to add would target
softmax or LayerNorm, not a wider MAC (as in D10).

