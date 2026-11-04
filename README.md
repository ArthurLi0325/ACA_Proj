# Quantized NN Acceleration on RV64IMC: an INT8 MAC ISA Extension in Spike

EE6455 Advanced Computer Architecture term project.

The workload is the full INT8 DeiT-Tiny vision transformer (W8A8, 1.25 G MACs for each image).
It runs bare-metal on RV64IMC. The project adds custom instructions to the Spike ISA simulator.
It measures the number of retired instructions for each variant.
All variants give the same output as LiteRT, bit for bit, and predict the same class.

| Variant | Instructions per image | Speedup | Instructions per MAC |
|---|---:|---:|---:|
| RV64IMC scalar baseline | 4,321,715,305 | 1.00× | 3.447 |
| X1 `Xqmac`: `qmac8` + `rq` / `rq.a` | 575,100,016 | 7.51× | 0.459 |
| X1 with 4×4 software tiles | 469,972,566 | 9.20× | 0.375 |
| X2 `qop4` (outer product, 4×4 accumulators) | 311,983,134 | 13.85× | 0.249 |
| X2 `qop8` (outer product, 8×8 accumulators) | 221,898,131 | 19.48× | 0.177 |
| X3 `vqdot4.vx` (one output row in each vector register) | 232,143,212 | 18.62× | 0.185 |
| X3 `vqdot8.vx` (one output row in each vector register) | 185,962,191 | 23.24× | 0.148 |

These counts are for the synthetic input image. You do not need a dataset to get them.
The program and LiteRT both predict class 904.
The project is an ISA evaluation in Spike only. It does not include RTL or PPA results.

`DESIGN.md` gives the design decisions. `EXPERIMENTS.md` gives the procedure for each result.
The slides are in `docs/slides.pdf`.

## 1. Set up the environment (WSL2 Ubuntu 24.04)

Do these commands in sequence:

```bash
git clone https://github.com/ArthurLi0325/ACA_Proj.git ACA_project && cd ACA_project
scripts/setup_env.sh apt       # install the compiler, the Spike build tools and Python (sudo)
scripts/setup_env.sh spike     # build Spike with the three patches into ~/aca/install/bin/spike
scripts/setup_env.sh python    # make the Python venv with LiteRT in ~/aca/venv
scripts/setup_env.sh check     # check the installation
```

| Tool | Version and location |
|---|---|
| Compiler | `riscv64-unknown-elf-gcc` 13.2.0 (Ubuntu apt package), freestanding, no newlib |
| Spike | `riscv-isa-sim` commit `5a760667bf0d0ab893173b9eeff9c109c8d7dd9a`, with `patches/spike-xqmac.patch`, then `patches/spike-xqop.patch`, then `patches/spike-xqv.patch` |
| Python | `~/aca/venv`, packages in `requirements.txt` |

The Makefile uses `~/aca/install/bin/spike` if it is available. If not, it uses `spike` on PATH.
To use a different Spike, set `SPIKE=...`.
The ISA string is `rv64imc_zicntr`.
Programs with custom instructions need `EXT=xqmac`. This option gives `--extension=xqmac` to Spike.
If you do not set it, the first custom instruction causes a trap (mcause 2).
The X3 programs (`*_x3`, `xqv_test`) also need `VEC=1`. This option adds the integer vector unit:
the compiler uses `-march=rv64imc_zicsr_zicntr_zve64x_zvl256b` and Spike uses
`--isa=rv64imc_zicntr_zve64x_zvl256b` (VLEN = 256).

To apply the patches to a Spike source tree manually, do these commands:

```bash
cd ~/aca/riscv-isa-sim && git checkout 5a760667bf0d0ab893173b9eeff9c109c8d7dd9a -b xqmac
git apply /path/to/ACA_project/patches/spike-xqmac.patch
git apply /path/to/ACA_project/patches/spike-xqop.patch
git apply /path/to/ACA_project/patches/spike-xqv.patch
mkdir -p build && cd build && ../configure --prefix=$HOME/aca/install --enable-histogram
make -j$(nproc) && make install
```

## 2. Do the verification tests

```bash
make test                          # all tests, approximately 40 s
bash scripts/run_tests.sh insn     # instruction tests only, without Python or the model
```

The test set has these parts:

* Instruction tests. `xqmac_test` checks `qmac8`, `dot8`, `rq` and `rq.a`. `xqop_test` checks `qop4`,
  `qop8` and `qacc.*`. `xqv_test` checks `vqdot4.vx` and `vqdot8.vx`.
  The tests compare each instruction with a C reference model.
  They use edge cases and random operands.
* Trap test. This test makes sure that a custom instruction causes a trap without `EXT=xqmac`.
* Kernel tests. These tests run FC and BATCH_MATMUL on the block-0 tensors from LiteRT.
  They compare all output bytes.
* Block tests. These tests run encoder block 0 in the scalar, Xqmac, `qop8` and `vqdot8.vx` versions.
* No-FP check. This check makes sure that the binary has no floating-point instructions.

The last line of the output must be `21 passed, 0 failed`.

## 3. Do the full-workload benchmark

```bash
make bench                                          # synthetic image, all 7 variants
make bench INPUT=~/aca/data/deit/eval/0000.npy      # a real image
python3 scripts/run_benchmarks.py --variants x1,x2_qop8     # some variants only
```

The benchmark does these steps for each variant:

1. It exports the weights of all 561 TFLite operators to C headers. It also saves the LiteRT output
   of each operator as a reference.
2. It builds `src/tests/model`, `src/tests/model_x2` or `src/tests/model_x3` and runs it on Spike.
3. The program compares the stem output, the 12 block outputs and the 1000 logits with LiteRT.
4. The program prints the instruction count of each stage.

The script writes the results to `results/benchmarks.csv` and `results/benchmark_stages.csv`.
For the synthetic input, it compares each count with `results/expected_instret.csv`.
The tolerance is 0.5 %. The last line is `PASS` if all variants are bit-exact and in tolerance.

The expected output for the synthetic input is:

```
variant            instret  speedup  instr/MAC  class  exact  vs expected
scalar       4,321,715,305    1.00x      3.447    904    yes       +0.000%
x1             575,100,016    7.51x      0.459    904    yes       +0.000%
x1_tile4       469,972,566    9.20x      0.375    904    yes       +0.000%
x2_qop4        311,983,134   13.85x      0.249    904    yes       +0.000%
x2_qop8        221,898,131   19.48x      0.177    904    yes       +0.000%
x3_qdot4       232,143,212   18.62x      0.185    904    yes       +0.000%
x3_qdot8       185,962,191   23.24x      0.148    904    yes       +0.000%
```

The count is the number of retired instructions (`minstret`) for the full inference.
It starts at the int8 image and stops at the 1000 logits.
The count does not include the reference checks and the one-time table setup (approximately 0.5 M).
The X2 and X3 weight packing occurs offline, and the count does not include it.

## 4. Custom instructions

All instructions are R-type. The C and assembly code use `.insn` (`src/common/xqmac.h`, `src/common/xqop.h`,
`src/common/xqv.h`). Thus, binutils needs no change.

Xqmac, custom-0 (`0001011`):

| Instruction | Operation |
|---|---|
| `qmac8 rd, rs1, rs2` | rd = sext32(rd[31:0] + Σ_{i<8} s8(rs1.b[i]) · s8(rs2.b[i])) |
| `dot8 rd, rs1, rs2` | rd = Σ_{i<8} s8(rs1.b[i]) · s8(rs2.b[i]) |
| `rq rd, rs1, rs2` | rd = clamp(zp + round(rs1 · M0 · 2^−31 · 2^shift), −128, 127); rs2 = {zp[47:40], shift[37:32], M0[31:0]}; ties toward +∞ |
| `rq.a rd, rs1, rs2` | the same as `rq`, with ties away from zero (gemmlowp) |

X2, custom-1 (`0101011`). The core has an internal 8×8 int32 accumulator array ACC.

| Instruction | Operation |
|---|---|
| `qop4 rs1, rs2` | ACC[i][j] += Σ_{t<2} rs1.b[2i+t] · rs2.b[2j+t], i, j < 4 (32 MACs) |
| `qop8 rs1, rs2` | ACC[i][j] += rs1.b[i] · rs2.b[j], i, j < 8 (64 MACs) |
| `qacc.col rs1, c` | set ACC[i][2c] and ACC[i][2c+1] for all i (bias) |
| `qacc.row rs1, c` | add to ACC[2c][j] and ACC[2c+1][j] for all j (zero-point terms) |
| `qacc.rq rd, rs1, n` / `qacc.rqa` | rd = rq(ACC[n], rs1) / rq.a(ACC[n], rs1) |

X3, custom-2 (`1011011`). The accumulators are in the standard vector registers. One vector register
holds one output row: 8 int32 lanes are 8 output columns (VLEN = 256, SEW = 32, i < vl).
There is no extra state. The requantization uses standard RVV instructions (`vsmul`, `vssra`, `vnclip`).

| Instruction | Operation |
|---|---|
| `vqdot4.vx vd, rs1, vs2` | vd[i] += Σ_{t<4} rs1.b[t] · vs2.e32[i].b[t] (32 MACs; as `vqdot.vx` in Zvqdotq) |
| `vqdot8.vx vd, rs1, vs2` | vd[i] += Σ_{t<8} rs1.b[t] · vs2.e64[i].b[t] (64 MACs; vs2 is a group of 2 registers) |

## 5. Run one program

```bash
make export-model [INPUT=...]                           # headers for model, model_x2 and model_x3
make -B TEST=model run OPT="-O2 -DSCALAR"                # scalar baseline
make -B TEST=model run EXT=xqmac                         # X1
make -B TEST=model run EXT=xqmac OPT="-O2 -DTILE4"       # X1, 4x4 software tiles
make -B TEST=model_x2 run EXT=xqmac                      # X2 qop4
make -B TEST=model_x2 run EXT=xqmac OPT="-O2 -DQOP8"     # X2 qop8
make -B TEST=model_x3 run EXT=xqmac VEC=1                # X3 vqdot4.vx
make -B TEST=model_x3 run EXT=xqmac VEC=1 OPT="-O2 -DQDOT8"   # X3 vqdot8.vx

make export                                             # block 0 only (BLOCK=n for a different block)
make -B TEST=block run EXT=xqmac                         # one encoder block
make -B TEST=fc_x2 run EXT=xqmac OPT="-O2 -DQOP8"        # one FC layer
make TEST=block hist HIST_ARGS="--funcs fc"             # instruction mix
make TEST=model checkfp                                 # check for FP instructions
```

## 6. Make the quantized model again (optional)

The repository includes `models/deit_tiny_w8a8.tflite`. The tests and the benchmark do not need this step.

```bash
scripts/get_model.sh          # get DeiT-Tiny, calibrate, quantize to W8A8 and W8A16
scripts/get_model.sh eval     # top-1 accuracy: W8A8 82.6 %, W8A16 83.2 % (500 images)
```

## Repository layout

```
src/common/        bare-metal runtime, xqmac.h, xqop.h, xqv.h, qnn.h (requantization), vit.h (ViT operators)
src/tests/<t>/     one program in each directory: model, model_x2, model_x3, block, block_x2, block_x3,
                   fc_*, bmm*, *_test (fc_x3/kern.S: the X3 tile kernels)
scripts/           environment setup, tests, benchmark, model export and quantization
patches/           Spike patches
models/            quantized model
results/           expected and measured instruction counts
DESIGN.md          design decisions
EXPERIMENTS.md     procedure for each result
```
