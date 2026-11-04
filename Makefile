# Build and run bare-metal programs on Spike.
#   make TEST=hello            build build/hello.elf (+ .dump)
#   make TEST=hello run        run on Spike
#   make TEST=hello hist       run with PC histogram -> results/hello.hist
#   make TEST=hello checkfp    verify no floating-point instructions/calls
#   make TEST=xqmac_test run EXT=xqmac   run with the Xqmac custom instructions (patched Spike)
#   make TEST=xqv_test run EXT=xqmac VEC=1   X3 tests: vector unit (Zve64x, VLEN 256) enabled
#   make export                block-0 weights/params + golden tensors -> build/model, build/golden
#   make export-model          whole model (stem, 12 blocks, head) + golden tensors -> build/model
#   make TEST=model run EXT=xqmac      whole-model inference (README lists every variant)
#   make test                  verification tests (scripts/run_tests.sh)
#   make bench                 whole-model benchmark of every variant (scripts/run_benchmarks.py)
#   INPUT=synthetic|<img.npy>  input image for export / export-model / bench
# Run inside WSL (Ubuntu).

TEST    ?= hello
PREFIX  ?= riscv64-unknown-elf-
CC      := $(PREFIX)gcc
OBJDUMP := $(PREFIX)objdump
SIZE    := $(PREFIX)size

SPIKE   ?= $(HOME)/aca/install/bin/spike
ifeq ($(wildcard $(SPIKE)),)
SPIKE   := spike
endif
ISA     ?= rv64imc_zicntr
EXT     ?=
SPIKE_FLAGS = --isa=$(ISA) $(if $(EXT),--extension=$(EXT))
PY      ?= $(HOME)/aca/venv/bin/python
MODEL   ?= models/deit_tiny_w8a8.tflite
BLOCK   ?= 0
MARCH   ?= rv64imc_zicsr_zicntr
OPT     ?= -O2
VEC     ?=
ifneq ($(VEC),)
# X3 (row-vector dot products) needs the integer vector unit, VLEN = 256
MARCH   := rv64imc_zicsr_zicntr_zve64x_zvl256b
ISA     := rv64imc_zicntr_zve64x_zvl256b
endif

CFLAGS  := -march=$(MARCH) -mabi=lp64 -mcmodel=medany $(OPT) -g \
           -ffreestanding -fno-builtin -nostdlib -nostartfiles \
           -Wall -Wextra -Isrc/common -Ibuild/model
LDFLAGS := -T src/common/link.ld -Wl,--gc-sections -Wl,--no-warn-rwx-segments -lgcc

COMMON  := src/common/crt.S src/common/runtime.c
SRCS    := $(COMMON) $(wildcard src/tests/$(TEST)/*.c src/tests/$(TEST)/*.S)
ELF     := build/$(TEST).elf

INPUT   ?=

.PHONY: all run hist checkfp size export export-model test bench clean

all: $(ELF)

$(ELF): $(SRCS) $(wildcard src/common/*.h src/tests/$(TEST)/*.h) src/common/link.ld | build
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS)
	$(OBJDUMP) -d -M no-aliases,numeric $@ > build/$(TEST).dump
	$(OBJDUMP) -d $@ > build/$(TEST).asm

build results:
	mkdir -p $@

run: $(ELF)
	$(SPIKE) $(SPIKE_FLAGS) $(ELF)

hist: $(ELF) | results
	$(SPIKE) $(SPIKE_FLAGS) -g $(ELF) 2> results/$(TEST).hist
	python3 scripts/insn_mix.py build/$(TEST).dump results/$(TEST).hist $(HIST_ARGS)

checkfp: $(ELF)
	scripts/check_no_fp.sh $(ELF)

size: $(ELF)
	$(SIZE) $(ELF)

# EXPORT_ARGS="--input ~/aca/data/deit/eval/0001.npy" picks another image
export:
	$(PY) scripts/export_block.py $(MODEL) --block $(BLOCK) $(if $(INPUT),--input $(INPUT)) $(EXPORT_ARGS)

export-model:
	$(PY) scripts/export_model.py $(MODEL) $(if $(INPUT),--input $(INPUT))

test:
	PY=$(PY) bash scripts/run_tests.sh

bench:
	PY=$(PY) python3 scripts/run_benchmarks.py $(if $(INPUT),--input $(INPUT))

clean:
	rm -rf build
