#!/usr/bin/env bash
# Verification tests on Spike (run inside WSL from the repo root).
#   scripts/run_tests.sh          instruction tests, then kernel and block tests (bit-exact vs LiteRT)
#   scripts/run_tests.sh insn     instruction tests only (no model or Python needed)
# The instruction tests compare every custom instruction with an independent C reference model on
# directed edge cases and random operands. The kernel tests run FC / BATCH_MATMUL / whole block 0 on
# LiteRT tensors of block 0 (exported once with the synthetic image if build/model is missing).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/aca/venv/bin/python}
pass=0
fail=0

# t "what" PASS|TRAP make-arguments...
t() {
  local what=$1 expect=$2 out ok=0
  shift 2
  out=$(make -s -B "$@" run 2>&1)
  if [[ $expect == TRAP ]]; then
    grep -q "TRAP mcause=2" <<<"$out" && ok=1
  else
    grep -q "^PASS" <<<"$out" && ok=1
  fi
  if ((ok)); then
    printf '  [PASS] %s\n' "$what"
    pass=$((pass + 1))
  else
    printf '  [FAIL] %s   (make %s run)\n' "$what" "$*"
    tail -6 <<<"$out" | sed 's/^/         /'
    fail=$((fail + 1))
  fi
}

echo "Instruction tests (Spike with patches/spike-xqmac.patch + spike-xqop.patch + spike-xqv.patch)"
t "Xqmac: qmac8 dot8 rq rq.a, directed + 100k random vs C model" PASS TEST=xqmac_test EXT=xqmac
t "X2: qop4 qop8 qacc.col/row/rq/rqa, directed + random vs C model" PASS TEST=xqop_test EXT=xqmac
t "X3: vqdot4.vx vqdot8.vx, directed + random vs C model" PASS TEST=xqv_test EXT=xqmac VEC=1
t "without --extension=xqmac the custom opcode traps (mcause 2)" TRAP TEST=xqmac_test

if [[ ${1:-} != insn ]]; then
  if [[ ! -f build/model/block0_params.h ]]; then
    echo "exporting block 0 (synthetic image) ..."
    "$PY" scripts/export_block.py models/deit_tiny_w8a8.tflite --block 0 --input synthetic >/dev/null || exit 1
  fi
  echo "Kernel tests (block 0 tensors from LiteRT, every output bit-exact)"
  t "FC QKV, scalar B1"                 PASS TEST=fc_b1
  t "FC QKV, qmac8 + rq"                PASS TEST=fc_x1 EXT=xqmac OPT="-O2 -DUSE_RQ"
  t "BMM QK^T, qmac8 + rq.a"            PASS TEST=bmm EXT=xqmac
  t "BMM AV, qmac8 + rq.a"              PASS TEST=bmm EXT=xqmac OPT="-O2 -DBMM_SEL=1"
  t "FC QKV, qop4"                      PASS TEST=fc_x2 EXT=xqmac
  t "FC QKV, qop8"                      PASS TEST=fc_x2 EXT=xqmac OPT="-O2 -DQOP8"
  t "BMM QK^T, qop4"                    PASS TEST=bmm_x2 EXT=xqmac
  t "BMM QK^T, qop8"                    PASS TEST=bmm_x2 EXT=xqmac OPT="-O2 -DQOP8"
  t "FC QKV, vqdot4.vx"                 PASS TEST=fc_x3 EXT=xqmac VEC=1
  t "FC QKV, vqdot8.vx"                 PASS TEST=fc_x3 EXT=xqmac VEC=1 OPT="-O2 -DQDOT8"
  t "BMM QK^T, vqdot4.vx"               PASS TEST=bmm_x3 EXT=xqmac VEC=1
  t "BMM AV, vqdot8.vx"                 PASS TEST=bmm_x3 EXT=xqmac VEC=1 OPT="-O2 -DQDOT8 -DBMM_SEL=1"
  t "block 0, scalar"                   PASS TEST=block OPT="-O2 -DSCALAR"
  t "block 0, Xqmac"                    PASS TEST=block EXT=xqmac
  t "block 0, X2 qop8"                  PASS TEST=block_x2 EXT=xqmac OPT="-O2 -DQOP8"
  t "block 0, X3 vqdot8.vx"             PASS TEST=block_x3 EXT=xqmac VEC=1 OPT="-O2 -DQDOT8"
  out=$(make -s -B TEST=block_x2 OPT="-O2 -DQOP8" checkfp 2>&1)
  if grep -q "^OK" <<<"$out"; then
    printf '  [PASS] %s\n' "no floating-point instructions or soft-float calls (block_x2 binary)"
    pass=$((pass + 1))
  else
    printf '  [FAIL] no-FP check\n'; tail -6 <<<"$out" | sed 's/^/         /'
    fail=$((fail + 1))
  fi
fi

echo "$pass passed, $fail failed"
((fail == 0))
