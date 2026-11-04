#!/usr/bin/env bash
# Fail if an ELF contains floating-point instructions or soft-float library calls.
set -euo pipefail
elf=${1:?usage: check_no_fp.sh <prog.elf>}
PREFIX=${PREFIX:-riscv64-unknown-elf-}

fp_insns=$(${PREFIX}objdump -d -M no-aliases "$elf" \
  | grep -P '^\s*[0-9a-f]+:\s+[0-9a-f]+\s+(c\.)?f(?!ence)[a-z.]+\s' || true)
soft_calls=$(${PREFIX}nm "$elf" \
  | grep -E '__(add|sub|mul|div|neg|cmp|eq|ne|lt|le|gt|ge|unord|fix|float|extend|trunc)[sdt]f[0-9]?' || true)

status=0
if [[ -n "$fp_insns" ]]; then
  echo "FAIL: floating-point instructions found:"; echo "$fp_insns" | head -20; status=1
fi
if [[ -n "$soft_calls" ]]; then
  echo "FAIL: soft-float library routines linked:"; echo "$soft_calls"; status=1
fi
[[ $status -eq 0 ]] && echo "OK: no floating-point instructions or soft-float calls in $elf"
exit $status
