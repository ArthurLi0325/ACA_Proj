#!/usr/bin/env bash
# Reproducible environment setup (run inside WSL Ubuntu 24.04).
#   scripts/setup_env.sh apt      # system packages (needs sudo password)
#   scripts/setup_env.sh spike    # Spike at a pinned commit + the project patches -> ~/aca/install
#   scripts/setup_env.sh python   # Python venv -> ~/aca/venv
#   scripts/setup_env.sh check    # verify everything
set -euo pipefail

ACA=${ACA:-$HOME/aca}
REPO=$(cd "$(dirname "$0")/.." && pwd)
SPIKE_SRC=$ACA/riscv-isa-sim
VENV=${VENV:-$ACA/venv}   # on the Linux filesystem: pip on /mnt/c is very slow
SPIKE_UPSTREAM=${SPIKE_UPSTREAM:-https://github.com/riscv-software-src/riscv-isa-sim.git}
# Upstream commit the patches were made against (they apply cleanly here; newer commits may need
# small fixes). patches/spike-xqmac.patch: Xqmac (custom-0); patches/spike-xqop.patch: X2 (custom-1).
SPIKE_COMMIT=${SPIKE_COMMIT:-5a760667bf0d0ab893173b9eeff9c109c8d7dd9a}

do_apt() {
  sudo apt-get update
  sudo apt-get install -y \
    gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf \
    build-essential device-tree-compiler libboost-regex-dev libboost-system-dev \
    python3-venv python3-pip git make
}

do_spike() {
  mkdir -p "$ACA"
  if [[ ! -d $SPIKE_SRC ]]; then
    # Reuse a local checkout when available, otherwise clone upstream.
    if [[ -d $HOME/riscv-isa-sim/.git ]]; then git clone "$HOME/riscv-isa-sim" "$SPIKE_SRC"
    else git clone "$SPIKE_UPSTREAM" "$SPIKE_SRC"; fi
    git -C "$SPIKE_SRC" checkout -b xqmac "$SPIKE_COMMIT"
    git -C "$SPIKE_SRC" apply "$REPO/patches/spike-xqmac.patch"
    git -C "$SPIKE_SRC" apply "$REPO/patches/spike-xqop.patch"
  elif ! grep -q qop8 "$SPIKE_SRC/riscv/xqmac.cc" 2>/dev/null; then
    echo "note: $SPIKE_SRC exists but has no X2 instructions; apply patches/spike-*.patch there" >&2
  fi
  mkdir -p "$SPIKE_SRC/build"
  cd "$SPIKE_SRC/build"
  [[ -f Makefile ]] || ../configure --prefix="$ACA/install" --enable-histogram
  make -j"$(nproc)"
  make install
}

do_python() {
  cd "$REPO"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r requirements.txt
}

do_check() {
  local ok=1
  check() { if eval "$2" >/dev/null 2>&1; then echo "  [ok]   $1"; else echo "  [MISS] $1"; ok=0; fi; }
  echo "Environment check"
  check "riscv64-unknown-elf-gcc"        "riscv64-unknown-elf-gcc --version"
  check "project Spike ($ACA/install)"   "test -x $ACA/install/bin/spike"
  check "Python venv ($VENV)"            "test -x $VENV/bin/python"
  check "numpy / ai-edge-litert"         "$VENV/bin/python -c 'import numpy, ai_edge_litert'"
  cd "$REPO"
  check "bare-metal hello on Spike"      "make -s TEST=hello run"
  check "no-FP check"                    "make -s TEST=hello checkfp"
  [[ $ok == 1 ]] && echo "All good." || echo "Some items missing (see above)."
}

case "${1:-check}" in
  apt) do_apt ;; spike) do_spike ;; python) do_python ;; check) do_check ;;
  all) do_apt; do_spike; do_python; do_check ;;
  *) echo "usage: $0 {apt|spike|python|check|all}"; exit 1 ;;
esac
