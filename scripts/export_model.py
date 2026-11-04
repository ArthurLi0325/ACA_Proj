#!/usr/bin/env python3
"""Export the whole DeiT-Tiny W8A8 model to C headers plus golden tensors, for src/tests/model.

usage: export_model.py model.tflite [--input x.npy|synthetic] [--out build/model] [--golden build/golden]

Runs scripts/export_block.py on every part of the graph with the same input image:
  ops   0..9    stem: patch-embedding convolution, class token, position embedding -> stem_*.h
  ops  10..549  12 encoder blocks of 45 ops                                       -> block<b>_*.h
  ops 550..560  final LayerNorm, class-token slice, classifier head              -> fin_*.h
and checks every operator's integer formulation against LiteRT (the same checks export_block.py
prints, summarized here). The C program checks the stem output, every block output and the 1000
logits against these goldens and reports the predicted class.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_block  # noqa: E402

N_BLOCKS = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--input", help="preprocessed NCHW float32 image (.npy), or 'synthetic' (default: "
                    "~/aca/data/deit/eval/0000.npy if present, else synthetic)")
    ap.add_argument("--out", default="build/model")
    ap.add_argument("--golden", default="build/golden")
    args = ap.parse_args()

    common = [args.model, "--out", args.out, "--golden", args.golden, "--quiet"]
    if args.input:
        common += ["--input", args.input]
    parts = [("stem", ["--range", "0:9", "--prefix", "STEM", "--base", "0"])]
    parts += [(f"block {b}", ["--block", str(b)]) for b in range(N_BLOCKS)]
    parts += [("tail", ["--range", "550:560", "--prefix", "FIN", "--base", "10"])]

    n_ops, worst = 0, []
    for name, extra in parts:
        summary = export_block.main(common + extra)
        n_ops += len(summary)
        bad = [(k, label, pct) for k, label, pct in summary if pct < 100]
        worst += bad
        print(f"{name:<9} {len(summary):3d} operators checked" +
              ("" if not bad else f", {len(bad)} NOT bit-exact: " +
               ", ".join(f"op {k} {label} {pct:.3f}%" for k, label, pct in bad)))

    logits = np.load(os.path.join(args.golden, "fin", f"t{_logits_index(args.golden):04d}.npy")).reshape(-1)
    top = np.argsort(-logits.astype(np.int64), kind="stable")[:5]
    print(f"\n{n_ops} operators, {'all bit-exact' if not worst else f'{len(worst)} not bit-exact'} "
          f"against LiteRT")
    print("LiteRT top-5 classes (int8 logits): " + ", ".join(f"{c} ({logits[c]})" for c in top))
    with open(os.path.join(args.out, "stem_golden.h")) as fh:
        print(fh.readline().split("input image: ", 1)[-1].strip(), "->", f"headers in {args.out}/")
    return 1 if worst else 0


def _logits_index(golden):
    import json
    with open(os.path.join(golden, "fin", "manifest.json")) as fh:
        return int(json.load(fh)["block_output"])


if __name__ == "__main__":
    sys.exit(main())
