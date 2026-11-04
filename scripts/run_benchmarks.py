#!/usr/bin/env python3
"""Full-workload benchmark: whole DeiT-Tiny W8A8 inference on Spike, every ISA variant.

usage (inside WSL, from the repo root):
  scripts/run_benchmarks.py                          synthetic input image (no dataset needed)
  scripts/run_benchmarks.py --input ~/aca/data/deit/eval/0000.npy      a real image
  scripts/run_benchmarks.py --variants x1,x2_qop8    a subset
  scripts/run_benchmarks.py --no-export              reuse build/model from the last export

Steps: export the model and the LiteRT golden tensors for the chosen image (export_model.py),
then build and run src/tests/model (or model_x2) once per variant. Each run checks the stem output,
all 12 block outputs and the 1000 logits against LiteRT and prints its retired-instruction count
(rdinstret around each operator; one-time table setup and the checks are excluded).

Writes results/benchmarks.csv (one row per variant), results/benchmark_stages.csv (instret per
operator stage) and the full Spike output in build/bench/<variant>.txt. With the synthetic input the
counts are compared with results/expected_instret.csv (tolerance --tol, default 0.5 %).
Exit status 0 only if every variant is bit-exact (and within tolerance where expected values exist).
"""
import argparse
import csv
import os
import re
import subprocess
import sys

VARIANTS = [  # id, program, Spike extension, compiler flags, description
    ("scalar", "model", "", "-O2 -DSCALAR", "RV64IMC only: scalar kernels, software requantization"),
    ("x1", "model", "xqmac", "-O2", "Xqmac: qmac8 + rq / rq.a (2x4 tiles)"),
    ("x1_tile4", "model", "xqmac", "-O2 -DTILE4", "Xqmac with 4x4 software tiles (option A)"),
    ("x2_qop4", "model_x2", "xqmac", "-O2", "X2 outer-product MAC, qop4 (4x4 accumulators)"),
    ("x2_qop8", "model_x2", "xqmac", "-O2 -DQOP8", "X2 outer-product MAC, qop8 (8x8 accumulators)"),
]
MODEL = "models/deit_tiny_w8a8.tflite"
EXPECTED = "results/expected_instret.csv"


def python_with_litert():
    for py in (os.environ.get("PY"), os.path.expanduser("~/aca/venv/bin/python"), sys.executable):
        if py and os.path.exists(py):
            return py
    return "python3"


def parse(out):
    m = re.search(r'^RESULT variant="([^"]*)" instret=(\d+) class=(\d+) golden_class=(\d+) exact=(\d)', out, re.M)
    if not m:
        return None
    macs = int(re.search(r"^total\s+\d+\s+\((\d+) MACs\)", out, re.M).group(1))
    stages = {}
    table = out.split("instr/MAC\n", 1)[1].split("\ntotal", 1)[0]
    for line in table.splitlines():
        name, rest = line[:17].strip(), line[17:].split()
        stages[name] = int(rest[0])
    return {"instret": int(m.group(2)), "class": int(m.group(3)), "golden": int(m.group(4)),
            "exact": m.group(5) == "1", "macs": macs, "stages": stages}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default="synthetic", help="'synthetic' (default) or a preprocessed .npy image")
    ap.add_argument("--variants", default=",".join(v[0] for v in VARIANTS))
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--tol", type=float, default=0.5, help="allowed instret deviation from expected, %%")
    ap.add_argument("--update-expected", action="store_true",
                    help="store this run's counts as the expected values for this input")
    args = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    want = args.variants.split(",")
    variants = [v for v in VARIANTS if v[0] in want]
    input_name = "synthetic" if args.input == "synthetic" else os.path.basename(args.input)

    if not args.no_export:
        print(f"== export model + LiteRT golden ({input_name})", flush=True)
        r = subprocess.run([python_with_litert(), "scripts/export_model.py", MODEL, "--input", args.input])
        if r.returncode:
            sys.exit("export failed")

    expected = {}
    if os.path.exists(EXPECTED):
        with open(EXPECTED) as fh:
            for row in csv.DictReader(fh):
                expected[(row["input"], row["variant"])] = int(row["instret"])

    os.makedirs("build/bench", exist_ok=True)
    results = {}
    for vid, prog, ext, opt, desc in variants:
        print(f"== {vid}: {desc}", flush=True)
        cmd = ["make", "-s", "-B", f"TEST={prog}", f"OPT={opt}", f"EXT={ext}", "run"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout + p.stderr
        with open(f"build/bench/{vid}.txt", "w") as fh:
            fh.write(" ".join(cmd) + "\n" + out)
        res = parse(out)
        if res is None:
            print(out[-2000:])
            print(f"   no RESULT line (exit {p.returncode}); full output in build/bench/{vid}.txt")
            results[vid] = None
            continue
        results[vid] = res
        print(f"   instret {res['instret']:,}  class {res['class']} (LiteRT {res['golden']})  "
              f"{'bit-exact' if res['exact'] else 'NOT bit-exact'}")

    base = results.get("scalar")
    rows, ok = [], True
    for vid, prog, ext, opt, desc in variants:
        res = results[vid]
        if res is None:
            ok = False
            continue
        exp = expected.get((input_name, vid))
        dev = None if exp is None else 100.0 * (res["instret"] - exp) / exp
        within = dev is None or abs(dev) <= args.tol
        ok = ok and res["exact"] and within
        rows.append({
            "variant": vid, "description": desc, "program": prog, "spike_ext": ext, "cflags": opt,
            "input": input_name, "instret": res["instret"],
            "speedup_vs_scalar": f"{base['instret'] / res['instret']:.2f}" if base else "",
            "instr_per_mac": f"{res['instret'] / res['macs']:.3f}",
            "predicted_class": res["class"], "litert_class": res["golden"],
            "bit_exact": int(res["exact"]),
            "expected_instret": "" if exp is None else exp,
            "deviation_pct": "" if dev is None else f"{dev:.3f}"})

    os.makedirs("results", exist_ok=True)
    if rows:
        with open("results/benchmarks.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        done = [v for v in variants if results[v[0]]]
        names = list(results[done[0][0]]["stages"])
        with open("results/benchmark_stages.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stage"] + [v[0] for v in done])
            for n in names:
                w.writerow([n] + [results[v[0]]["stages"].get(n, "") for v in done])

    if args.update_expected:
        keep = {k: v for k, v in expected.items() if k[0] != input_name}
        for r in rows:
            keep[(input_name, r["variant"])] = r["instret"]
        with open(EXPECTED, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["input", "variant", "instret"])
            for (i, v), n in sorted(keep.items()):
                w.writerow([i, v, n])
        print(f"expected values for '{input_name}' written to {EXPECTED}")

    print(f"\nwhole-model inference, input {input_name}")
    print(f"{'variant':<10} {'instret':>15} {'speedup':>8} {'instr/MAC':>10} {'class':>6} {'exact':>6} {'vs expected':>12}")
    for r in rows:
        dev = f"{float(r['deviation_pct']):+.3f}%" if r["deviation_pct"] != "" else "-"
        print(f"{r['variant']:<10} {r['instret']:>15,} {r['speedup_vs_scalar'] + 'x':>8} {r['instr_per_mac']:>10} "
              f"{r['predicted_class']:>6} {'yes' if r['bit_exact'] else 'NO':>6} {dev:>12}")
    print("results/benchmarks.csv, results/benchmark_stages.csv written")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
