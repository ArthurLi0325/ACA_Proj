#!/usr/bin/env python3
"""Summarize a LiteRT/TFLite model: op histogram, per-op MACs, tensor dtypes.

usage: model_info.py model.tflite [--ops] [--csv out.csv]
  --ops   print every op with input/output shapes and dtypes
"""
import argparse
import collections

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def op_macs(name, ins, outs):
    """MAC count for compute-heavy ops; 0 for everything else."""
    if not outs:
        return 0
    out = outs[0]["shape"]
    if name == "FULLY_CONNECTED":
        w = ins[1]["shape"]                      # [out_features, in_features]
        return int(np.prod(out)) * int(w[-1])
    if name == "BATCH_MATMUL":
        a = ins[0]["shape"]
        return int(np.prod(out)) * int(a[-1])    # K = last dim of lhs (adj_x not handled)
    if name == "CONV_2D":
        w = ins[1]["shape"]                      # [O, kh, kw, I]
        return int(np.prod(out)) * int(np.prod(w[1:]))
    if name == "DEPTHWISE_CONV_2D":
        w = ins[1]["shape"]                      # [1, kh, kw, O]
        return int(np.prod(out)) * int(w[1] * w[2])
    return 0


def fmt(t):
    return f"{[int(d) for d in t['shape']]}:{np.dtype(t['dtype']).name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ops", action="store_true")
    ap.add_argument("--csv")
    args = ap.parse_args()

    # Disable XNNPACK so ops are not fused into DELEGATE nodes.
    it = Interpreter(model_path=args.model,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    tensors = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()

    count = collections.Counter()
    macs = collections.Counter()
    dtypes = collections.defaultdict(set)
    rows = []
    for op in ops:
        name = op["op_name"]
        ins = [tensors[i] for i in op["inputs"] if i >= 0]
        outs = [tensors[i] for i in op["outputs"] if i >= 0]
        m = op_macs(name, ins, outs)
        count[name] += 1
        macs[name] += m
        for t in ins + outs:
            dtypes[name].add(np.dtype(t["dtype"]).name)
        rows.append((op["index"], name, m,
                     " ".join(fmt(t) for t in ins),
                     " ".join(fmt(t) for t in outs)))
        if args.ops:
            print(f"{op['index']:4d} {name:<22} MAC={m:>12,}  in={rows[-1][3]}  out={rows[-1][4]}")

    total = sum(macs.values())
    params = sum(int(np.prod(t["shape"])) for t in tensors.values()
                 if t["name"] and np.dtype(t["dtype"]).name in ("int8", "float32") and
                 any(t["index"] == op["inputs"][1] for op in ops if len(op["inputs"]) > 1
                     and op["op_name"] in ("FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D")))
    print(f"\nModel: {args.model}")
    print(f"Ops: {len(ops)}   Total MACs: {total:,} ({total / 1e9:.3f} G)   Weight params (FC/conv): {params:,}")
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    print(f"Input : {fmt(inp)} quant={inp['quantization']}")
    print(f"Output: {fmt(outp)} quant={outp['quantization']}")
    print(f"\n{'op':<24}{'count':>6}{'MACs':>16}{'%MAC':>8}  dtypes")
    for name, n in sorted(count.items(), key=lambda kv: -macs[kv[0]]):
        pct = 100 * macs[name] / total if total else 0
        print(f"{name:<24}{n:>6}{macs[name]:>16,}{pct:>7.2f}%  {','.join(sorted(dtypes[name]))}")

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("index,op,macs,inputs,outputs\n")
            for r in rows:
                fh.write(f"{r[0]},{r[1]},{r[2]},\"{r[3]}\",\"{r[4]}\"\n")


if __name__ == "__main__":
    main()
