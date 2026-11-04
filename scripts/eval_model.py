#!/usr/bin/env python3
"""Top-1 accuracy of float and quantized models, plus float/int8 prediction agreement.

usage: eval_model.py <eval_dir> <model.tflite> [<model2.tflite> ...] [--limit N]
eval_dir contains *.npy inputs (NCHW float32) and labels.csv from prepare_data.py.
Integer-input models get their input quantized with the model's input scale/zero-point.
"""
import argparse
import csv

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def make_runner(path):
    it = Interpreter(model_path=path,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]

    def run(x):
        if inp["dtype"] != np.float32:
            s, zp = inp["quantization"]
            info = np.iinfo(inp["dtype"])
            x = np.clip(np.round(x / s) + zp, info.min, info.max).astype(inp["dtype"])
        it.set_tensor(inp["index"], x)
        it.invoke()
        return it.get_tensor(out["index"])[0].astype(np.float32)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(f"{args.eval_dir}/labels.csv")))
    if args.limit:
        rows = rows[: args.limit]
    runners = [make_runner(m) for m in args.models]
    preds = [[] for _ in args.models]
    for r in rows:
        x = np.load(f"{args.eval_dir}/{r['file']}")
        for k, run in enumerate(runners):
            preds[k].append(int(np.argmax(run(x))))
    labels = np.array([int(r["imagenet_index"]) for r in rows])

    print(f"{len(rows)} images")
    for m, p in zip(args.models, preds):
        acc = np.mean(np.array(p) == labels)
        print(f"  top-1 {100 * acc:6.2f}%  {m}")
    for k in range(1, len(args.models)):
        agree = np.mean(np.array(preds[0]) == np.array(preds[k]))
        print(f"  agreement with {args.models[0]}: {100 * agree:6.2f}%  ({args.models[k]})")


if __name__ == "__main__":
    main()
