#!/usr/bin/env python3
"""Static INT8 post-training quantization of a float LiteRT model (ai-edge-quantizer).

usage: quantize_model.py float.tflite out.tflite [--recipe w8a8|w8a16]
                         [--calib-dir DIR] [--num-calib N]
  --calib-dir  directory of preprocessed calibration inputs (*.npy, NCHW float32);
               if omitted, random normal inputs are used (feasibility only).
"""
import argparse
import collections
import glob
import os

import numpy as np
from ai_edge_quantizer import quantizer, recipe
from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.tools import flatbuffer_utils

RECIPES = {"w8a8": recipe.static_wi8_ai8, "w8a16": recipe.static_wi8_ai16}


def make_tensor_names_unique(src, dst):
    """The quantizer requires unique tensor names; converted models may repeat them."""
    model = flatbuffer_utils.read_model(src)
    renamed = 0
    for sg in model.subgraphs:
        seen = collections.Counter()
        for t in sg.tensors:
            name = t.name.decode() if isinstance(t.name, bytes) else t.name
            if seen[name]:
                t.name = f"{name}__{seen[name]}".encode()
                renamed += 1
            seen[name] += 1
    flatbuffer_utils.write_model(model, dst)
    return renamed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("float_model")
    ap.add_argument("out")
    ap.add_argument("--recipe", default="w8a8", choices=RECIPES)
    ap.add_argument("--calib-dir")
    ap.add_argument("--num-calib", type=int, default=8)
    args = ap.parse_args()

    sig = Interpreter(model_path=args.float_model).get_signature_list()
    sig_key = next(iter(sig))
    in_name = sig[sig_key]["inputs"][0]

    if args.calib_dir:
        files = sorted(glob.glob(f"{args.calib_dir}/*.npy"))[: args.num_calib]
        samples = [np.load(f).astype(np.float32) for f in files]
    else:
        rng = np.random.default_rng(0)
        samples = [rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
                   for _ in range(args.num_calib)]
    print(f"signature={sig_key} input={in_name} calibration samples={len(samples)}")

    uniq = os.path.splitext(args.out)[0] + "_uniq_f32.tflite"
    print(f"renamed {make_tensor_names_unique(args.float_model, uniq)} duplicate tensor names")

    qt = quantizer.Quantizer(uniq)
    qt.load_quantization_recipe(RECIPES[args.recipe]())
    calib = qt.calibrate({sig_key: [{in_name: s} for s in samples]})
    result = qt.quantize(calib)
    result.export_model(args.out, overwrite=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
