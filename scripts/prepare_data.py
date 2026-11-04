#!/usr/bin/env python3
"""Preprocess mini-imagenet validation images for DeiT (timm deit_tiny_patch16_224.fb_in1k).

Preprocessing: resize shorter side to 256 (bicubic, crop_pct 0.875), center crop 224,
scale to [0,1], normalize with ImageNet mean/std, layout NCHW float32.

usage: prepare_data.py <parquet> <synsets.txt> <out_dir> --calib 128 --eval 500
Writes out_dir/calib/*.npy, out_dir/eval/*.npy and out_dir/eval/labels.csv
(labels are ImageNet-1k class indices = position in the sorted synset list).
"""
import argparse
import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img, size=224, crop_pct=0.875):
    img = img.convert("RGB")
    short = int(size / crop_pct)
    w, h = img.size
    scale = short / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)        # 1x3x224x224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("synsets")
    ap.add_argument("out_dir")
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--eval", type=int, default=500)
    args = ap.parse_args()

    synsets = [s.strip() for s in open(args.synsets) if s.strip()]
    wnid_to_idx = {w: i for i, w in enumerate(synsets)}

    table = pq.read_table(args.parquet)
    names = table.schema.field("label").metadata  # class names live in the HF features metadata
    meta = pq.read_metadata(args.parquet).metadata or {}
    import json
    features = json.loads(meta.get(b"huggingface", b"{}")).get("info", {}).get("features", {})
    label_names = features.get("label", {}).get("names")
    if label_names is None:
        raise SystemExit("could not find label names in parquet metadata")

    # Shuffle deterministically so calib/eval sets cover many classes.
    order = np.random.default_rng(0).permutation(table.num_rows)
    need = args.calib + args.eval
    images = table.column("image")
    labels = table.column("label")

    os.makedirs(f"{args.out_dir}/calib", exist_ok=True)
    os.makedirs(f"{args.out_dir}/eval", exist_ok=True)
    with open(f"{args.out_dir}/eval/labels.csv", "w") as fh:
        fh.write("file,imagenet_index,wnid\n")
        for k, row in enumerate(order[:need]):
            img = Image.open(io.BytesIO(images[int(row)].as_py()["bytes"]))
            x = preprocess(img)
            wnid = label_names[labels[int(row)].as_py()]
            if k < args.calib:
                np.save(f"{args.out_dir}/calib/{k:04d}.npy", x)
            else:
                name = f"{k - args.calib:04d}.npy"
                np.save(f"{args.out_dir}/eval/{name}", x)
                fh.write(f"{name},{wnid_to_idx[wnid]},{wnid}\n")
    print(f"wrote {args.calib} calibration and {args.eval} eval samples to {args.out_dir}")


if __name__ == "__main__":
    main()
