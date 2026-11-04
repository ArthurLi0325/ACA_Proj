#!/usr/bin/env bash
# Reproduce the quantized DeiT-Tiny models (run inside WSL from the repo root).
#   scripts/get_model.sh            download, preprocess, quantize (w8a8 + w8a16)
#   scripts/get_model.sh eval       also evaluate top-1 on the eval set (~10 min)
set -euo pipefail

PY=${PY:-$HOME/aca/venv/bin/python}
DATA=${DATA:-$HOME/aca/data}
MODELS=models
mkdir -p "$DATA" "$MODELS"

# 1. Float DeiT-Tiny from litert-community (timm deit_tiny_patch16_224.fb_in1k)
if [[ ! -f $MODELS/deit_tiny_f32.tflite ]]; then
  "$PY" - <<EOF
import shutil
from huggingface_hub import hf_hub_download
shutil.copy(hf_hub_download("litert-community/deit_tiny_patch16_224", "model.tflite"),
            "$MODELS/deit_tiny_f32.tflite")
EOF
fi

# 2. Calibration / evaluation images: one mini-imagenet validation shard (358 MB)
PARQUET=$DATA/mini-imagenet/data/validation-00000-of-00003.parquet
if [[ ! -f $PARQUET ]]; then
  "$PY" - <<EOF
from huggingface_hub import hf_hub_download
hf_hub_download("timm/mini-imagenet", "data/validation-00000-of-00003.parquet",
                repo_type="dataset", local_dir="$DATA/mini-imagenet")
EOF
fi
[[ -f $DATA/imagenet_synsets.txt ]] || curl -sSfL -o "$DATA/imagenet_synsets.txt" \
  https://raw.githubusercontent.com/huggingface/pytorch-image-models/v1.0.15/timm/data/_info/imagenet_synsets.txt

# 3. Preprocess: 128 calibration + 500 eval images
[[ -d $DATA/deit/eval ]] || "$PY" scripts/prepare_data.py "$PARQUET" "$DATA/imagenet_synsets.txt" \
  "$DATA/deit" --calib 128 --eval 500

# 4. Static post-training quantization
for r in w8a8 w8a16; do
  "$PY" scripts/quantize_model.py "$MODELS/deit_tiny_f32.tflite" "$MODELS/deit_tiny_$r.tflite" \
    --recipe "$r" --calib-dir "$DATA/deit/calib" --num-calib 128
done

# 5. Optional accuracy check
if [[ ${1:-} == eval ]]; then
  "$PY" scripts/eval_model.py "$DATA/deit/eval" "$MODELS/deit_tiny_f32.tflite" \
    "$MODELS/deit_tiny_w8a8.tflite" "$MODELS/deit_tiny_w8a16.tflite"
fi
