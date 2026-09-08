#!/bin/bash
# Wrapper for the native-clean litmus-test eval script -- real file, not inline
# `bash -c "..."`, for the same nested-quoting reason as the other wrappers here.
set -euo pipefail
CKPT_DIR="${1:?Usage: $0 <ckpt_dir> <tag> [n_steps] [n_frames]}"
TAG="${2:?Usage: $0 <ckpt_dir> <tag> [n_steps] [n_frames]}"
N_STEPS="${3:-20}"
N_FRAMES="${4:-8}"

pip install --quiet datasets peft imagecorruptions webdataset s3fs seaborn plotly \
    "opencv-python-headless>=4.9" invisible_watermark
pip install --quiet "git+https://github.com/giannisdaras/ambient_utils.git"

python /myhome/sdate/scripts/tr_diffusion_ambient_tweedie_sdxl_nativeclean_eval.py \
  --ckpt_dir "${CKPT_DIR}" --tag "${TAG}" --n_steps "${N_STEPS}" --n_frames "${N_FRAMES}"
