#!/bin/bash
# Wrapper executed INSIDE the RunAI pod (via sdate_launcher.sh) -- real file, not
# inline `bash -c "..."`, for the same nested-quoting reason as the other wrappers
# in this project.
#
# This is the LITERAL-RECIPE litmus test: after the "treat our real noisy
# measurement as the ambient observation" approach regressed with more training
# (checkpoint-24000 was WORSE than checkpoint-4000/5000, both metrically and
# visually -- a growing honeycomb/tiling artifact), we're dropping that entirely
# and reproducing their recipe exactly as they run it on real (clean) LAION
# photos: --sdate_native_clean uses the NATIVE full-dose reference frame as the
# "clean" image (no synthetic noise injected by us at all), and timestep_nature=100
# is copied verbatim from their shipped configs/train_low_level_laion10k.yaml (not
# our own calibrated 102, which assumed a real-noisy-measurement setting that no
# longer applies). This deliberately is NOT the real problem we care about -- it
# only tests whether their pipeline can recover a clean image below an assumed
# noise floor AT ALL on our grayscale content.
set -euo pipefail
DATA=/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2

pip install --quiet datasets peft imagecorruptions webdataset s3fs seaborn plotly \
    "opencv-python-headless>=4.9" invisible_watermark
pip install --quiet "git+https://github.com/giannisdaras/ambient_utils.git"

python /myhome/sdate/sdate/tr_diffusion/ambient_tweedie_sdxl/train_lora_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
  --sdate_mov_path "${DATA}/212_Wunderkerze2.mov" \
  --sdate_memmap_path "${DATA}/frames_400k_500k.u16" \
  --sdate_frame_start 400201 --sdate_frame_end 449799 \
  --sdate_crop_h 128 --sdate_crop_w 512 \
  --sdate_native_clean \
  --sdate_latent_cache_path "${DATA}/sdxl_ambient_latents_nativeclean.f16" \
  --output_dir /myhome/data/sdate/shared/checkpoints/tr_diff_ambient_tweedie_sdxl_lora_nativeclean \
  --train_batch_size 16 --gradient_accumulation_steps 1 \
  --num_train_epochs 200 --checkpointing_steps 500 --min_validation_steps 500 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 \
  --mixed_precision fp16 --rank 4 \
  --noisy_ambient --timestep_nature 100 --x0_pred \
  --consistency_coeff 0.015 --num_consistency_steps 2 --max_steps_diff 50 \
  --run_consistency_everywhere \
  --seed 42 --dataloader_num_workers 4 --time_limit 2880 \
  --resume_from_checkpoint latest \
  --report_to tensorboard
