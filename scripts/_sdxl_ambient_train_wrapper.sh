#!/bin/bash
# Wrapper executed INSIDE the RunAI pod (via sdate_launcher.sh) -- kept as a real file
# rather than an inline `bash -c "..."` string, since nested quoting through
# sdate_launcher.sh's `eval "$COMMAND"` has repeatedly mangled multi-line/quoted
# commands in this project. Takes the calibrated timestep_nature as $1.
set -euo pipefail
TIMESTEP_NATURE="${1:?Usage: $0 <timestep_nature>}"
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
  --sdate_extra_noise_dose 0.05 --sdate_noise_seed 999 \
  --sdate_latent_cache_path "${DATA}/sdxl_ambient_latents_dose05.f16" \
  --output_dir /myhome/data/sdate/shared/checkpoints/tr_diff_ambient_tweedie_sdxl_lora \
  --train_batch_size 8 --gradient_accumulation_steps 1 \
  --num_train_epochs 200 --checkpointing_steps 500 --min_validation_steps 500 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 \
  --mixed_precision fp16 --rank 4 \
  --noisy_ambient --timestep_nature "${TIMESTEP_NATURE}" --x0_pred \
  --consistency_coeff 0.015 --num_consistency_steps 2 --max_steps_diff 50 \
  --run_consistency_everywhere \
  --seed 42 --dataloader_num_workers 4 --time_limit 2880 \
  --resume_from_checkpoint latest \
  --report_to tensorboard
