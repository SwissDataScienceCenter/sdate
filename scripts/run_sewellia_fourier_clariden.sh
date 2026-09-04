#!/bin/bash
# Runs sewellia_fourier_mle_slice.py for one or more z-slices in parallel on Clariden,
# one process per GPU (CUDA_VISIBLE_DEVICES pinned), inside a single container/allocation.
# GH200 nodes are exclusive-per-node regardless of --gpus requested, so fanning out
# multiple slices inside one node allocation (rather than one job per slice) avoids
# reserving N whole nodes to do N GPUs' worth of work.
#
# Usage: SLICES="72 217 362 507" STEPS=1200 bash scripts/run_sewellia_fourier_clariden.sh
set -e
cd /users/lbarba/sdate

SLICES=${SLICES:-"72 217 362 507"}
K=${K:-2}
STEPS=${STEPS:-1200}
LR=${LR:-0.001}
GRAD_CLIP=${GRAD_CLIP:-10000}
BATCH_SIZE=${BATCH_SIZE:-2000}
SEED=${SEED:-0}
TAG_SUFFIX=${TAG_SUFFIX:-clariden}

DATA_DIR=/capstor/store/cscs/swissai/aa006/data/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01
DATA_PATH=$DATA_DIR/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5
PHASE_TXT=$DATA_DIR/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt
CALIB_PATH=/iopsstor/scratch/cscs/lbarba/sewellia_lineolata/sewellia_real_calibration.npz
OUT_DIR=/iopsstor/scratch/cscs/lbarba/sewellia_lineolata/fourier_mle
mkdir -p "$OUT_DIR"

gpu=0
pids=()
for z in $SLICES; do
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/sewellia_fourier_mle_slice.py \
    --z_slice "$z" --k "$K" --steps "$STEPS" --lr "$LR" --grad_clip "$GRAD_CLIP" \
    --batch_size "$BATCH_SIZE" --seed "$SEED" \
    --data_path "$DATA_PATH" --phase_txt "$PHASE_TXT" --calib_path "$CALIB_PATH" \
    --out_dir "$OUT_DIR" --tag "z${z}_k${K}_${TAG_SUFFIX}" \
    > "$OUT_DIR/log_z${z}_${TAG_SUFFIX}.txt" 2>&1 &
  pids+=($!)
  gpu=$((gpu + 1))
done

echo "launched ${#pids[@]} process(es) on GPUs 0..$((gpu-1)), pids: ${pids[@]}"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "ALL DONE (fail=$fail)"
exit $fail
