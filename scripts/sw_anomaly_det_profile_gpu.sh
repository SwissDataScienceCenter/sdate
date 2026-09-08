#!/bin/bash
# Sample nvidia-smi GPU utilization concurrently with a short sw_anomaly_det
# run, to answer "how well is the GPU actually being used" with real numbers
# instead of guessing from wall-clock alone. Single self-contained script (no
# nested bash -c quoting through runai/sdate_launcher.sh -- that broke before).
set -e
SAMPLE_LOG=/tmp/sw_anomaly_det_nvidia_smi_samples.csv
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l 1 > "$SAMPLE_LOG" &
SMI_PID=$!

python scripts/sw_anomaly_det_run.py --profile wunderkerze2 \
  --frame_start 412000 --frame_end 468000 --T 21 11 5 \
  --out_dir /myhome/data/sdate/shared/time_resolved/sw_anomaly_det --tag gpu_profile \
  --max_windows 30 --log_every 1 --plot_every 0

kill "$SMI_PID" 2>/dev/null || true
sleep 1
echo "--- nvidia-smi samples (1 Hz) ---"
cat "$SAMPLE_LOG"
