#!/bin/bash
# Launches the SDXL-LoRA ambient-Tweedie training run on RunAI. Algorithmic
# hyperparameters (LoRA rank, consistency_coeff/num_consistency_steps/max_steps_diff,
# x0_pred, run_consistency_everywhere, with_grad, mixed_precision, learning_rate,
# lr_scheduler) are copied VERBATIM from the upstream repo's shipped production config
# (configs/train_low_level_laion10k.yaml, in _sdxl_ambient_train_wrapper.sh) -- see this
# project's memory for why: the user wants to try their exact recipe, on our data,
# before deciding whether to deviate. Data/resource knobs (batch size, crop shape,
# frame range, dataloader cache) are adapted to our setting.
#
# The actual pip-install + python invocation lives in the SEPARATE file
# _sdxl_ambient_train_wrapper.sh, not inlined here as a `bash -c "..."` string --
# nested quoting through sdate_launcher.sh's `eval "$COMMAND"` has repeatedly mangled
# multi-line/quoted commands in this project (confirmed again on the first launch
# attempt: `bash: -c: option requires an argument`). A real script file sidesteps
# the whole class of bug.
#
# --timestep_nature is NOT hardcoded here -- pass it in via $1 (the value from
# scripts/tr_diffusion_ambient_tweedie_sdxl_calibrate.py's output), since it's
# something WE calibrate empirically, not a value we can copy from their config
# (their 100 assumes THEIR synthetic-noise-injection setup, meaningless for us).
set -euo pipefail
TIMESTEP_NATURE="${1:?Usage: $0 <timestep_nature> [job_name]}"
JOB_NAME="${2:-tr-diff-sdxl-ambient-n2c}"

runai training standard submit "$JOB_NAME" -p sdate-luisb -i lfbarba/sdsc_image:1.0.0 \
  --node-type A100 --gpu-devices-request 1 --large-shm \
  --cpu-core-request 8 --cpu-core-limit 8 --cpu-memory-request 64G --cpu-memory-limit 64G \
  --preemptibility preemptible \
  --command -- bash /myhome/sdate/scripts/sdate_launcher.sh \
    bash /myhome/sdate/scripts/_sdxl_ambient_train_wrapper.sh "${TIMESTEP_NATURE}"
