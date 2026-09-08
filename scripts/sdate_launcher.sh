#!/bin/bash --noprofile --norc

# chip_launcher.sh - Setup and run commands for the chip project
#
# This script automatically sets up the chip project environment and then runs a specified command.
# It performs the following setup steps:
#   1. Changes to /myhome/chip-project directory
#   2. Runs scripts/setup.sh
#   3. Installs odl package via pip
#
# Usage examples:
#   # Direct command (recommended for simplicity):
#   ./chip_launcher.sh python chip/training/train_crippled_unet.py --epochs=10 --batch_size=16
#
#   # With --command flag:
#   ./chip_launcher.sh --command "python chip/inference/compute_reconstruction.py --im_size=512"
#
#   # Complex command with many arguments:
#   ./chip_launcher.sh python chip/inference/compute_reconstruction.py --im_size=512 --exp_name=LungSparseEvaluation --rescale=512 --testset_size=50 --batch_size=1 --angle_sampling=uniform --dataset_type=h5 --dataset_path=/mydata/chip/shared/data/lung/ground_truth_train --wandb --sparsity=130 --reconstruction_type=unet --load_checkpoint=checkpoints/Unet_Lung_512_C1.pt --compression=1 --finetuning_type=model --finetuning_steps=[5000] --finetune_learning_rate=[1e-4]

# Function to show usage
usage() {
    echo "Usage: $0 [--command] \"<command_to_run>\""
    echo "Examples:"
    echo "  $0 --command \"python /myhome/chip-project/chip/inference/compute_reconstruction.py --im_size=512\""
    echo "  $0 python /myhome/chip-project/chip/training/train_crippled_unet.py --epochs=10"
    exit 1
}

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    echo "Error: No command provided"
    usage
fi

# If first argument is --command, parse it properly
if [[ "$1" == "--command" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "Error: --command requires an argument"
        usage
    fi
    COMMAND="$2"
elif [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
else
    # Treat all arguments as the command to run
    COMMAND="$*"
fi

# Setup commands
echo "Setting up sdate project..."
cd /myhome/sdate

echo "Running setup script..."
# Sourced, not `sh scripts/setup.sh` -- that ran it as a subprocess, so every
# `export` inside setup.sh (PATH, HOME, ffmpeg's PATH fix-up, etc.) died with
# that subshell and never reached the actual command below. This had been
# silently harmless (the image's own default PATH/env already covered
# everything setup.sh used to export) until the ffmpeg persistent-cache PATH
# addition needed it to actually take effect.
. scripts/setup.sh

echo "Installing odl..."
pip install --quiet odl

echo "Setup complete. Running command..."
echo "Command: $COMMAND"

# Execute the provided command
eval "$COMMAND"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Command completed successfully"
else
    echo "Command failed with exit code $?"
    exit 1
fi
