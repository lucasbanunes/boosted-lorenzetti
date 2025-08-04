#!/bin/bash
#SBATCH --job-name=mlp-run-training-lzt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH -o /home/lucas.nunes/logs/boosted-lorenzetti/%x-%j.out
# %x is the job name
# %j is the job ID

# Usage
# $ sbatch run_mlp_training.sh <img> <tracking_uri> <run-id-1., run-id-2, ...>

img=$1
tracking_uri=$2
run_ids=${@:3}
command="cd /home/${USER}/workspaces/lorenzetti/boosted-lorenzetti &&"
command="${command} conda run -n dev --live-stream"
command="${command} python cli.py mlp run-training"
command="${command} ${run_ids}"
command="${command} --tracking-uri ${tracking_uri}"
command="${command} --experiment-name boosted-lorenzetti"

echo "Running command ${command} on ${img}"
singularity exec \
    --nv \
    --bind /mnt/cern_data:/mnt/cern_data \
    $SIF_IMGS_DIR/$img /usr/bin/bash -c "${command}"
