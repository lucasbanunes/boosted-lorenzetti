#!/bin/bash
#SBATCH --job-name=mlp-lzt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH -o /home/lucas.nunes/logs/boosted-lorenzetti/%x-%j.out
# %x is the job name
# %j is the job ID

# Usage
# $ sbatch run_mlp_training.sh <img> <lzt_dataset_path> <run-id-1., run-id-2, ...>

img=$1
lzt_dataset_path=$2
run_ids=${@:3}

command="cd /home/${USER}/workspaces/lorenzetti/boosted-lorenzetti && conda run -n dev python cli.py mlp run-training ${run_ids} --tracking-uri file://${lzt_dataset_path}/mlruns --experiment-name boosted-lorenzetti"

echo "Running command ${command} on ${img}"
singularity exec \
    --nv \
    --bind /mnt/cern_data:/mnt/cern_data \
    $SIF_IMGS_DIR/$img /usr/bin/bash -c "${command}"
