#!/bin/bash
#SBATCH --job-name=lzt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH -o /home/lucas.nunes/logs/lorenzetti/%x-%j.out
# %x is the job name
# %j is the job ID

# Launches dev environment
# Usage
# $ sbatch singularity_lzt.sh <command>

echo "Running command: ${@} on boosted-lorenzetti_latest.sif"
singularity exec \
--nv \
--bind /mnt/cern_data:/mnt/cern_data \
$SIF_IMGS_DIR/boosted-lorenzetti_latest.sif "${@}"