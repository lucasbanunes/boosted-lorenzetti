#!/bin/bash
#SBATCH --job-name=lzt
#SBATCH --partition=cpu-large
#SBATCH --cpus-per-task=40
#SBATCH -o /home/lucas.nunes/logs/lorenzetti/%x-%j.out
# %x is the job name
# %j is the job ID

# Launches dev environment
# Usage
# $ sbatch submit_lorenzetti.sh <command>

echo "Running command: ${@} on lorenzetti_latest.sif"
singularity exec \
--nv \
--bind /mnt/cern_data:/mnt/cern_data \
$SIF_IMGS_DIR/lorenzetti_v2.1.0.sif "${@}"