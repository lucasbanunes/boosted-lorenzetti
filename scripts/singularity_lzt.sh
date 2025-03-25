#!/bin/bash
#SBATCH --job-name=lzt
#SBATCH --partition=cpu-large
#SBATCH --cpus-per-task=40
#SBATCH -o /home/lucas.nunes/logs/lorenzetti/%x-%j.out

# Launches dev environment
# Usage
# $ sbatch submit_lorenzetti.sh <command>

singularity exec \
--nv \
--bind /mnt/cern_data:/mnt/cern_data \
$SIF_IMGS_DIR/lorenzetti_latest.sif "${@}"