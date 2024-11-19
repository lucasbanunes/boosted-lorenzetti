#!/bin/bash
#SBATCH --job-name=img-pull
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH -o /home/lucas.nunes/logs/tcc/%x-%j.out

# Launches dev environment
# Usage
# $ sbatch submit_jobs.sh <image-uri>

cd /mnt/cern_data/lucas.nunes/imgs
singularity pull --disable-cache $1