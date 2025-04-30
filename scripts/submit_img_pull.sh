#!/bin/bash
#SBATCH --job-name=img-pull
#SBATCH -o /home/lucas.nunes/logs/%x-%j.out

# Launches dev environment
# Usage
# $ sbatch submit_jobs.sh <image-uri>

cd /mnt/cern_data/lucas.nunes/imgs
singularity pull --disable-cache $1