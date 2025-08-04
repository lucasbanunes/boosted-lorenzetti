#!/bin/bash
#SBATCH --job-name=kmlp-lzt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH -o /home/lucas.nunes/logs/boosted-lorenzetti/%x-%j.out
# %x is the job name
# %j is the job ID

# Usage
# $ sbatch mlp_run_kfold.sh <kfold-run-id> <tracking-uri> <img>
# $ sbatch mlp_run_kfold.sh 4978c033d2864d1e8bafb1d9bea3841f /home/test.user/mlruns boosted-lorenzetti_0.1.0.sif

kfold_run_id=$1
tracking_uri=$2
img=$3
command="cd /home/${USER}/workspaces/lorenzetti/boosted-lorenzetti &&"
command="${command} conda run -n dev --live-stream"
command="${command} python cli.py mlp run-kfold"
command="${command} ${kfold_run_id}"
command="${command} --tracking-uri file://${tracking_uri}"
command="${command} --experiment-name boosted-lorenzetti"
# "cd /mnt/cern_data/${USER}/lorenzetti && conda run -n dev mlflow ui -h 0.0.0.0 -p ${port}"

echo "Running command ${command} on ${img}"
singularity exec \
    --nv \
    --bind /mnt/cern_data:/mnt/cern_data \
    $SIF_IMGS_DIR/$img /usr/bin/bash -c "${command}"
