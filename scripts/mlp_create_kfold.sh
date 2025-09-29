#!/bin/bash
#SBATCH --job-name=mlp-create-kfold-lzt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH -o /home/lucas.nunes/logs/boosted-lorenzetti/%x-%j.out
# %x is the job name
# %j is the job ID

# Usage
# $ sbatch mlp_create_kfold.sh <img> <db-path> <tracking-uri> <checkpoints-dir>
# $ sbatch mlp_create_kfold.sh boosted-lorenzetti_latest.sif /home/test.user/lorenzetti.db /home/test.user/mlruns checkpoints

img=$1
db_path=$2
tracking_uri=$3
checkpoints_dir="${4}/checkpoints/mlp_kfold_$(date -d "today" +"%Y_%m_%d_%H_%M_%s")"
command="cd /home/${USER}/workspaces/lorenzetti/boosted-lorenzetti &&"
command="${command} conda run -n dev --live-stream"
command="${command} python cli.py mlp create-kfold 100 2 1"
command="${command} --db-path ${db_path}"
command="${command} --table-name data"
command="${command} --best-metric val_max_sp"
command="${command} --best-metric-mode max"
command="${command} --rings-col cl_rings"
command="${command} --label-col label"
command="${command} --fold-col fold"
command="${command} --activation relu"
command="${command} --batch-size 32"
command="${command} --patience 5"
command="${command} --fold 5"
command="${command} --inits 5"
command="${command} --max-epochs 1000"
command="${command} --name mlp-kfold-multi-init-zee-jf17"
command="${command} --tracking-uri file://${tracking_uri}"
command="${command} --experiment-name boosted-lorenzetti"
# "cd /mnt/cern_data/${USER}/lorenzetti && conda run -n dev mlflow ui -h 0.0.0.0 -p ${port}"

echo "Running command ${command} on ${img}"
mkdir -p $checkpoints_dir && \
singularity exec \
    --nv \
    --bind /mnt/cern_data:/mnt/cern_data \
    $SIF_IMGS_DIR/$img /usr/bin/bash -c "${command}"
