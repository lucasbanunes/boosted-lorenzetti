#!/bin/bash
#SBATCH --job-name=mlflow-lzt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH -o /home/lucas.nunes/logs/boosted-lorenzetti/%x-%j.out
# %x is the job name
# %j is the job ID

# Usage
# $ sbatch run_mlflow_ui.sh <img> <mlruns-path> <port>
# $ sbatch run_mlflow_ui.sh boosted-lorenzetti_0.1.0.sif /home/test.user/mlruns 5000

img=$1
mlruns_path=$2
port=$3
command="conda run -n dev mlflow ui -h 0.0.0.0 -p ${port} --backend-store-uri file://${mlruns_path}"
# "cd /mnt/cern_data/${USER}/lorenzetti && conda run -n dev mlflow ui -h 0.0.0.0 -p ${port}"

echo "Running command ${command} on ${img}"
singularity exec \
    --bind /mnt/cern_data:/mnt/cern_data \
    $SIF_IMGS_DIR/$img /usr/bin/bash -c "${command}"
