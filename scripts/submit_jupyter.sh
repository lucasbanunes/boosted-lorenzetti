#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH -o /home/lucas.nunes/logs/lorenzetti/%x-%j.out

# Usage
# $ sbatch run_mlflow_ui.sh <img> <mlruns-path> <port>
# $ sbatch run_mlflow_ui.sh boosted-lorenzetti_latest.sif /home/test.user/mlruns 5000

img=$1
port=$2

singularity exec \
--nv \
--bind /mnt/cern_data:/mnt/cern_data \
$SIF_IMGS_DIR/lorenzetti_latest.sif "bash run_jupyter.sh $img $port"