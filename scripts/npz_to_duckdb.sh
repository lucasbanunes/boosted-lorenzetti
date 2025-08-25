#!/bin/bash
#SBATCH --job-name=npz-to-duckdb-lzt
#SBATCH --partition=cpu-large
#SBATCH --cpus-per-task=40
#SBATCH -o /home/lucas.nunes/logs/boosted-lorenzetti/%x-%j.out
# %x is the job name
# %j is the job ID

# Usage
# $ sbatch npz_to_duckdb.sh <img> <dataset_dir> <output_file>

img=$1
dataset_dir=$2
output_file=$3
command="cd /home/${USER}/workspaces/lorenzetti/boosted-lorenzetti &&"
command="${command} conda run -n dev --live-stream"
command="${command} python cli.py npz to-duckdb"
command="${command} --dataset-dir ${dataset_dir}"
command="${command} --output-file ${output_file}"

echo "Running command ${command} on ${img}"
singularity exec \
    --bind /mnt/cern_data:/mnt/cern_data \
    $SIF_IMGS_DIR/$img /usr/bin/bash -c "${command}"
