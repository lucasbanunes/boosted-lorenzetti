#!/bin/bash
#SBATCH --job-name=aod-create-ringer-dataset
#SBATCH -o /home/lucas.nunes/logs/boosted-lorenzetti/%x-%j.out
# %x is the job name
# %j is the job ID

img="boosted-lorenzetti_latest.sif"
command="cd /home/${USER}/workspaces/lorenzetti/boosted-lorenzetti &&"
command="${command} conda run -n dev --live-stream"
command="${command} python cli.py duckdb add-kfold"
command="${command} --db-path /mnt/cern_data/${USER}/lorenzetti/v2.2.0/ringer_dataset_jf17_zee_avgmu300.duckdb"
command="${command} --src-table data"
command="${command} --n-folds 5"
command="${command} --filter-cond \"abs(cl_eta) <= 2.5 and cl_et >= 15000\""

echo "Running command ${command} on ${img}"
singularity exec \
    --bind /mnt/cern_data:/mnt/cern_data \
    $SIF_IMGS_DIR/$img /usr/bin/bash -c "${command}"
