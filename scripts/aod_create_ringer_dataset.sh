#!/bin/bash
#SBATCH --job-name=aod-create-ringer-dataset
#SBATCH -o /home/lucas.nunes/logs/boosted-lorenzetti/%x-%j.out
# %x is the job name
# %j is the job ID

img="boosted-lorenzetti_latest.sif"
command="cd /home/${USER}/workspaces/lorenzetti/boosted-lorenzetti &&"
command="${command} conda run -n dev --live-stream"
command="${command} python cli.py aod create-ringer-dataset"
command="${command} --input-dbs"
command="${command} \"/mnt/cern_data/${USER}/lorenzetti/v2.2.0/user.joao.pinto.mc25_13TeV.250601.Pythia8EvtGen_JF17.50k.avgmu300_sigmamu50_stage_3.result.AOD.duckdb,"
command="${command} /mnt/cern_data/${USER}/lorenzetti/v2.2.0/uuser.joao.pinto.mc25_13TeV.250601.Pythia8EvtGen_Zee.50k.avgmu300_sigmamu50_stage_3.result.AOD.duckdb\""
command="${command} --labels \"0, 1\""
command="${command} --output-file /mnt/cern_data/${USER}/lorenzetti/v2.2.0/ringer_dataset_jf17_zee_avgmu300.duckdb"
command="${command} --description 'Ringer dataset with JF17 and Zee samples, avgmu 300'"

echo "Running command ${command} on ${img}"
singularity exec \
    --bind /mnt/cern_data:/mnt/cern_data \
    $SIF_IMGS_DIR/$img /usr/bin/bash -c "${command}"
