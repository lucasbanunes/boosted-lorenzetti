#!/bin/bash
#SBATCH --job-name=create-dataset-lzt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH -o /home/lucas.nunes/logs/boosted-lorenzetti/%x-%j.out
# %x is the job name
# %j is the job ID

# Usage
# $ sbatch create_dataset.sh <img>
# $ sbatch create_dataset.sh boosted-lorenzetti_0.1.0.sif

img=$1
command="conda run -n dev --live-stream"
command="${command} python cli.py ntuple cli-create-dataset"
command="${command} /mnt/cern_data/lucas.nunes/lorenzetti/v2.2.0/user.joao.pinto.mc25_13TeV.250520.Pythia8EvtGen_Zee.100k.avgmu250_sigmamu50_stage_4.result.NTUPLE \"1\""
command="${command} /mnt/cern_data/lucas.nunes/lorenzetti/v2.2.0/user.joao.pinto.mc25_13TeV.250531.Pythia8EvtGen_JF17.100k.avgmu250_sigmamu50_stage_4.result.NTUPLE \"0\""
command="${command} --output-path /mnt/cern_data/lucas.nunes/lorenzetti/v2.2.0/zee-jf17-250pileup-v2.2.0.duckdb"
command="${command} --table-name data"
command="${command} --lzt-version v2.2.0"
command="${command} --n-folds 5"
command="${command} --query \"abs(cl_eta) <=2.5 and cl_et >= 15000\"" # Energy is in MeV
command="${command} --tracking-uri file:///mnt/cern_data/lucas.nunes/lorenzetti/mlruns"

echo "Running command ${command} on ${img}"
singularity exec \
    --nv \
    --bind /mnt/cern_data:/mnt/cern_data \
    $SIF_IMGS_DIR/$img /usr/bin/bash -c "${command}"
