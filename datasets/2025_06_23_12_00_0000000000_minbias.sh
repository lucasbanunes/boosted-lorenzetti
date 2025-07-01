call_dir=$PWD
n_workers=$2 && \
lzt_workspace=${HOME}/workspaces/lorenzetti && \
lzt_repo="${HOME}/workspaces/lorenzetti/lorenzetti" && \
nov=1000 && \
seed=729378 && \
base_dir="${1}/2025_06_23_12_00_0000000000_minbias" && \
evt_dir="${base_dir}/EVT" && \
hit_dir="${base_dir}/HIT" && \
esd_dir="${base_dir}/ESD" && \
aod_dir="${base_dir}/AOD" && \
ntuple_dir="${base_dir}/NTUPLE" && \
cd "${lzt_repo}/build" && source lzt_setup.sh && \
# generate events with pythia
mkdir -p "${base_dir}/EVT" && cd "${base_dir}/EVT" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started EVT sim" > "${base_dir}/started_EVT.log" && \
(gen_minbias.py --output-file minbias.EVT.root -nt $n_workers --nov $nov --seed $seed --events-per-job 100 --pileup-avg 1 --pileup-sigma 0 --no-poisson -o "${base_dir}/EVT/minbias.EVT.root" |& tee "${base_dir}/minbias.EVT.log")  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished EVT sim" > "${base_dir}/finished_EVT.log"
# generate hits around the truth particle seed
# mkdir -p $hit_dir && cd $hit_dir && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started HIT sim" |& tee "${base_dir}/started_HIT.log" && \
# (simu_trf.py -i $evt_dir -o "minbias.HIT.root" -nt $n_workers -t 5 |& tee "${base_dir}/minbias.HIT.log") && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished HIT sim" |& tee "${base_dir}/finished_HIT.log"

cd $call_dir
