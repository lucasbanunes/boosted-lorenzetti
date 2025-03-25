call_dir=$PWD
n_workers=$2 && \
NOV=10000 && \
seed=3973534 && \
base_dir="${1}/2025_03_22_10_00_0000000000_zee" && \
evt_dir="${base_dir}/EVT" && \
hit_dir="${base_dir}/HIT" && \
esd_dir="${base_dir}/ESD" && \
source /hep/setup_hep.sh && cd "${HOME}/workspaces/lorenzetti/lorenzetti" && \
source setup.sh && cd $call_dir && \
# generate events with pythia
mkdir -p $evt_dir && cd $evt_dir && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started EVT sim" > "${base_dir}/started_EVT.log" && \
(gen_zee.py --output-file zee.EVT.root --nov $NOV --pileup-avg 0 --seed $seed -nt $n_workers --events-per-job 500 |& tee "${base_dir}/zee.EVT.log") && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished EVT sim" > "${base_dir}/finished_EVT.log" && \
# generate hits around the truth particle seed
mkdir -p $hit_dir && cd $hit_dir && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started HIT sim" > "${base_dir}/started_HIT.log" && \
(simu_trf.py -i $evt_dir -o "zee.HIT.root" -nt $n_workers -t 10 |& tee "${base_dir}/zee.HIT.log") && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished HIT sim" > "${base_dir}/finished_HIT.log"
# # digitalization
mkdir -p $esd_dir && cd $esd_dir && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started ESD sim" > "${base_dir}/started_ESD.log" && \
(digit_trf.py -i $hit_dir -o "zee.ESD.root" -nt $n_workers |& tee "${base_dir}/zee.ESD.log") && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished ESD sim" > "${base_dir}/finished_ESD.log"
# # reconstruction
# mkdir -p "${base_dir}/AOD" && cd "${base_dir}/AOD" && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started AOD sim" > "${base_dir}/started_AOD.log" && \
# (prun_jobs.py -c "reco_trf.py -i %IN -o %OUT" -i "${base_dir}/ESD" -o "${base_dir}/AOD/zee.AOD.root" -nt $n_workers -m |& tee "${base_dir}/zee.AOD.log" )  && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished AOD sim" > "${base_dir}/finished_AOD.log" && \
# # ntuple
# mkdir -p "${base_dir}/NTUPLE" && cd "${base_dir}/NTUPLE" && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started NTUPLE sim" > "${base_dir}/started_NTUPLE.log" && \
# (ntuple_trf.py -i "${base_dir}/AOD/zee.AOD.root" -o "${base_dir}/NTUPLE/zee.NTUPLE.root" |& tee "${base_dir}/zee.NTUPLE.log" )  && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished NTUPLE sim" > "${base_dir}/finished_NTUPLE.log"

cd $call_dir
