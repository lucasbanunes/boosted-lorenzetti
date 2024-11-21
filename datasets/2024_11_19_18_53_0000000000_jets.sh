# Generates a dataset out of jets
# Usage example 
# 2024_11_19_18_53_0000000000_jets.sh <n_events> <n_workers>
# 2024_11_19_18_53_0000000000_jets.sh 100 10
# This will generate a dataset with 100 events using 10 workers
call_dir=$PWD
NOV=$1 && \
n_workers=$2 && \
seed=2137420 && \
base_dir="${LZT_DATA}/2024_11_19_18_53_0000000000_jets"
# if [ -d "$base_dir" ]; then
#     echo "Directory $base_dir already exists. Exiting..."
#     exit 1
# fi
lzt_path="${HOME}/workspaces/lorenzetti/lorenzetti" && \
cd $lzt_path && source setup_envs.sh && source setup.sh && \
# generate events with pythia
# mkdir -p "${base_dir}/EVT" && cd "${base_dir}/EVT" && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started EVT sim" > "${base_dir}/started_EVT.log" && \
# (prun_evts.py -c "gen_jets.py --pileupAvg 0 --nov %NOV --eventNumber %OFFSET -o %OUT -s %SEED" -nt $n_workers --nov $NOV --seed $seed --novPerJob 200 -o "${base_dir}/EVT/jf17.EVT.root" |& tee "${base_dir}/jf17.EVT.log")  && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished EVT sim" > "${base_dir}/finished_EVT.log" && \
# # generate hits around the truth particle seed
# mkdir -p "${base_dir}/HIT" && cd "${base_dir}/HIT" && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started HIT sim" > "${base_dir}/started_HIT.log" && \
# (prun_jobs.py -c "simu_trf.py -i %IN -o %OUT -nt $n_workers -t 10" -nt 1 -i "${base_dir}/EVT" -o "${base_dir}/HIT/jf17.HIT.root" |& tee "${base_dir}/jf17.HIT.log") && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished HIT sim" > "${base_dir}/finished_HIT.log" && \
# # digitalization
# mkdir -p "${base_dir}/ESD" && cd "${base_dir}/ESD" && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started ESD sim" > "${base_dir}/started_ESD.log" && \
# (prun_jobs.py -c "digit_trf.py -i %IN -o %OUT" -i "${base_dir}/HIT" -o "${base_dir}/ESD/jf17.ESD.root" -nt $n_workers |& tee "${base_dir}/jf17.ESD.log") && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished ESD sim" > "${base_dir}/finished_ESD.log" && \
# reconstruction
mkdir -p "${base_dir}/AOD" && cd "${base_dir}/AOD" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started AOD sim" > "${base_dir}/started_AOD.log" && \
(prun_jobs.py -c "reco_trf.py -i %IN -o %OUT" -i "${base_dir}/ESD" -o "${base_dir}/AOD/jf17.AOD.root" -nt $n_workers -m |& tee "${base_dir}/jf17.AOD.log" )  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished AOD sim" > "${base_dir}/finished_AOD.log" && \
# ntuple
mkdir -p "${base_dir}/NTUPLE" && cd "${base_dir}/NTUPLE" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started NTUPLE sim" > "${base_dir}/started_NTUPLE.log" && \
(ntuple_trf.py -i "${base_dir}/AOD/jf17.AOD.root" -o "${base_dir}/NTUPLE/jf17.NTUPLE.root" |& tee "${base_dir}/jf17.NTUPLE.log" )  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished NTUPLE sim" > "${base_dir}/finished_NTUPLE.log"

cd $call_dir
