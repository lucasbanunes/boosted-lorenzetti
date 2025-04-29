call_dir=$PWD
n_workers=2 && \
lzt_workspace=${HOME}/workspaces/lorenzetti && \
lzt_repo="${HOME}/workspaces/lorenzetti/lorenzetti" && \
nov=20 && \
seed=365024 && \
base_dir="${1}/2025_04_29_15_34_0000000000_gg2H2ZZ2ee" && \
evt_dir="${base_dir}/EVT" && \
hit_dir="${base_dir}/HIT" && \
esd_dir="${base_dir}/ESD" && \
aod_dir="${base_dir}/AOD" && \
ntuple_dir="${base_dir}/NTUPLE" && \
cd "${lzt_repo}/build" && source lzt_setup.sh && \
# generate events with pythia
mkdir -p "${base_dir}/EVT" && cd "${base_dir}/EVT" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started EVT sim" > "${base_dir}/started_EVT.log" && \
(gen_overlapped_zee.py --output-file gg2H2ZZ2ee.EVT.root -nt $n_workers --nov $nov --seed $seed --events-per-job 10 -o "${base_dir}/EVT/gg2H2ZZ2ee.EVT.root" |& tee "${base_dir}/gg2H2ZZ2ee.EVT.log")  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished EVT sim" > "${base_dir}/finished_EVT.log"
# generate hits around the truth particle seed
mkdir -p $hit_dir && cd $hit_dir && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started HIT sim" |& tee "${base_dir}/started_HIT.log" && \
(simu_trf.py -i $evt_dir -o "gg2H2ZZ2ee.HIT.root" -nt $n_workers -t 5 |& tee "${base_dir}/gg2H2ZZ2ee.HIT.log") && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished HIT sim" |& tee "${base_dir}/finished_HIT.log"
# digitalization
mkdir -p $esd_dir && cd $esd_dir && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started ESD sim" |& tee "${base_dir}/started_ESD.log" && \
(digit_trf.py -i $hit_dir -o "gg2H2ZZ2ee.ESD.root" -nt $n_workers |& tee "${base_dir}/gg2H2ZZ2ee.ESD.log") && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished ESD sim" |& tee "${base_dir}/finished_ESD.log"
# reconstruction
mkdir -p $aod_dir && cd $aod_dir && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started AOD sim" |& tee "${base_dir}/started_AOD.log" && \
(reco_trf.py -i $esd_dir -o "gg2H2ZZ2ee.AOD.root" -nt $n_workers |& tee "${base_dir}/gg2H2ZZ2ee.AOD.log" )  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished AOD sim" |& tee "${base_dir}/finished_AOD.log"
# ntuple
mkdir -p $ntuple_dir && cd $ntuple_dir && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started NTUPLE sim" |& tee "${base_dir}/started_NTUPLE.log" && \
(ntuple_trf.py -i $aod_dir -o "gg2H2ZZ2ee.NTUPLE.root" -nt $n_workers |& tee "${base_dir}/gg2H2ZZ2ee.NTUPLE.log" )  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished NTUPLE sim" |& tee "${base_dir}/finished_NTUPLE.log"

cd $call_dir
