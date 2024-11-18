call_dir=$PWD
NOV=10000 && \
seed=242746 && \
base_dir="/home/lucasbanunes/data//lorenzetti/2024_08_19_10_41_0000000000_lorenzetti_gg2H2ZZ2ee" && \
# generate events with pythia
mkdir -p "${base_dir}/EVT" && cd "${base_dir}/EVT" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started EVT sim" > "${base_dir}/started_EVT.log" && \
(prun_evts.py -c "gen_ggH.py --pileupAvg 0 --nov %NOV --eventNumber %OFFSET -o %OUT -s %SEED" -nt 40 --nov $NOV --seed $seed --novPerJob 200 -o "${base_dir}/EVT/gg2H2ZZ2ee.EVT.root" |& tee "${base_dir}/gg2H2ZZ2ee.EVT.log")  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished EVT sim" > "${base_dir}/finished_EVT.log"
# generate hits around the truth particle seed
mkdir -p "${base_dir}/HIT" && cd "${base_dir}/HIT" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started HIT sim" > "${base_dir}/started_HIT.log" && \
(prun_jobs.py -c "simu_trf.py -i %IN -o %OUT -nt 40 -t 10" -nt 1 -i "${base_dir}/EVT" -o "${base_dir}/HIT/gg2H2ZZ2ee.HIT.root" |& tee "${base_dir}/gg2H2ZZ2ee.HIT.log") && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished HIT sim" > "${base_dir}/finished_HIT.log" && \
# digitalization
mkdir -p "${base_dir}/ESD" && cd "${base_dir}/ESD" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started ESD sim" > "${base_dir}/started_ESD.log" && \
(prun_jobs.py -c "digit_trf.py -i %IN -o %OUT" -i "${base_dir}/HIT" -o "${base_dir}/ESD/gg2H2ZZ2ee.ESD.root" -nt 40 |& tee "${base_dir}/gg2H2ZZ2ee.ESD.log") && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished ESD sim" > "${base_dir}/finished_ESD.log" && \
# reconstruction
mkdir -p "${base_dir}/AOD" && cd "${base_dir}/AOD" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started AOD sim" > "${base_dir}/started_AOD.log" && \
(prun_jobs.py -c "reco_trf.py -i %IN -o %OUT" -i "${base_dir}/ESD" -o "${base_dir}/AOD/gg2H2ZZ2ee.AOD.root" -nt 40 -m |& tee "${base_dir}/gg2H2ZZ2ee.AOD.log" )  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished AOD sim" > "${base_dir}/finished_AOD.log" && \
# ntuple
mkdir -p "${base_dir}/NTUPLE" && cd "${base_dir}/NTUPLE" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started NTUPLE sim" > "${base_dir}/started_NTUPLE.log" && \
(ntuple_trf.py -i "${base_dir}/AOD/gg2H2ZZ2ee.AOD.root" -o "${base_dir}/NTUPLE/gg2H2ZZ2ee.NTUPLE.root" |& tee "${base_dir}/gg2H2ZZ2ee.NTUPLE.log" )  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished NTUPLE sim" > "${base_dir}/finished_NTUPLE.log"

cd $call_dir
