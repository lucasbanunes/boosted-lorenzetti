call_dir=$PWD
NOV=10000 && \
seed=108625 && \
# base_dir="/home/lucas.nunes/data/$(date -d "today" +"%Y_%m_%d_%H_%M_%s")_lorenzetti_single_electron" && \
base_dir="/home/lucas.nunes/cern_data/lucas.nunes/lorenzetti/2024_08_19_21_18_0000000000_single_electron" && \
# # generate events with pythia
# mkdir -p "${base_dir}/EVT" && cd "${base_dir}/EVT" && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started EVT sim" > "${base_dir}/started_EVT.log" && \
# (prun_evts.py -c "gen_electron.py --pileupAvg 0 --doEtaRanged True --eta_min -2.5 --eta_max 2.5 --energy_min 15 --energy_max 2000 --doPhiRanged True --phi_min -3.14159 --phi_max 3.14159 --nov %NOV --eventNumber %OFFSET -o %OUT -s %SEED" -nt 8 --nov $NOV --seed $seed --novPerJob 200 -o "${base_dir}/EVT/single_electron.EVT.root" |& tee "${base_dir}/single_electron.EVT.log")  && \
# echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished EVT sim" > "${base_dir}/finished_EVT.log" && \
# generate hits around the truth particle seed
mkdir -p "${base_dir}/HIT" && cd "${base_dir}/HIT" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started HIT sim" > "${base_dir}/started_HIT.log" && \
(prun_jobs.py -c "simu_trf.py -i %IN -o %OUT -nt 8 -t 10" -nt 1 -i "${base_dir}/EVT" -o "${base_dir}/HIT/single_electron.HIT.root" |& tee "${base_dir}/single_electron.HIT.log") && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished HIT sim" > "${base_dir}/finished_HIT.log" && \
digitalization
mkdir -p "${base_dir}/ESD" && cd "${base_dir}/ESD" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started ESD sim" > "${base_dir}/started_ESD.log" && \
(prun_jobs.py -c "digit_trf.py -i %IN -o %OUT" -i "${base_dir}/HIT" -o "${base_dir}/ESD/single_electron.ESD.root" -nt 8 |& tee "${base_dir}/single_electron.ESD.log") && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished ESD sim" > "${base_dir}/finished_ESD.log" && \
# reconstruction
mkdir -p "${base_dir}/AOD" && cd "${base_dir}/AOD" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started AOD sim" > "${base_dir}/started_AOD.log" && \
(prun_jobs.py -c "reco_trf.py -i %IN -o %OUT" -i "${base_dir}/ESD" -o "${base_dir}/AOD/single_electron.AOD.root" -nt 8 -m |& tee "${base_dir}/single_electron.AOD.log" )  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished AOD sim" > "${base_dir}/finished_AOD.log" && \
# ntuple
mkdir -p "${base_dir}/NTUPLE" && cd "${base_dir}/NTUPLE" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started NTUPLE sim" > "${base_dir}/started_NTUPLE.log" && \
(ntuple_trf.py -i "${base_dir}/AOD/single_electron.AOD.root" -o "${base_dir}/NTUPLE/single_electron.NTUPLE.root" |& tee "${base_dir}/single_electron.NTUPLE.log" )  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished NTUPLE sim" > "${base_dir}/finished_NTUPLE.log"

cd $call_dir
