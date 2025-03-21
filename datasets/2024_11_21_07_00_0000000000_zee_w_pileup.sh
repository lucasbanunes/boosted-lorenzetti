call_dir=$PWD
n_workers=$1 && \
overwrite=$2 && \
base_dir="${LZT_DATA}/2024_11_21_07_00_0000000000_zee_w_pileup" && \
boosted_repo_dir="${HOME}/workspaces/lorenzetti/boosted-lorenzetti" && \
src_zee="${LZT_DATA}/2024_08_20_21_08_0000000000_zee" && \
src_pileup="${LZT_DATA}/2024_08_22_12_00_0000000000_minibias" && \
if [ -d "$base_dir" ] && [ "$overwrite" != "overwrite" ]; then
    echo "Directory $base_dir already exists. Exiting..."
    exit 1
else
    rm -rf $base_dir && mkdir -p $base_dir
fi
# generate hits around the truth particle seed
mkdir -p "${base_dir}/HIT" && cd "${base_dir}/HIT" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started HIT sim" > "${base_dir}/started_HIT.log" && cd $boosted_repo_dir && \
(python merge_pileup_files.py -i /root/data/local/lorenzetti/2024_08_19_21_08_0000000000_zee/HIT \
                             -p /root/data/local/lorenzetti/2024_09_17_08_00_0000000000_minibias/HIT/minias.HIT.root \
                             -o /root/data/local/lorenzetti/2024_11_21_07_00_0000000000_zee_w_pileup/HIT \
                             --output-name zee_w_pileup |& tee "${base_dir}/HIT.log") && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished HIT sim" > "${base_dir}/finished_HIT.log" && \
# digitalization
mkdir -p "${base_dir}/ESD" && cd "${base_dir}/ESD" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started ESD sim" > "${base_dir}/started_ESD.log" && \
(prun_jobs.py -c "digit_trf.py -i %IN -o %OUT" -i "${base_dir}/HIT" -o "${base_dir}/ESD/zee_w_pileup.ESD.root" -nt $n_workers |& tee "${base_dir}/ESD.log") && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished ESD sim" > "${base_dir}/finished_ESD.log" && \
# reconstruction
mkdir -p "${base_dir}/AOD" && cd "${base_dir}/AOD" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started AOD sim" > "${base_dir}/started_AOD.log" && \
(prun_jobs.py -c "reco_trf.py -i %IN -o %OUT" -i "${base_dir}/ESD" -o "${base_dir}/AOD/zee_w_pileup.AOD.root" -nt $n_workers -m |& tee "${base_dir}/AOD.log" )  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished AOD sim" > "${base_dir}/finished_AOD.log" && \
# ntuple
mkdir -p "${base_dir}/NTUPLE" && cd "${base_dir}/NTUPLE" && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Started NTUPLE sim" > "${base_dir}/started_NTUPLE.log" && \
(ntuple_trf.py -i "${base_dir}/AOD/zee_w_pileup.AOD.root" -o "${base_dir}/NTUPLE/zee_w_pileup.NTUPLE.root" |& tee "${base_dir}/NTUPLE.log" )  && \
echo "$(date -d "today" +"%Y/%m/%d %H-%M-%s") - Finished NTUPLE sim" > "${base_dir}/finished_NTUPLE.log"

cd $call_dir
