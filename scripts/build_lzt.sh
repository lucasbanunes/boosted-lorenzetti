call_dir=$PWD
logs_dir=~/logs/lorenzetti
log_path="${logs_dir}/lorenzetti_build_$(date -d "today" +"%Y_%m_%d_%H_%M_%s").log"
mkdir -p $logs_dir
# |& redirects stderr to stdout from the previous command do the stdin of the next command
# in this case redirects stderr and stdout to tee, which writes to a file and stdout
lzt_repo=/root/workspaces/lorenzetti/lorenzetti
# build_dir=$lzt_repo/build
# if [ -d "$build_dir" ]; then
#     rm -r $build_dir
# fi
cd $lzt_repo && (make |& tee $log_path) && source setup.sh
cd $call_dir