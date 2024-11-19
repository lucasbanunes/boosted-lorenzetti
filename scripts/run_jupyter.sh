# Usage:
# $launch_jupyter <jupyter-port>
# $launch_jupyter 1234

call_dir=$(pwd)
# Binds the repo to the python path
export PYTHONPATH="${PYTHONPATH}:${HOME}/workspaces/lorenzetti/boosted-lorenzetti"
export LZT_DATA="${HOME}/cern_data/lucas.nunes/lorenzetti"
cd "${HOME}/workspaces/lorenzetti/lorenzetti"
source setup_envs.sh
cd ~
jupyter lab --no-browser --port $1 --ip='*' --NotebookApp.token='' --NotebookApp.password=''
cd $call_dir
