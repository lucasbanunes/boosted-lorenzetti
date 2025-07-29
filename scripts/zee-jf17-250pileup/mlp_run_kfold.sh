kfold_run_id=$1
lzt_dataset_path=$2

conda run -n dev python cli.py mlp run-kfold \
    $kfold_run_id \
    --tracking-uri file://${lzt_dataset_path}/mlruns \
    --experiment-name boosted-lorenzetti