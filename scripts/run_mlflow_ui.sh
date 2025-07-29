lzt_dataset_path=$1
port=$2

cd ${lzt_dataset_path} && conda run -n dev mlflow ui -h 0.0.0.0 -p ${port}