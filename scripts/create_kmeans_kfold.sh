#!/bin/bash
#SBATCH --job-name=create-kmeans-kfold
#SBATCH -o /home/lucas.nunes/logs/boosted-lorenzetti/%x-%j.out
# %x is the job name
# %j is the job ID

# Usage
# $ sbatch create_kmeans_kfold.sh <img> <db-path> <tracking-uri> <checkpoints-dir>
# $ sbatch create_kmeans_kfold.sh boosted-lorenzetti_0.1.0.sif /home/test.user/lorenzetti.db /home/test.user/mlruns checkpoints

img=$1
db_path=$2
tracking_uri=$3
checkpoints_dir="${4}/checkpoints/mlp_kfold_$(date -d "today" +"%Y_%m_%d_%H_%M_%s")"
command="cd /home/${USER}/workspaces/lorenzetti/boosted-lorenzetti &&"
command="${command} conda run -n dev --live-stream"
command="${command} python cli.py kmeans create-kfold"
command="${command} --db-path /root/data/lorenzetti/v2.2.0/zee-jf17-250pileup-v2.2.0.duckdb"
command="${command} --table-name data"
command="${command} --feature-cols"
command="${command} \"trig_L2_cl_ring_1, trig_L2_cl_ring_2, trig_L2_cl_ring_3, trig_L2_cl_ring_4, trig_L2_cl_ring_5, trig_L2_cl_ring_6, trig_L2_cl_ring_7, trig_L2_cl_ring_8, trig_L2_cl_ring_9, trig_L2_cl_ring_10,"
command="${command} trig_L2_cl_ring_11, trig_L2_cl_ring_12, trig_L2_cl_ring_13, trig_L2_cl_ring_14, trig_L2_cl_ring_15, trig_L2_cl_ring_16, trig_L2_cl_ring_17, trig_L2_cl_ring_18, trig_L2_cl_ring_19, trig_L2_cl_ring_20,"
command="${command} trig_L2_cl_ring_21, trig_L2_cl_ring_22, trig_L2_cl_ring_23, trig_L2_cl_ring_24, trig_L2_cl_ring_25, trig_L2_cl_ring_26, trig_L2_cl_ring_27, trig_L2_cl_ring_28, trig_L2_cl_ring_29, trig_L2_cl_ring_30,"
command="${command} trig_L2_cl_ring_31, trig_L2_cl_ring_32, trig_L2_cl_ring_33, trig_L2_cl_ring_34, trig_L2_cl_ring_35, trig_L2_cl_ring_36, trig_L2_cl_ring_37, trig_L2_cl_ring_38, trig_L2_cl_ring_39, trig_L2_cl_ring_40,"
command="${command} trig_L2_cl_ring_41, trig_L2_cl_ring_42, trig_L2_cl_ring_43, trig_L2_cl_ring_44, trig_L2_cl_ring_45, trig_L2_cl_ring_46, trig_L2_cl_ring_47, trig_L2_cl_ring_48, trig_L2_cl_ring_49, trig_L2_cl_ring_50,"
command="${command} trig_L2_cl_ring_51, trig_L2_cl_ring_52, trig_L2_cl_ring_53, trig_L2_cl_ring_54, trig_L2_cl_ring_55, trig_L2_cl_ring_56, trig_L2_cl_ring_57, trig_L2_cl_ring_58, trig_L2_cl_ring_59, trig_L2_cl_ring_60,"
command="${command} trig_L2_cl_ring_61, trig_L2_cl_ring_62, trig_L2_cl_ring_63, trig_L2_cl_ring_64, trig_L2_cl_ring_65, trig_L2_cl_ring_66, trig_L2_cl_ring_67, trig_L2_cl_ring_68, trig_L2_cl_ring_69, trig_L2_cl_ring_70,"
command="${command} trig_L2_cl_ring_71, trig_L2_cl_ring_72, trig_L2_cl_ring_73, trig_L2_cl_ring_74, trig_L2_cl_ring_75, trig_L2_cl_ring_76, trig_L2_cl_ring_77, trig_L2_cl_ring_78, trig_L2_cl_ring_79, trig_L2_cl_ring_80,"
command="${command} trig_L2_cl_ring_81, trig_L2_cl_ring_82, trig_L2_cl_ring_83, trig_L2_cl_ring_84, trig_L2_cl_ring_85, trig_L2_cl_ring_86, trig_L2_cl_ring_87, trig_L2_cl_ring_88, trig_L2_cl_ring_89, trig_L2_cl_ring_90,"
command="${command} trig_L2_cl_ring_91, trig_L2_cl_ring_92, trig_L2_cl_ring_93, trig_L2_cl_ring_94, trig_L2_cl_ring_95, trig_L2_cl_ring_96, trig_L2_cl_ring_97, trig_L2_cl_ring_98, trig_L2_cl_ring_99, rig_L2_cl_ring_100]\""
command="${command} --best-metric val.inertia"
command="${command} --best-metric-mode min"
command="${command} --fold-col fold"
command="${command} --n-folds 5"
command="${command} --clusters \"1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20\""
command="${command} --label-col label"
command="${command} --name \"KMeans K-Fold on Zee JF17 250 Pileup v2.2.0 - 2\""
command="${command} --experiment-name \"boosted-lorenzetti\""
command="${command} --tracking-uri ${tracking_uri}"

echo "Running command ${command} on ${img}"
mkdir -p $checkpoints_dir && \
singularity exec \
    --nv \
    --bind /mnt/cern_data:/mnt/cern_data \
    $SIF_IMGS_DIR/$img /usr/bin/bash -c "${command}"