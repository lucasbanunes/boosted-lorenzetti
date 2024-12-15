N_FOLDS=5
python fit_mlp.py \
    --sig-dir "/root/ext_data/lorenzetti/zee-mb-2024-12-08" \
    --bkg-dir "/root/ext_data/lorenzetti/jet-mb-2024-12-08" \
    --output-dir "/root/ext_data/aprendizado-profundo/models-2024-12-15/mlpv1" \
    --dense-layers 5 \
    --n-folds $N_FOLDS

python fit_mlp.py \
    --sig-dir "/root/ext_data/lorenzetti/zee-mb-2024-12-08" \
    --bkg-dir "/root/ext_data/lorenzetti/jet-mb-2024-12-08" \
    --output-dir "/root/ext_data/aprendizado-profundo/models-2024-12-15/mlpv2" \
    --dense-layers 1 \
    --n-folds $N_FOLDS

python fit_mlp.py \
    --sig-dir "/root/ext_data/lorenzetti/zee-mb-2024-12-08" \
    --bkg-dir "/root/ext_data/lorenzetti/jet-mb-2024-12-08" \
    --output-dir "/root/ext_data/aprendizado-profundo/models-2024-12-15/mlpv3" \
    --dense-layers 2 \
    --n-folds $N_FOLDS

python fit_mlp.py \
    --sig-dir "/root/ext_data/lorenzetti/zee-mb-2024-12-08" \
    --bkg-dir "/root/ext_data/lorenzetti/jet-mb-2024-12-08" \
    --output-dir "/root/ext_data/aprendizado-profundo/models-2024-12-15/mlpv4" \
    --dense-layers 8 \
    --n-folds $N_FOLDS

python fit_mlp.py \
    --sig-dir "/root/ext_data/lorenzetti/zee-mb-2024-12-08" \
    --bkg-dir "/root/ext_data/lorenzetti/jet-mb-2024-12-08" \
    --output-dir "/root/ext_data/aprendizado-profundo/models-2024-12-15/mlpv5" \
    --dense-layers 1 2 \
    --n-folds $N_FOLDS

python fit_mlp.py \
    --sig-dir "/root/ext_data/lorenzetti/zee-mb-2024-12-08" \
    --bkg-dir "/root/ext_data/lorenzetti/jet-mb-2024-12-08" \
    --output-dir "/root/ext_data/aprendizado-profundo/models-2024-12-15/mlpv6" \
    --dense-layers 2 4 \
    --n-folds $N_FOLDS

python fit_mlp.py \
    --sig-dir "/root/ext_data/lorenzetti/zee-mb-2024-12-08" \
    --bkg-dir "/root/ext_data/lorenzetti/jet-mb-2024-12-08" \
    --output-dir "/root/ext_data/aprendizado-profundo/models-2024-12-15/mlpv7" \
    --dense-layers 8 16 \
    --n-folds $N_FOLDS