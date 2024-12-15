N_FOLDS=5
python fit_cnn.py \
    --sig-dir "/root/ext_data/lorenzetti/zee-mb-2024-12-08" \
    --bkg-dir "/root/ext_data/lorenzetti/jet-mb-2024-12-08" \
    --output-dir "/root/ext_data/aprendizado-profundo/models-2024-12-15/cnnv1" \
    --dense-layers 5 \
    --n-folds $N_FOLDS \
    --kernel-sizes 2 4 \
    --incep-modules 1 \
    --n-filters 2
python fit_cnn.py \
    --sig-dir "/root/ext_data/lorenzetti/zee-mb-2024-12-08" \
    --bkg-dir "/root/ext_data/lorenzetti/jet-mb-2024-12-08" \
    --output-dir "/root/ext_data/aprendizado-profundo/models-2024-12-15/cnnv2" \
    --dense-layers 5 \
    --n-folds $N_FOLDS \
    --kernel-sizes 2 4 \
    --incep-modules 1 2 \
    --n-filters 2

python fit_cnn.py \
    --sig-dir "/root/ext_data/lorenzetti/zee-mb-2024-12-08" \
    --bkg-dir "/root/ext_data/lorenzetti/jet-mb-2024-12-08" \
    --output-dir "/root/ext_data/aprendizado-profundo/models-2024-12-15/cnnv3" \
    --dense-layers 5 \
    --n-folds $N_FOLDS \
    --kernel-sizes 2 4 \
    --incep-modules 1 \
    --n-filters 4

python fit_cnn.py \
    --sig-dir "/root/ext_data/lorenzetti/zee-mb-2024-12-08" \
    --bkg-dir "/root/ext_data/lorenzetti/jet-mb-2024-12-08" \
    --output-dir "/root/ext_data/aprendizado-profundo/models-2024-12-15/cnnv4" \
    --dense-layers 5 \
    --n-folds $N_FOLDS \
    --kernel-sizes 2 4 \
    --incep-modules 1 2 \
    --n-filters 4
