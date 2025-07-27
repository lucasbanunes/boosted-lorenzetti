python cli.py mlp create-kfold \
    /root/data/lorenzetti/mldatasets/zee-jf17-250pileup \
    100 2 1 \
    --best-metric val_max_sp \
    --best-metric-mode max \
    --activation relu \
    --df-name data \
    --batch-size 32 \
    --patience 15 \
    --fold 5 \
    --inits 5 \
    --tracking-uri file://~/data/lorenzetti/mlruns \
    --experiment-name boosted-lorenzetti \
    --max-epochs 1000 \
    --job-name mlp-kfold-multi-init-zee-jf17-250pileup \