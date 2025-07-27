python cli.py ingest \
    ~/data/lorenzetti/v2.2.0/user.joao.pinto.mc25_13TeV.250520.Pythia8EvtGen_Zee.100k.avgmu250_sigmamu50 \
    ~/data/lorenzetti/v2.2.0/user.joao.pinto.mc25_13TeV.250531.Pythia8EvtGen_JF17.100k.avgmu250_sigmamu50 \
    zee-jf17-250pileup \
    /root/data/lorenzetti/mldatasets \
    v2.2.0 \
    --experiment-name boosted-lorenzetti \
    --n-folds 5 \
    --seed 64387340 \
    --description "Zee vs JF17 250 pileup dataset for electron classification" \
    --tracking-uri file://~/data/lorenzetti/mlruns \
    --no-include-metrics \
    --query "abs(cl_eta) < 2.5" # Central barrel data only