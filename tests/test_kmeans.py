from boosted_lorenzetti.models import kmeans
from pathlib import Path

from boosted_lorenzetti.constants import N_RINGS


def test_full_training(test_dataset_path: Path):
    experiment_name = 'kmeans_test_full_training'

    ring_cols = [f'cl_rings[{i+1}]' for i in range(N_RINGS)]
    query_cols = ring_cols + ['label']
    query_cols_str = ', '.join(query_cols)
    train_query = f"SELECT {query_cols_str} FROM data WHERE fold != 0;"
    val_query = f"SELECT {query_cols_str} FROM data WHERE fold = 0;"

    run_id = kmeans.create_training(
        db_path=test_dataset_path,
        train_query=train_query,
        val_query=val_query,
        label_cols='label',
        experiment_name=experiment_name,
        n_clusters=2
    )

    kmeans.run_training(
        run_ids=[run_id],
        experiment_name=experiment_name,
    )
