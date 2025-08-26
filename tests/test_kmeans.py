from boosted_lorenzetti.models import kmeans
from pathlib import Path
import subprocess

from boosted_lorenzetti.constants import N_RINGS


def test_full_training(test_dataset_path: Path):
    experiment_name = 'kmeans_test_full_training'

    ring_cols = [f'cl_rings[{i+1}]' for i in range(N_RINGS)]
    query_cols = ring_cols + ['label']
    query_cols_str = ', '.join(query_cols)
    train_query = f"SELECT {query_cols_str} FROM data WHERE fold != 0;"
    test_query = f"SELECT {query_cols_str} FROM data WHERE fold = 0;"

    run_id = kmeans.create_training(
        db_path=test_dataset_path,
        train_query=train_query,
        test_query=test_query,
        label_cols='label',
        experiment_name=experiment_name,
        n_clusters=2
    )

    kmeans.run_training(
        run_ids=[run_id],
        experiment_name=experiment_name,
    )

    kmeans.KMeansTrainingJob.from_mlflow_run_id(run_id)


def test_full_training_cli(test_dataset_path: Path,
                           repo_path: Path):
    experiment_name = 'kmeans_test_full_training'

    ring_cols = [f'cl_rings[{i+1}]' for i in range(N_RINGS)]
    query_cols = ring_cols + ['label']
    query_cols_str = ', '.join(query_cols)
    train_query = f"SELECT {query_cols_str} FROM data WHERE fold != 0;"
    test_query = f"SELECT {query_cols_str} FROM data WHERE fold = 0;"

    subprocess.run(['python',
                    f'{str(repo_path)}/cli.py',
                    'kmeans',
                    'create-training',
                    '--db-path', str(test_dataset_path),
                    '--train-query', train_query,
                    '--val-query', test_query,
                    '--label-cols', 'label',
                    '--experiment-name', experiment_name,
                    '--n-clusters', '2'])


def test_best_cluster_number_search(test_dataset_path: Path):
    experiment_name = 'kmeans_test_best_cluster_number_search'

    ring_cols = [f'cl_rings[{i+1}]' for i in range(N_RINGS)]
    query_cols = ring_cols + ['label']
    query_cols_str = ', '.join(query_cols)
    train_query = f"SELECT {query_cols_str} FROM data WHERE fold != 0;"
    test_query = f"SELECT {query_cols_str} FROM data WHERE fold = 0;"

    run_id = kmeans.create_best_cluster_number_search(
        db_path=test_dataset_path,
        train_query=train_query,
        test_query=test_query,
        label_cols='label',
        experiment_name=experiment_name,
        clusters=[1, 2, 3, 4, 5]
    )

    kmeans.run_best_cluster_number_search(
        run_ids=[run_id],
        experiment_name=experiment_name
    )

    kmeans.BestClusterNumberSearch.from_mlflow_run_id(run_id)


def test_kfold_kmeans(test_dataset_path: Path):
    experiment_name = 'test_kfold_kmeans'

    ring_cols = [f'cl_rings[{i+1}]' for i in range(N_RINGS)]

    run_id = kmeans.create_kfold(
        db_path=test_dataset_path,
        table_name='data',
        feature_cols=ring_cols,
        best_metric='test.inertia',
        best_metric_mode='min',
        n_folds=5,
        clusters=list(range(1, 5)),
        label_col='label',
        fold_col='fold',
        experiment_name=experiment_name,
    )

    kmeans.run_kfold(
        run_ids=[run_id],
        experiment_name=experiment_name
    )

    kmeans.KFoldKMeansTrainingJob.from_mlflow_run_id(run_id)
