from boosted_lorenzetti import kmeans
from pathlib import Path
import subprocess
import logging

from boosted_lorenzetti.constants import N_RINGS


def test_full_training(test_dataset_path: Path):
    experiment_name = 'kmeans_test_full_training'

    ring_cols = [f'cl_rings[{i+1}]' for i in range(N_RINGS)]
    query_cols = ring_cols + ['label']
    query_cols_str = ', '.join(query_cols)
    train_query = f"SELECT {query_cols_str} FROM data WHERE fold != 0;"
    test_query = f"SELECT {query_cols_str} FROM data WHERE fold = 0;"

    run_id = kmeans.cli.create_training(
        db_path=test_dataset_path,
        train_query=train_query,
        test_query=test_query,
        label_cols='label',
        experiment_name=experiment_name,
        n_clusters=2
    )

    kmeans.cli.run_training(
        run_ids=[run_id],
        experiment_name=experiment_name,
    )

    kmeans.jobs.KMeansTrainingJob.from_mlflow_run_id(run_id)


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

    run_id = kmeans.cli.create_best_cluster_number_search(
        db_path=test_dataset_path,
        train_query=train_query,
        test_query=test_query,
        label_cols='label',
        experiment_name=experiment_name,
        clusters=[1, 2, 3, 4, 5]
    )

    kmeans.cli.run_best_cluster_number_search(
        run_ids=[run_id],
        experiment_name=experiment_name
    )

    kmeans.jobs.BestClusterNumberSearch.from_mlflow_run_id(run_id)


def test_kfold_kmeans(test_dataset_path: Path):
    experiment_name = 'test_kfold_kmeans'

    ring_cols = [f'cl_rings[{i+1}]' for i in range(N_RINGS)]

    run_id = kmeans.cli.create_kfold(
        db_path=test_dataset_path,
        table_name='data',
        feature_cols=ring_cols,
        best_metric='val.inertia',
        best_metric_mode='min',
        n_folds=5,
        clusters=list(range(1, 5)),
        label_col='label',
        fold_col='fold',
        experiment_name=experiment_name,
    )

    kmeans.cli.run_kfold(
        run_ids=[run_id],
        experiment_name=experiment_name
    )

    kmeans.jobs.KFoldKMeansTrainingJob.from_mlflow_run_id(run_id)


def test_kfold_kmeans_cli(test_dataset_path: Path,
                          repo_path: Path):
    experiment_name = 'test_kfold_kmeans'

    ring_cols = ', '.join([f'cl_rings[{i+1}]' for i in range(N_RINGS)])
    command = ['python',
               f'{str(repo_path)}/cli.py',
               'kmeans',
               'create-kfold',
               '--db-path', str(test_dataset_path),
               '--table-name', 'data',
               '--feature-cols', ring_cols,
               '--best-metric', 'val.inertia',
               '--best-metric-mode', 'min',
               '--n-folds', '5',
               '--clusters', '1, 2, 3, 4, 5',
               '--label-col', 'label',
               '--fold-col', 'fold',
               '--experiment-name', experiment_name,
               ]
    logging.info("Running KFold KMeans via CLI")
    logging.info(f"Command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True)
    logging.info("STDOUT: %s", result.stdout)
    logging.error("STDERR: %s", result.stderr)
    assert result.returncode == 0, "KFold KMeans CLI command failed"
