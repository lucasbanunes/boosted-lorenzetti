from boosted_lorenzetti.mlp.cli import create_training, run_training, create_kfold, run_kfold
from pathlib import Path
import subprocess


from boosted_lorenzetti.constants import N_RINGS


def test_mlp_full_training(test_dataset_path: Path):
    experiment_name = 'test_mlp_full_training'

    ring_cols = [f'cl_rings[{i+1}]' for i in range(N_RINGS)]
    query_cols = ring_cols + ['label']
    query_cols_str = ', '.join(query_cols)
    train_query = f"SELECT {query_cols_str} FROM data WHERE fold != 0;"
    val_query = f"SELECT {query_cols_str} FROM data WHERE fold = 0;"

    run_id = create_training(
        db_path=test_dataset_path,
        train_query=train_query,
        val_query=val_query,
        label_col='label',
        dims="100, 2, 1",
        experiment_name=experiment_name
    )

    run_training(
        run_ids=run_id,
        experiment_name=experiment_name,
    )


def test_mlp_multiple_trainings(test_dataset_path: Path, repo_path: Path):
    experiment_name = 'test_mlp_multiple_trainings'

    ring_cols = [f'cl_rings[{i+1}]' for i in range(N_RINGS)]
    query_cols = ring_cols + ['label']
    query_cols_str = ', '.join(query_cols)
    train_query = f"SELECT {query_cols_str} FROM data WHERE fold != 0;"
    val_query = f"SELECT {query_cols_str} FROM data WHERE fold = 0;"

    run_ids = []
    run_ids.append(
        create_training(
            db_path=test_dataset_path,
            train_query=train_query,
            val_query=val_query,
            label_col='label',
            dims="100, 2, 1",
            experiment_name=experiment_name
        )
    )
    run_ids.append(
        create_training(
            db_path=test_dataset_path,
            train_query=train_query,
            val_query=val_query,
            label_col='label',
            dims="100, 2, 1",
            experiment_name=experiment_name
        )
    )

    subprocess.run([
        'python',
        f'{str(repo_path)}/cli.py',
        'mlp',
        'run-training',
        '--run-ids', ','.join(run_ids),
        '--experiment-name', experiment_name
    ])


def test_mlp_kfold_training(test_dataset_path: Path):
    experiment_name = 'test_mlp_kfold_training'

    run_id = create_kfold(
        db_path=test_dataset_path,
        ring_col='cl_rings',
        table_name='data',
        dims="100, 2, 1",
        best_metric='val.max_sp',
        best_metric_mode='max',
        label_col='label',
        fold_col='fold',
        folds=5,
        inits=1,
        experiment_name=experiment_name,
        max_epochs=2,
        # n_jobs=2
    )

    run_kfold(
        run_id=run_id,
        experiment_name=experiment_name,
    )


def test_mlp_create_kfold_cli(test_dataset_path: Path, repo_path: Path):
    experiment_name = 'test_mlp_create_kfold_cli'

    result = subprocess.run(['python',
                             f'{str(repo_path)}/cli.py',
                             'mlp',
                             'create-kfold',
                             '--db-path', str(test_dataset_path),
                             '--ring-col', 'cl_rings',
                             '--dims', "100, 2, 1",
                             '--best-metric', 'val.max_sp',
                             '--best-metric-mode', 'max',
                             '--folds', '5',
                             '--inits', '1',
                             '--experiment-name', experiment_name,
                             '--max-epochs', '2'],
                            capture_output=True, text=True)
    print("STDOUT: %s", result.stdout)
    print("STDERR: %s", result.stderr)
    assert "Created K-Fold training job with run ID:" in result.stdout, "K-Fold creation failed."
