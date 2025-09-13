import subprocess
from boosted_lorenzetti.deeponet import cli
from pathlib import Path
import logging


def test_deeponet_training_job(test_dataset_path: Path):
    experiment_name = 'test_deeponet_training_job'

    run_id = cli.create_training(
        db_path=test_dataset_path,
        table_name='data',
        ring_col='cl_rings',
        et_col='cl_et',
        eta_col='cl_eta',
        pileup_col='avgmu',
        fold=0,
        branch_dims='100, 2',
        branch_activations='relu',
        trunk_dims='3, 1',
        trunk_activations='relu',
        experiment_name=experiment_name
    )

    cli.run_training(
        run_ids=run_id,
        experiment_name=experiment_name,
    )


def test_deeponet_multiple_training_jobs(test_dataset_path: Path):
    experiment_name = 'test_deeponet_multiple_training_jobs'
    run_ids = []
    for _ in range(2):
        run_id = cli.create_training(
            db_path=test_dataset_path,
            table_name='data',
            ring_col='cl_rings',
            et_col='cl_et',
            eta_col='cl_eta',
            pileup_col='avgmu',
            fold=0,
            branch_dims='100, 2',
            branch_activations='relu',
            trunk_dims='3, 1',
            trunk_activations='relu',
            experiment_name=experiment_name
        )
        run_ids.append(run_id)

    cli.run_training(
        run_ids=', '.join(run_ids),
        experiment_name=experiment_name,
    )


def test_deeponet_kfold_training(test_dataset_path: Path):
    experiment_name = 'test_deeponet_kfold_training'

    run_id = cli.create_kfold(
        db_path=test_dataset_path,
        table_name='data',
        ring_col='cl_rings',
        et_col='cl_et',
        eta_col='cl_eta',
        pileup_col='avgmu',
        branch_dims='100, 2',
        branch_activations='relu',
        trunk_dims='3, 1',
        trunk_activations='relu',
        folds=5,
        inits=1,
        best_metric='val.max_sp',
        best_metric_mode='max',
        experiment_name=experiment_name
        # n_jobs=2
    )

    cli.run_kfold(
        run_ids=run_id,
        experiment_name=experiment_name,
    )


def test_deeponet_create_kfold_cli(test_dataset_path: Path, repo_path: Path):
    experiment_name = 'test_deeponet_create_kfold_cli'

    result = subprocess.run([
        'python',
        f'{str(repo_path)}/cli.py',
        'deeponet',
        'mlp',
        'create-kfold',
        '--db-path', str(test_dataset_path),
        '--table-name', 'data',
        '--ring-col', 'cl_rings',
        '--et-col', 'cl_et',
        '--eta-col', 'cl_eta',
        '--pileup-col', 'avgmu',
        '--branch-dims', '100, 2',
        '--branch-activations', 'relu',
        '--trunk-dims', '3, 1',
        '--trunk-activations', 'relu',
        '--folds', '5',
        '--inits', '1',
        '--best-metric', 'val.max_sp',
        '--best-metric-mode', 'max',
        '--experiment-name', experiment_name
    ], capture_output=True, text=True)

    logging.info("STDOUT: %s", result.stdout)
    logging.error("STDERR: %s", result.stderr)

    assert "Created K-Fold training job with run ID:" in result.stdout, "K-Fold creation failed."


def test_deeponet_run_kfold_cli(test_dataset_path: Path, repo_path: Path):
    experiment_name = 'test_deeponet_run_kfold_cli'

    run_id = cli.create_kfold(
        db_path=test_dataset_path,
        table_name='data',
        ring_col='cl_rings',
        et_col='cl_et',
        eta_col='cl_eta',
        pileup_col='avgmu',
        branch_dims='100, 2',
        branch_activations='relu',
        trunk_dims='3, 1',
        trunk_activations='relu',
        folds=5,
        inits=1,
        best_metric='val.max_sp',
        best_metric_mode='max',
        experiment_name=experiment_name
        # n_jobs=2
    )

    result = subprocess.run([
        'python',
        f'{str(repo_path)}/cli.py',
        'deeponet',
        'mlp',
        'run-kfold',
        '--run-ids', run_id,
        '--experiment-name', experiment_name
    ], capture_output=True, text=True)

    logging.info("STDOUT: %s", result.stdout)
    logging.error("STDERR: %s", result.stderr)

    assert "K-Fold training jobs completed and logged to MLFlow." in result.stdout, "K-Fold creation failed."
