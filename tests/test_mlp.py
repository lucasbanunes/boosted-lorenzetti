from boosted_lorenzetti.models import mlp
from pathlib import Path


def test_run_training(test_dataset_path: Path,
                      tmp_path: Path):
    mlp.run_training(
        dataset_path=test_dataset_path,
        dims=[2, 1],
        batch_size=2,
        seed=42,
        feature_cols=['ring_0', 'ring_1'],
        label_cols=['label'],
        init=0,
        fold=0,
        tracking_uri=None,
        experiment_name='test_experiment',
        run_id=None,
        accelerator='cpu',
        checkpoints_dir=tmp_path / 'checkpoints',
    )


def test_full_training(test_dataset_path: Path,
                       tmp_path: Path):
    experiment_name = 'test_experiment'

    run_id = mlp.create_training(
        dataset_path=test_dataset_path,
        dims=[2, 1],
        batch_size=2,
        seed=42,
        feature_cols=['ring_0', 'ring_1'],
        label_cols=['label'],
        init=0,
        fold=0,
        experiment_name=experiment_name,
        run_id=None,
        run_name='MLP Training',
        accelerator='cpu',
        checkpoints_dir=tmp_path / 'checkpoints'
    )

    mlp.run_training(
        run_id=run_id,
        experiment_name=experiment_name,
    )
