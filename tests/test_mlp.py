import logging
from boosted_lorenzetti.models import mlp
from pathlib import Path

from boosted_lorenzetti.constants import N_RINGS


def test_full_training(test_dataset_path: Path):
    experiment_name = 'test_experiment'

    run_id = mlp.create_training(
        dataset_path=test_dataset_path,
        dims=[N_RINGS, 1],
        experiment_name=experiment_name
    )

    mlp.run_training(
        run_ids=run_id,
        experiment_name=experiment_name,
    )


def test_multiple_trainings(test_dataset_path: Path):
    experiment_name = 'test_experiment'

    run_ids = []
    run_ids.append(
        mlp.create_training(
            dataset_path=test_dataset_path,
            dims=[N_RINGS, 1],
            experiment_name=experiment_name
        )
    )
    run_ids.append(
        mlp.create_training(
            dataset_path=test_dataset_path,
            dims=[N_RINGS, 1],
            experiment_name=experiment_name
        )
    )

    mlp.run_training(
        run_ids=run_ids,
        experiment_name=experiment_name,
    )


def test_kfold_training(test_dataset_path: Path):
    experiment_name = 'test_experiment_kfold'

    run_id = mlp.create_kfold(
        dataset_path=test_dataset_path,
        dims=[N_RINGS, 1],
        folds=5,
        experiment_name=experiment_name,
        max_epochs=2,
        best_metric='val_max_sp',
        best_metric_mode='max'
    )

    mlp.run_kfold(
        run_id=run_id,
        experiment_name=experiment_name,
    )
