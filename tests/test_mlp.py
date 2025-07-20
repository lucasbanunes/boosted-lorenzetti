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
        run_id=run_id,
        experiment_name=experiment_name,
    )
