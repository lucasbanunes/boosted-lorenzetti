from boosted_lorenzetti.deeponet import cli
from pathlib import Path


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
        branch_dims='100, 2, 1',
        branch_activations='relu, None',
        trunk_dims='3, 2, 1',
        trunk_activations='relu, None',
        experiment_name=experiment_name
    )

    cli.run_training(
        run_ids=[run_id],
        experiment_name=experiment_name,
    )
