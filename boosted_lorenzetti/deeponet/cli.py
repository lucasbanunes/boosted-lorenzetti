from typing import Annotated
import mlflow
import logging
import typer

from .. import types
from . import jobs as deeponet_jobs


app = typer.Typer(
    name='deeponet',
    help='Utility for training DeepONet models on electron classification data.'
)


BranchDimsOption = Annotated[
    str,
    typer.Option(
        help=deeponet_jobs.BRANCH_DIMS_OPTION_FIELD_HELP
    )
]

BranchActivationsOption = Annotated[
    str,
    typer.Option(
        help=deeponet_jobs.BRANCH_ACTIVATIONS_OPTION_FIELD_HELP
    ),
]

TrunkDimsOption = Annotated[
    str,
    typer.Option(
        help=deeponet_jobs.TRUNK_DIMS_OPTION_FIELD_HELP
    )
]

TrunkActivationsOption = Annotated[
    str,
    typer.Option(
        help=deeponet_jobs.TRUNK_ACTIVATIONS_OPTION_FIELD_HELP
    )
]


mlp_app = typer.Typer(
    name='mlp',
    help='Utilities for training MLP based deeponets models on electron classification data.'
)


def parse_activations(activations: str) -> list[str | None]:
    activations_list = []
    for a in activations.split(','):
        a = a.strip()
        if a.lower() == 'none':
            activations_list.append(None)
        else:
            activations_list.append(a)
    return activations_list


@mlp_app.command(
    help='Create a training run for a DeepONet model'
)
def create_training(
    db_path: deeponet_jobs.DbPathOptionField,
    table_name: deeponet_jobs.TableNameOptionField,
    ring_col: deeponet_jobs.RingColOptionField,
    et_col: deeponet_jobs.EtColOptionField,
    eta_col: deeponet_jobs.EtaColOptionField,
    pileup_col: deeponet_jobs.PileupColOptionField,
    fold: deeponet_jobs.FoldOptionField,
    branch_dims: BranchDimsOption,
    branch_activations: BranchActivationsOption,
    trunk_dims: TrunkDimsOption,
    trunk_activations: TrunkActivationsOption,
    fold_col: deeponet_jobs.FoldColOptionField = deeponet_jobs.MLPUnstackedDeepONetTrainingJob.model_fields['fold_col'].default,
    label_col: deeponet_jobs.LabelColOptionField = deeponet_jobs.MLPUnstackedDeepONetTrainingJob.model_fields['label_col'].default,
    learning_rate: deeponet_jobs.LearningRateOptionField = deeponet_jobs.MLPUnstackedDeepONetTrainingJob.model_fields['learning_rate'].default,
    batch_size: types.BatchSizeType = deeponet_jobs.MLPUnstackedDeepONetTrainingJob.model_fields['batch_size'].default,
    accelerator: types.AcceleratorType = deeponet_jobs.MLPUnstackedDeepONetTrainingJob.model_fields['accelerator'].default,
    patience: types.PatienceType = deeponet_jobs.MLPUnstackedDeepONetTrainingJob.model_fields['patience'].default,
    checkpoints_dir: types.CheckpointsDirType = deeponet_jobs.MLPUnstackedDeepONetTrainingJob.model_fields['checkpoints_dir'].default,
    max_epochs: types.MaxEpochsType = deeponet_jobs.MLPUnstackedDeepONetTrainingJob.model_fields['max_epochs'].default,
    monitor: types.MonitorOptionField = deeponet_jobs.MLPUnstackedDeepONetTrainingJob.model_fields['monitor'].default,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
    if isinstance(branch_dims, str):
        branch_dims = [int(d.strip()) for d in branch_dims.split(',')]
    if isinstance(trunk_dims, str):
        trunk_dims = [int(d.strip()) for d in trunk_dims.split(',')]
    if isinstance(branch_activations, str):
        branch_activations = parse_activations(branch_activations)
    if isinstance(trunk_activations, str):
        trunk_activations = parse_activations(trunk_activations)

    logging.debug('Setting MLFlow tracking URI and experiment name.')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    job = deeponet_jobs.MLPUnstackedDeepONetTrainingJob(
        db_path=db_path,
        table_name=table_name,
        ring_col=ring_col,
        et_col=et_col,
        eta_col=eta_col,
        pileup_col=pileup_col,
        fold=fold,
        branch_dims=branch_dims,
        branch_activations=branch_activations,
        trunk_dims=trunk_dims,
        trunk_activations=trunk_activations,
        fold_col=fold_col,
        label_col=label_col,
        learning_rate=learning_rate,
        batch_size=batch_size,
        accelerator=accelerator,
        patience=patience,
        checkpoints_dir=checkpoints_dir,
        max_epochs=max_epochs,
        monitor=monitor,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )
    run_id = job.to_mlflow()
    logging.info(f'Created training job with run ID: {run_id}')

    return run_id


@mlp_app.command(
    help='Run a training job given its MLFlow run ID'
)
def run_training(
    run_ids: types.RunIdsListOptionType,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
    logging.debug(
        f'Tracking URI: {tracking_uri}, Experiment Name: {experiment_name}')
    if isinstance(run_ids, str):
        run_ids = [run_id.strip() for run_id in run_ids.split(',')]

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if isinstance(run_ids, str):
        run_ids = [run_ids]

    for run_id in run_ids:
        logging.info(f'Running training job with run ID: {run_id}')
        job = deeponet_jobs.MLPUnstackedDeepONetTrainingJob.from_mlflow_run_id(run_id)
        job.execute(experiment_name=experiment_name,
                    tracking_uri=tracking_uri)


app.add_typer(mlp_app)
