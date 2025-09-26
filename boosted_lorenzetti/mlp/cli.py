import typer
import logging
from typing import Annotated
from pathlib import Path
import mlflow
import duckdb
import torch
import polars as pl

from .. import types
from .jobs import KFoldTrainingJob, TrainingJob
from ..constants import N_RINGS

app = typer.Typer(
    name='mlp',
    help='Utility for training MLP models on electron classification data.'
)


@app.command(
    help='Create a training run for an MLP model.'
)
def create_training(
    db_path: Path,
    train_query: str,
    dims: types.DimsOptionType,
    val_query: str | None = None,
    test_query: str | None = None,
    predict_query: str | None = None,
    label_col: str | None = TrainingJob.model_fields['label_col'].default,
    activation: types.ActivationType = TrainingJob.model_fields['activation'].default,
    batch_size: types.BatchSizeType = TrainingJob.model_fields['batch_size'].default,
    name: str = TrainingJob.model_fields['name'].default,
    accelerator: types.AcceleratorType = TrainingJob.model_fields['accelerator'].default,
    patience: types.PatienceType = TrainingJob.model_fields['patience'].default,
    checkpoints_dir: types.CheckpointsDirType = TrainingJob.model_fields[
        'checkpoints_dir'].default,
    max_epochs: types.MaxEpochsType = TrainingJob.model_fields['max_epochs'].default,
    monitor: types.MonitorOptionField = TrainingJob.model_fields['monitor'].default,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
) -> str:

    logging.debug('Setting MLFlow tracking URI and experiment name.')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if isinstance(dims, str):
        dims = [int(dim.strip()) for dim in dims.split(',')]

    job = TrainingJob(
        db_path=db_path,
        train_query=train_query,
        dims=dims,
        val_query=val_query,
        test_query=test_query,
        predict_query=predict_query,
        label_col=label_col,
        activation=activation,
        batch_size=batch_size,
        name=name,
        accelerator=accelerator,
        patience=patience,
        checkpoints_dir=checkpoints_dir,
        max_epochs=max_epochs,
        monitor=monitor
    )

    run_id = job.to_mlflow()
    logging.info(f'Created training job with run ID: {run_id}')

    return run_id


@app.command(
    help='Train an MLP model on ingested data.'
)
def run_training(
    run_ids: types.RunIdsListOptionType,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
    logging.debug(
        f'Tracking URI: {tracking_uri}, Experiment Name: {experiment_name}')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if isinstance(run_ids, str):
        logging.info(f'Parsing run IDs from string: {run_ids}')
        run_ids = [id_.strip() for id_ in run_ids.split(',')]

    for run_id in run_ids:
        logging.info(f'Running training job with run ID: {run_id}')
        job = TrainingJob.from_mlflow_run_id(run_id)
        job.execute(experiment_name=experiment_name,
                    tracking_uri=tracking_uri)


@app.command(
    help='Create a K-Fold training run for an MLP model.'
)
def create_kfold(
    db_path: Annotated[Path, typer.Option(
        help='Path to the DuckDB database file.'
    )],
    ring_col: Annotated[
        str,
        typer.Option(help='Name of the column containing the ring data.')
    ],
    dims: types.DimsOptionType,
    best_metric: types.BestMetricType,
    best_metric_mode: types.BestMetricModeType,
    fold_col: str = KFoldTrainingJob.model_fields['fold_col'].default,
    label_col: str = KFoldTrainingJob.model_fields['label_col'].default,
    table_name: Annotated[str, typer.Option(
        help='Name of the DuckDB table containing the dataset.'
    )] = KFoldTrainingJob.model_fields['table_name'].default,
    inits: types.InitsType = KFoldTrainingJob.model_fields['inits'].default,
    folds: types.FoldsType = KFoldTrainingJob.model_fields['folds'].default,
    activation: types.ActivationType = KFoldTrainingJob.model_fields['activation'].default,
    batch_size: types.BatchSizeType = KFoldTrainingJob.model_fields['batch_size'].default,
    name: str = KFoldTrainingJob.model_fields['name'].default,
    accelerator: types.AcceleratorType = KFoldTrainingJob.model_fields['accelerator'].default,
    patience: types.PatienceType = KFoldTrainingJob.model_fields['patience'].default,
    checkpoints_dir: types.CheckpointsDirType = KFoldTrainingJob.model_fields[
        'checkpoints_dir'].default,
    max_epochs: types.MaxEpochsType = KFoldTrainingJob.model_fields['max_epochs'].default,
    monitor: types.MonitorOptionField = KFoldTrainingJob.model_fields['monitor'].default,
    # n_jobs: int = KFoldTrainingJob.model_fields['n_jobs'].default,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',

) -> str:

    logging.debug('Setting MLFlow tracking URI and experiment name.')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if isinstance(dims, str):
        dims = [int(dim.strip()) for dim in dims.split(',')]

    job = KFoldTrainingJob(
        db_path=db_path,
        ring_col=ring_col,
        dims=dims,
        best_metric=best_metric,
        best_metric_mode=best_metric_mode,
        fold_col=fold_col,
        label_col=label_col,
        table_name=table_name,
        inits=inits,
        folds=folds,
        activation=activation,
        batch_size=batch_size,
        name=name,
        accelerator=accelerator,
        patience=patience,
        checkpoints_dir=checkpoints_dir,
        max_epochs=max_epochs,
        monitor=monitor,
        # n_jobs=n_jobs
    )

    run_id = job.to_mlflow()
    logging.info(f'Created K-Fold training job with run ID: {run_id}')

    return run_id


@app.command(
    help='Train an MLP model using K-Fold cross-validation.'
)
def run_kfold(
    run_id: types.RunIdsListOptionType,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
    force: str | None = None
):
    logging.info(f'Running K-Fold training job with run ID: {run_id}')
    logging.debug(
        f'Tracking URI: {tracking_uri}, Experiment Name: {experiment_name}')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if isinstance(run_id, str):
        run_id = [id_.strip() for id_ in run_id.split(',')]

    for id_ in run_id:
        logging.info(f'Running K-Fold training job with run ID: {id_}')

        job = KFoldTrainingJob.from_mlflow_run_id(id_)

        job.execute(experiment_name=experiment_name,
                    tracking_uri=tracking_uri,
                    force=force)


@app.command(
    help='Infers using a trained MLP model.'
)
def inference(
    db_path: Path,
    run_id: str,
    ring_col: str,
    table_name: str,
    output_table: str | None = None,
    filter: str | None = None,
):
    logging.info(f'Infers using MLP model with run ID: {run_id}')

    job = TrainingJob.from_mlflow_run_id(run_id)
    job.model.eval()
    rings = [f'{ring_col}[{i+1}]/ring_norm as ring_{i}' for i in range(N_RINGS)]
    cols = ['id', f'ringer_l1({ring_col}) as ring_norm'] + rings

    with duckdb.connect(str(db_path), read_only=True) as conn:
        relation = conn.table(table_name)
        if filter:
            relation = relation.filter(filter)
        relation = relation.select(', '.join(cols))
        data = relation.pl()

    model_inputs = [f'ring_{i}' for i in range(N_RINGS)]
    job.model.eval()
    with torch.no_grad():
        results: torch.Tensor = job.model(data[model_inputs].to_torch(dtype=pl.Float32))
    threshold = job.metrics['train']['max_sp_thresh']
    data = data.select('id') \
        .with_columns(pl.Series('prediction', results.cpu().numpy().squeeze())) \
        .with_columns(pl.Series('proba', torch.sigmoid(results).cpu().numpy().squeeze())) \
        .with_columns(pl.when(pl.col('proba') >= threshold).then(1).otherwise(0).alias('predicted_label'))

    if output_table is None:
        return data

    raise NotImplementedError('Saving to database not implemented yet.')
