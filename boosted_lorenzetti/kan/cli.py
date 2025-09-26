import typer
import logging
import mlflow

from .. import types
from .jobs import KFoldTrainingJob, TrainingJob

app = typer.Typer(
    name='kan',
    help='Utility for training KAN models on electron classification data.'
)


@app.command(
    help='Create a training run for a KAN model.'
)
def create_training(
    db_path: types.DbPathOptionField,
    dims: types.DimsOptionType,
    train_query: str,
    grid: int = 5,
    k: int = 3,
    val_query: str | None = None,
    test_query: str | None = None,
    predict_query: str | None = None,
    label_col: str | None = TrainingJob.model_fields['label_col'].default,
    batch_size: types.BatchSizeType = TrainingJob.model_fields['batch_size'].default,
    name: str = TrainingJob.model_fields['name'].default,
    patience: types.PatienceType = TrainingJob.model_fields['patience'].default,
    checkpoints_dir: types.CheckpointsDirType = TrainingJob.model_fields['checkpoints_dir'].default,
    max_epochs: types.MaxEpochsType = TrainingJob.model_fields['max_epochs'].default,
    monitor: types.MonitorOptionField = TrainingJob.model_fields['monitor'].default,
    learning_rate: types.LearningRateOptionField = TrainingJob.model_fields['learning_rate'].default,
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
        grid=grid,
        k=k,
        val_query=val_query,
        test_query=test_query,
        predict_query=predict_query,
        label_col=label_col,
        batch_size=batch_size,
        name=name,
        patience=patience,
        checkpoints_dir=checkpoints_dir,
        max_epochs=max_epochs,
        monitor=monitor,
        learning_rate=learning_rate
    )

    run_id = job.to_mlflow()
    logging.info(f'Created training job with run ID: {run_id}')

    return run_id


@app.command(
    help='Train a KAN model on ingested data.'
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
    help='Create a K-Fold training run for a KAN model.'
)
def create_kfold(
    db_path: types.DbPathOptionField,
    ring_col: types.RingColOptionField,
    dims: types.DimsOptionType,
    best_metric: types.BestMetricType,
    best_metric_mode: types.BestMetricModeType,
    grid: int = KFoldTrainingJob.model_fields['grid'].default,
    k: int = KFoldTrainingJob.model_fields['k'].default,
    fold_col: str = KFoldTrainingJob.model_fields['fold_col'].default,
    label_col: str = KFoldTrainingJob.model_fields['label_col'].default,
    table_name: types.TableNameOptionField = KFoldTrainingJob.model_fields['table_name'].default,
    inits: types.InitsType = KFoldTrainingJob.model_fields['inits'].default,
    folds: types.FoldsType = KFoldTrainingJob.model_fields['folds'].default,
    batch_size: types.BatchSizeType = KFoldTrainingJob.model_fields['batch_size'].default,
    name: str = KFoldTrainingJob.model_fields['name'].default,
    patience: types.PatienceType = KFoldTrainingJob.model_fields['patience'].default,
    checkpoints_dir: types.CheckpointsDirType = KFoldTrainingJob.model_fields[
        'checkpoints_dir'].default,
    max_epochs: types.MaxEpochsType = KFoldTrainingJob.model_fields['max_epochs'].default,
    monitor: types.MonitorOptionField = KFoldTrainingJob.model_fields['monitor'].default,
    learning_rate: types.LearningRateOptionField = KFoldTrainingJob.model_fields['learning_rate'].default,
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
        grid=grid,
        k=k,
        fold_col=fold_col,
        label_col=label_col,
        table_name=table_name,
        inits=inits,
        folds=folds,
        batch_size=batch_size,
        name=name,
        patience=patience,
        checkpoints_dir=checkpoints_dir,
        max_epochs=max_epochs,
        monitor=monitor,
        learning_rate=learning_rate
        # n_jobs=n_jobs
    )

    run_id = job.to_mlflow()
    logging.info(f'Created K-Fold training job with run ID: {run_id}')

    return run_id


@app.command(
    help='Train a KAN model using K-Fold cross-validation.'
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
