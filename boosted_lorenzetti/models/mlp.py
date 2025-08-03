from contextlib import contextmanager
from itertools import product
import mlflow.entities
import torch
import torch.nn as nn
import lightning as L
from typing import Annotated, Any, Set, Tuple, List, Literal, Dict
import typer
import mlflow
from pathlib import Path
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryROC
)
from mlflow.models import infer_signature
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import logging
import shutil
import pandas as pd
import plotly.express as px

from ..data import tensor_dataset_from_df
from ..dataset.file_dataset import FileDataset
from ..metrics import sp_index, ringer_norm1
from ..log import set_logger
from ..cross_validation import ColumnKFold
from .. import types
from ..constants import N_RINGS
from ..jobs import MLFlowLoggedJob


class MLP(L.LightningModule):
    def __init__(self,
                 dims: List[int],
                 class_weights: List[float],
                 activation: Literal['relu'] = 'relu'):
        super().__init__()
        self.save_hyperparameters()
        self.class_weights = torch.FloatTensor(class_weights)
        self.train_metrics = MetricCollection({
            'acc': BinaryAccuracy(),
            'roc': BinaryROC()
        })
        self.val_metrics = self.train_metrics.clone()
        self.example_input_array = torch.randn(5, dims[0])
        layers = []

        match activation:
            case 'relu':
                def activation(): return nn.ReLU()  # noqa: E731
            case _:
                raise ValueError(
                    f'Unsupported activation function: {activation}')

        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(activation())
        layers.pop()  # Remove the last activation
        self.model = nn.Sequential(*layers)
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
        self.example_input_array = torch.Tensor(dims[0])

    def forward(self, x):
        return self.model(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def get_metrics(self, collection_dict: Dict[str, Any]
                    ) -> Tuple[float, float, float, float, float]:
        """
        Extracts metrics from the collection dictionary.
        """
        fpr, tpr, thresh = collection_dict['roc']
        sp = sp_index(
            tpr,
            fpr,
            backend='torch'
        )
        max_sp_idx = torch.argmax(sp)
        auc = torch.trapezoid(tpr, fpr)
        return (
            collection_dict['acc'],
            sp[max_sp_idx],
            auc,
            fpr[max_sp_idx],
            tpr[max_sp_idx],
            thresh[max_sp_idx]
        )

    def get_loss(self, logits: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        weights = torch.empty(
            y.shape, dtype=torch.float32, device=self.device)
        weights[y == 1] = self.class_weights[1]
        weights[y == 0] = self.class_weights[0]
        return torch.mean(weights*self.loss_func(logits, y))

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: torch.Tensor):
        x, y = batch
        logits = self(x)
        loss = self.get_loss(logits, y.float())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        prob = torch.sigmoid(logits)
        batch_values = self.train_metrics(prob, y)
        acc, max_sp, roc_auc, max_sp_fpr, max_sp_tpr, _ = self.get_metrics(
            batch_values)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        self.log("train_max_sp", max_sp, on_epoch=True, prog_bar=True)
        self.log("train_roc_auc", roc_auc, on_epoch=True, prog_bar=True)
        self.log("train_max_sp_fpr", max_sp_fpr, on_epoch=True)
        self.log("train_max_sp_tpr", max_sp_tpr, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # Reset metrics after each epoch
        self.train_metrics.reset()

    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: torch.Tensor):
        x, y = batch
        logits = self(x)
        loss = self.get_loss(logits, y.float())
        prob = torch.sigmoid(logits)
        self.val_metrics.update(prob, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        metric_values = self.val_metrics.compute()
        acc, max_sp, roc_auc, max_sp_fpr, max_sp_tpr, max_sp_thresh = self.get_metrics(
            metric_values)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log("val_max_sp", max_sp, on_epoch=True, prog_bar=True)
        self.log("val_roc_auc", roc_auc, on_epoch=True, prog_bar=True)
        self.log("val_max_sp_fpr", max_sp_fpr, on_epoch=True)
        self.log("val_max_sp_tpr", max_sp_tpr, on_epoch=True)
        self.log("val_max_sp_thresh", max_sp_thresh, on_epoch=True)
        # Reset metrics after logging
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class TrainingJob(MLFlowLoggedJob):
    dataset_path: Path
    dims: types.DimsType
    activation: types.ActivationType = 'relu'
    df_name: types.DfNameType = 'data'
    batch_size: types.BatchSizeType = 32
    feature_cols: types.FeatureColsType = [f'ring_{i}' for i in range(N_RINGS)]
    label_cols: types.LabelColsType = ['label']
    init: types.InitType = 0
    fold: types.FoldType = 0
    fold_col: types.FoldColType = 'fold'
    run_id: types.MLFlowRunId = None
    name: types.JobNameType = 'MLP Training Job'
    accelerator: types.AcceleratorType = 'cpu'
    patience: types.PatienceType = 10
    checkpoints_dir: types.CheckpointsDirType = Path('checkpoints/')
    completed: bool = False
    max_epochs: types.MaxEpochsType = 3

    def model_post_init(self, context):
        super().model_post_init(context)

        if self.dims[0] != len(self.feature_cols):
            raise ValueError(
                f'Input dimension {self.dims[0]} does not match feature columns {len(self.feature_cols)}')

        if self.dims[-1] != 1:
            raise ValueError(f'Output dimension {self.dims[-1]} must be 1')

    def to_mlflow(self,
                  nested: bool = False,
                  tags: List[str] = [],
                  extra_tags: Dict[str, Any] = {}) -> str:
        extra_tags['model'] = 'MLP'
        return super().to_mlflow(nested=nested,
                                 tags=tags,
                                 extra_tags=extra_tags)

    def exec(self,
             cache_dir: Path,
             experiment_name: str,
             tracking_uri: str | None = None):
        file_dataset = FileDataset(self.dataset_path)
        load_cols = self.feature_cols + self.label_cols + [self.fold_col, 'id']
        data_df = file_dataset.get_df(self.df_name,
                                      load_cols=load_cols)
        logging.info('Loaded dataset from file.')
        data_df[self.feature_cols] = ringer_norm1(
            data_df[self.feature_cols].values)
        mlflow_dataset = mlflow.data.from_pandas(
            data_df,
            source=file_dataset.get_df_path(self.df_name),
            targets='label'
        )
        cv = ColumnKFold(fold_col=self.fold_col)
        train_idx, val_idx = cv.get_fold_idx(data_df, self.fold)
        train_y = data_df.loc[train_idx, self.label_cols].values.flatten()
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0., 1.]),
            y=train_y
        ).tolist()
        mlflow.log_param("class_weights", class_weights)
        model = MLP(dims=self.dims,
                    class_weights=class_weights,
                    activation=self.activation)
        with torch.no_grad():
            model.eval()
            model_input = data_df.iloc[:5][self.feature_cols].values
            output = model(torch.from_numpy(model_input.astype(np.float32)))
            signature = infer_signature(
                model_input=model_input,
                model_output=output.numpy()  # Convert to numpy for signature
            )
        logging.info('Model initialized and example input processed.')
        train_dataset = tensor_dataset_from_df(
            data_df,
            feature_cols=self.feature_cols,
            label_cols=self.label_cols,
            idx=train_idx
        )
        val_dataset = tensor_dataset_from_df(
            data_df,
            feature_cols=self.feature_cols,
            label_cols=self.label_cols,
            idx=val_idx
        )

        del data_df  # Free memory
        datamodule = L.LightningDataModule.from_datasets(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=self.batch_size,
        )
        logging.info('Data module created from datasets.')

        logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=self.name,
            tracking_uri=tracking_uri,
            run_id=self.run_id
        )

        checkpoint = ModelCheckpoint(
            monitor="val_max_sp",  # Monitor a validation metric
            dirpath=self.checkpoints_dir,  # Directory to save checkpoints
            filename='best-model-{epoch:02d}-{val_max_sp:.2f}',
            save_top_k=3,
            mode="min",  # Save based on minimum validation loss,
            save_on_train_epoch_end=False
        )
        callbacks = [
            EarlyStopping(
                monitor="val_max_sp",
                patience=self.patience,
                mode="min"
            ),
            checkpoint,
        ]

        mlflow.log_input(
            mlflow_dataset,
            context='training'
        )

        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=1,
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=False,
        )
        logging.info('Starting training process...')
        trainer.fit(model, datamodule=datamodule)
        logging.info('Training completed.')

        best_model = MLP.load_from_checkpoint(
            checkpoint.best_model_path
        )
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            name="ml_model",
            signature=signature,
        )
        onnx_path = cache_dir / 'mlp_model.onnx'
        best_model.to_onnx(onnx_path, export_params=True)
        mlflow.log_artifact(str(onnx_path), artifact_path='onnx_model')
        self.completed = True
        mlflow.log_param("completed", self.completed)
        logging.info('Training completed and model logged to MLFlow.')
        shutil.rmtree(str(self.checkpoints_dir))

    def as_metric_dict(self) -> Dict[str, Any]:
        """
        Converts the metrics of the training job to a dictionary format.
        """
        return {
            'run_id': self.run_id,
            'init': self.init,
            'fold': self.fold,
            'completed': self.completed,
            **self.metrics
        }

    @staticmethod
    def get_metrics_df(
        jobs: List['TrainingJob']
    ) -> Tuple[pd.DataFrame, Set[str]]:
        """
        Collects metrics from a list of TrainingJob instances and returns a DataFrame.
        """
        metric_names = set()
        data = list()
        for job in jobs:
            for metric_name in job.metrics.keys():
                metric_names.add(metric_name)
            job_data = job.as_metric_dict()
            data.append(job_data)
        return pd.DataFrame.from_records(data), metric_names


app = typer.Typer(
    name='mlp',
    help='Utility for training MLP models on electron classification data.'
)


@app.command(
    help='Create a training run for an MLP model.'
)
def create_training(
    dataset_path: Path,
    dims: types.DimsType,
    checkpoints_dir: types.CheckpointsDirType = TrainingJob.model_fields[
        'checkpoints_dir'].default,
    activation: types.ActivationType = TrainingJob.model_fields['activation'].default,
    df_name: types.DfNameType = TrainingJob.model_fields['df_name'].default,
    batch_size: types.BatchSizeType = TrainingJob.model_fields['batch_size'].default,
    feature_cols: types.FeatureColsType = TrainingJob.model_fields['feature_cols'].default,
    label_cols: types.LabelColsType = TrainingJob.model_fields['label_cols'].default,
    init: types.InitType = TrainingJob.model_fields['init'].default,
    fold: types.FoldType = TrainingJob.model_fields['fold'].default,
    fold_col: types.FoldColType = TrainingJob.model_fields['fold_col'].default,
    accelerator: types.AcceleratorType = TrainingJob.model_fields['accelerator'].default,
    patience: types.PatienceType = TrainingJob.model_fields['patience'].default,
    name: types.JobNameType = TrainingJob.model_fields['name'].default,
    max_epochs: types.MaxEpochsType = TrainingJob.model_fields['max_epochs'].default,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
) -> str:

    set_logger()
    logging.debug('Setting MLFlow tracking URI and experiment name.')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    job = TrainingJob(
        dataset_path=dataset_path,
        dims=dims,
        activation=activation,
        df_name=df_name,
        batch_size=batch_size,
        feature_cols=feature_cols,
        label_cols=label_cols,
        init=init,
        fold=fold,
        fold_col=fold_col,
        name=name,
        accelerator=accelerator,
        patience=patience,
        checkpoints_dir=checkpoints_dir,
        max_epochs=max_epochs,
    )

    run_id = job.to_mlflow()
    logging.info(f'Created training job with run ID: {run_id}')

    return run_id


@app.command(
    help='Train an MLP model on ingested data.'
)
def run_training(
    run_ids: List[str],
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
    set_logger()
    logging.debug(
        f'Tracking URI: {tracking_uri}, Experiment Name: {experiment_name}')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if isinstance(run_ids, str):
        run_ids = [run_ids]

    for run_id in run_ids:
        logging.info(f'Running training job with run ID: {run_id}')
        job = TrainingJob.from_mlflow(run_id)
        job.execute(experiment_name=experiment_name,
                    tracking_uri=tracking_uri)


class KFoldTrainingJob(MLFlowLoggedJob):
    dataset_path: Path
    dims: types.DimsType
    activation: types.ActivationType = TrainingJob.model_fields['activation'].default
    df_name: types.DfNameType = TrainingJob.model_fields['df_name'].default
    batch_size: types.BatchSizeType = TrainingJob.model_fields['batch_size'].default
    feature_cols: types.FeatureColsType = TrainingJob.model_fields['feature_cols'].default
    label_cols: types.LabelColsType = TrainingJob.model_fields['label_cols'].default
    inits: types.InitsType = 5
    fold_col: types.FoldColType = TrainingJob.model_fields['fold_col'].default
    folds: types.FoldType = 5
    run_id: types.MLFlowRunId = None
    name: types.JobNameType = 'MLP K-Fold Training Job'
    accelerator: types.AcceleratorType = TrainingJob.model_fields['accelerator'].default
    patience: types.PatienceType = TrainingJob.model_fields['patience'].default
    checkpoints_dir: types.CheckpointsDirType = TrainingJob.model_fields[
        'checkpoints_dir'].default
    max_epochs: types.MaxEpochsType = TrainingJob.model_fields['max_epochs'].default
    completed: bool = False
    best_metric: types.BestMetricType
    best_metric_mode: types.BestMetricModeType

    def model_post_init(self, context):
        super().model_post_init(context)

        if self.dims[0] != len(self.feature_cols):
            raise ValueError(
                f'Input dimension {self.dims[0]} does not match feature columns {len(self.feature_cols)}')

        if self.dims[-1] != 1:
            raise ValueError(f'Output dimension {self.dims[-1]} must be 1')

    @contextmanager
    def to_mlflow_context(self,
                          nested: bool = False,
                          tags: List[str] = [],
                          extra_tags: Dict[str, Any] = {}):
        extra_tags['model'] = 'MLP'
        with super().to_mlflow_context(nested=nested, tags=tags, extra_tags=extra_tags) as run:
            logging.info(
                f'Creating K-Fold training job with run ID: {run.info.run_id}')
            for init, fold in product(range(self.inits), range(self.folds)):
                training_job = TrainingJob(
                    dataset_path=self.dataset_path,
                    dims=self.dims,
                    activation=self.activation,
                    df_name=self.df_name,
                    batch_size=self.batch_size,
                    feature_cols=self.feature_cols,
                    label_cols=self.label_cols,
                    init=init,
                    fold=fold,
                    fold_col=self.fold_col,
                    name=f'{self.name} init {init} fold {fold}',
                    accelerator=self.accelerator,
                    patience=self.patience,
                    checkpoints_dir=self.checkpoints_dir /
                    f'fold_{fold}_init_{init}',
                    max_epochs=self.max_epochs
                )
                run_id = training_job.to_mlflow(
                    nested=True, tags=tags, extra_tags=extra_tags)
                logging.info(
                    f'Created child training job with run ID: {run_id}')
            logging.info(f'K-Fold training job with run ID: {run.info.run_id} completed.')
            yield run

    # Unable to use mlflow.entities.run.Run because it is not compatible with pydantic
    def get_children(self, experiment_name: str, tracking_uri: str | None = None) -> List[Any]:
        client = mlflow.MlflowClient(
            tracking_uri=tracking_uri
        )
        children = client.search_runs(
            experiment_ids=[client.get_experiment_by_name(
                experiment_name).experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{self.run_id}'",
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=1000
        )
        return children

    def get_children_jobs(self, experiment_name: str, tracking_uri: str | None = None) -> List[TrainingJob]:
        """
        Retrieves child training jobs from MLFlow based on the parent run ID.
        """
        children = self.get_children(experiment_name, tracking_uri)

        children_jobs = []
        for child in children:
            child_run_id = child.info.run_id
            child_job = TrainingJob.from_mlflow(child_run_id)
            logging.info(
                f'Found child training job with: run ID - {child_run_id} - fold - {child_job.fold} - init - {child_job.init}')
            children_jobs.append(child_job)

        return children_jobs

    def exec(self, cache_dir: Path,
             experiment_name: str,
             tracking_uri: str | None,
             force: Literal['all', 'error'] | None = None):
        children = self.get_children(experiment_name, tracking_uri)
        if not children:
            raise RuntimeError('No child training jobs found.')

        children_jobs = []

        for child in children:
            child_run_id = child.info.run_id
            child_job = TrainingJob.from_mlflow(child_run_id)
            logging.info(
                f'Running child training job with: run ID - {child_run_id} - fold - {child_job.fold} - init - {child_job.init}')
            child_job.execute(experiment_name=experiment_name,
                              tracking_uri=tracking_uri,
                              nested=True,
                              force=force)
            children_jobs.append(child_job)

        children_metrics_df, metric_names = TrainingJob.get_metrics_df(
            children_jobs)
        if self.best_metric_mode == 'min':
            best_idx = children_metrics_df[self.best_metric].idxmin()
        elif self.best_metric_mode == 'max':
            best_idx = children_metrics_df[self.best_metric].idxmax()
        else:
            raise ValueError(
                f'Unsupported best metric mode: {self.best_metric_mode}')
        best_metrics = children_metrics_df.loc[best_idx]
        mlflow.log_param("best_job_run_id", best_metrics['run_id'])
        mlflow.log_param("best_job_fold", best_metrics['fold'])
        mlflow.log_param("best_job_init", best_metrics['init'])

        children_metrics_df_path = cache_dir / 'job_metrics.csv'
        children_metrics_df.to_csv(
            children_metrics_df_path, index=False)
        mlflow.log_artifact(str(children_metrics_df_path))

        aggregation_dict = dict()
        for metric in metric_names:
            aggregation_dict[f'{metric}_mean'] = pd.NamedAgg(
                column=metric, aggfunc='mean')
            aggregation_dict[f'{metric}_std'] = pd.NamedAgg(
                column=metric, aggfunc='std')
            aggregation_dict[f'{metric}_min'] = pd.NamedAgg(
                column=metric, aggfunc='min')
            aggregation_dict[f'{metric}_max'] = pd.NamedAgg(
                column=metric, aggfunc='max')
            aggregation_dict[f'{metric}_median'] = pd.NamedAgg(
                column=metric, aggfunc='median')
        aggregated_metrics = children_metrics_df \
            .groupby('fold').agg(**aggregation_dict) \
            .reset_index() \
            .melt(id_vars='fold',
                  var_name='metric')
        aggregated_metrics_path = cache_dir / 'aggregated_metrics.csv'
        aggregated_metrics.to_csv(aggregated_metrics_path, index=False)
        mlflow.log_artifact(str(aggregated_metrics_path))

        best_metric_label = self.best_metric.replace('_', ' ').capitalize()
        fig = px.box(children_metrics_df, x="fold", y=self.best_metric)
        fig.update_layout(
            title=f'K-Fold {best_metric_label} Distribution',
            xaxis_title='Fold',
            yaxis_title=best_metric_label
        )
        fig.add_hline(y=best_metrics[self.best_metric],
                      line_dash="dash",
                      line_color="red",
                      annotation_text='Best',
                      annotation_position="top left")
        mlflow.log_figure(fig, f'{self.best_metric}_box_plot.html')

        self.completed = True
        mlflow.log_param("completed", True)
        logging.info('K-Fold training jobs completed and logged to MLFlow.')


DatasetPathType = Annotated[
    Path,
    typer.Argument(
        help='Path to the dataset file.'
    )
]


@app.command(
    help='Create a K-Fold training run for an MLP model.'
)
def create_kfold(
    dataset_path: DatasetPathType,
    dims: types.DimsType,
    best_metric: types.BestMetricType,
    best_metric_mode: types.BestMetricModeType,
    checkpoints_dir: types.CheckpointsDirType = KFoldTrainingJob.model_fields[
        'checkpoints_dir'].default,
    activation: types.ActivationType = KFoldTrainingJob.model_fields['activation'].default,
    df_name: types.DfNameType = KFoldTrainingJob.model_fields['df_name'].default,
    batch_size: types.BatchSizeType = KFoldTrainingJob.model_fields['batch_size'].default,
    feature_cols: types.FeatureColsType = KFoldTrainingJob.model_fields['feature_cols'].default,
    label_cols: types.LabelColsType = KFoldTrainingJob.model_fields['label_cols'].default,
    inits: types.InitsType = KFoldTrainingJob.model_fields['inits'].default,
    fold_col: types.FoldColType = KFoldTrainingJob.model_fields['fold_col'].default,
    folds: types.FoldType = KFoldTrainingJob.model_fields['folds'].default,
    accelerator: types.AcceleratorType = KFoldTrainingJob.model_fields['accelerator'].default,
    patience: types.PatienceType = KFoldTrainingJob.model_fields['patience'].default,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
    max_epochs: types.MaxEpochsType = KFoldTrainingJob.model_fields['max_epochs'].default,
    name: types.JobNameType = KFoldTrainingJob.model_fields['name'].default,
) -> str:

    set_logger()
    logging.debug('Setting MLFlow tracking URI and experiment name.')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    job = KFoldTrainingJob(
        dataset_path=dataset_path,
        dims=dims,
        activation=activation,
        df_name=df_name,
        batch_size=batch_size,
        feature_cols=feature_cols,
        label_cols=label_cols,
        inits=inits,
        fold_col=fold_col,
        folds=folds,
        name=name,
        accelerator=accelerator,
        patience=patience,
        checkpoints_dir=checkpoints_dir,
        max_epochs=max_epochs,
        best_metric=best_metric,
        best_metric_mode=best_metric_mode
    )

    run_id = job.to_mlflow()
    logging.info(f'Created K-Fold training job with run ID: {run_id}')

    return run_id


@app.command(
    help='Train an MLP model using K-Fold cross-validation.'
)
def run_kfold(
    run_id: str,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
    force: str | None = None
):
    set_logger()
    logging.info(f'Running K-Fold training job with run ID: {run_id}')
    logging.debug(
        f'Tracking URI: {tracking_uri}, Experiment Name: {experiment_name}')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    job = KFoldTrainingJob.from_mlflow(run_id)

    job.execute(experiment_name=experiment_name,
                tracking_uri=tracking_uri)
