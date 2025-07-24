from functools import cached_property
import mlflow.entities
import torch
import torch.nn as nn
import lightning as L
from typing import Any, Tuple, List, Literal, Dict
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
# from sklearn.model_selection import StratifiedKFold
from pydantic import BaseModel, computed_field
import json
import logging
from itertools import product
import shutil

from ..data import tensor_dataset_from_df
from ..dataset.file_dataset import FileDataset
from ..metrics import sp_index, ringer_norm1
from ..log import set_logger
from ..cross_validation import ColumnKFold
from .. import types
from ..constants import N_RINGS


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
        acc, max_sp, roc_auc, max_sp_tpr, max_sp_fpr, _ = self.get_metrics(
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
        acc, max_sp, roc_auc, max_sp_tpr, max_sp_fpr, max_sp_thresh = self.get_metrics(
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


class TrainingJob(BaseModel):
    dataset_path: Path
    dims: types.DimsType
    activation: types.ActivationType = 'relu'
    df_name: types.DfNameType = 'data'
    batch_size: types.BatchSizeType = 32
    seed: types.SeedType
    feature_cols: types.FeatureColsType = [f'ring_{i}' for i in range(N_RINGS)]
    label_cols: types.LabelColsType = ['label']
    init: types.InitType = 0
    fold: types.FoldType = 0
    fold_col: types.FoldColType = 'fold'
    run_id: types.MLFlowRunId = None
    job_name: types.JobNameType = 'MLP Training Job'
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

    @computed_field
    @cached_property
    def run(self) -> mlflow.entities.run.Run:
        if self.run_id is None:
            raise ValueError("Run ID must be set before accessing the run.")
        client = mlflow.MlflowClient()
        return client.get_run(self.run_id)

    @classmethod
    def from_mlflow(cls, run_id: str) -> 'TrainingJob':
        logging.debug(f'Loading training job from MLFlow run ID: {run_id}')
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        params = run.data.params
        return cls(
            dataset_path=Path(params['dataset_path']),
            dims=json.loads(params['dims']),
            activation=params['activation'],
            df_name=params['df_name'],
            batch_size=int(params['batch_size']),
            seed=int(params['seed']),
            feature_cols=json.loads(params['feature_cols']),
            label_cols=json.loads(params['label_cols']),
            init=int(params['init']),
            fold=int(params['fold']),
            run_id=run_id,
            job_name=params['job_name'],
            accelerator=params['accelerator'],
            patience=int(params['patience']),
            checkpoints_dir=Path(params['checkpoints_dir']),
            completed=bool(params.get('completed', False)),
            max_epochs=int(params['max_epochs'])
        )

    def model_dump_mlflow(self, nested: bool = False) -> str:
        tags = {
            'init': self.init,
            'fold': self.fold,
            'model': 'MLP',
        }

        params = self.model_dump()
        params.pop('completed')
        params['dims'] = json.dumps(self.dims)
        params['feature_cols'] = json.dumps(self.feature_cols)
        params['label_cols'] = json.dumps(self.label_cols)

        with mlflow.start_run(run_name=self.job_name,
                              tags=tags,
                              nested=nested) as run:
            mlflow.log_params(params)
        self.run_id = run.info.run_id
        return self.run_id

    def _exec(self,
              experiment_name: str,
              tracking_uri: str | None = None):
        if self.completed:
            logging.info('Training job already completed.')
            return
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
        # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        # y = data_df[self.label_cols].values.flatten()
        # for i, (train_idx, val_idx) in enumerate(cv.split(y, y)):
        #     if i < self.fold:
        #         continue
        #     break  # Only process the specified fold
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
            run_name=self.job_name,
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
            checkpoint
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
            enable_progress_bar=False
        )
        logging.info('Starting training process.')
        trainer.fit(model, datamodule=datamodule)

        best_model = MLP.load_from_checkpoint(
            checkpoint.best_model_path
        )
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            artifact_path="model",
            signature=signature,
        )
        self.completed = True
        mlflow.log_param("completed", self.completed)
        logging.info('Training completed and model logged to MLFlow.')
        shutil.rmtree(str(self.checkpoints_dir))

    def exec(self,
             experiment_name: str,
             tracking_uri: str | None = None,
             nested: bool = False):
        if self.run_id is None:
            raise ValueError()
        with mlflow.start_run(self.run_id, nested=nested):
            self._exec(experiment_name, tracking_uri)


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
    seed: types.SeedType = None,
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
    job_name: types.JobNameType = TrainingJob.model_fields['job_name'].default,
    max_epochs: types.MaxEpochsType = TrainingJob.model_fields['max_epochs'].default,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
) -> str:

    set_logger()
    logging.debug('Setting MLFlow tracking URI and experiment name.')
    if seed is None:
        seed = types.seed_factory()

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
        seed=seed,
        feature_cols=feature_cols,
        label_cols=label_cols,
        init=init,
        fold=fold,
        fold_col=fold_col,
        job_name=job_name,
        accelerator=accelerator,
        patience=patience,
        checkpoints_dir=checkpoints_dir,
        max_epochs=max_epochs,
    )

    run_id = job.model_dump_mlflow()
    logging.info(f'Created training job with run ID: {run_id}')

    return run_id


@app.command(
    help='Train an MLP model on ingested data.'
)
def run_training(
    run_id: str | None = None,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
    set_logger()
    logging.info(f'Running training job with run ID: {run_id}')
    logging.debug(
        f'Tracking URI: {tracking_uri}, Experiment Name: {experiment_name}')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    job = TrainingJob.from_mlflow(run_id)

    job.exec(experiment_name=experiment_name,
             tracking_uri=tracking_uri)


class KFoldTrainingJob(BaseModel):
    dataset_path: Path
    dims: types.DimsType
    activation: types.ActivationType = TrainingJob.model_fields['activation'].default
    df_name: types.DfNameType = TrainingJob.model_fields['df_name'].default
    batch_size: types.BatchSizeType = TrainingJob.model_fields['batch_size'].default
    seed: types.SeedType = TrainingJob.model_fields['seed'].default
    feature_cols: types.FeatureColsType = TrainingJob.model_fields['feature_cols'].default
    label_cols: types.LabelColsType = TrainingJob.model_fields['label_cols'].default
    inits: types.InitsType = 5
    fold_col: types.FoldColType = TrainingJob.model_fields['fold_col'].default
    folds: types.FoldType = 5
    run_id: types.MLFlowRunId = None
    job_name: types.JobNameType = 'MLP K-Fold Training Job'
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

    @computed_field
    @cached_property
    def run(self) -> mlflow.entities.run.Run:
        if self.run_id is None:
            raise ValueError("Run ID must be set before accessing the run.")
        client = mlflow.MlflowClient()
        return client.get_run(self.run_id)

    @classmethod
    def from_mlflow(cls, run_id: str) -> 'KFoldTrainingJob':
        logging.debug(
            f'Loading K-Fold training job from MLFlow run ID: {run_id}')
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        params = run.data.params
        return cls(
            dataset_path=Path(params['dataset_path']),
            dims=json.loads(params['dims']),
            activation=params['activation'],
            df_name=params['df_name'],
            batch_size=int(params['batch_size']),
            seed=int(params['seed']),
            feature_cols=json.loads(params['feature_cols']),
            label_cols=json.loads(params['label_cols']),
            inits=int(params['inits']),
            fold_col=params['fold_col'],
            folds=int(params['folds']),
            run_id=run_id,
            job_name=params['job_name'],
            accelerator=params['accelerator'],
            patience=int(params['patience']),
            checkpoints_dir=Path(params['checkpoints_dir']),
            completed=bool(params.get('completed', False)),
            max_epochs=int(params['max_epochs'])
        )

    def model_dump_mlflow(self, nested: bool = False) -> str:
        tags = {
            'model': 'MLP',
        }

        params = self.model_dump()
        params.pop('completed')
        params['dims'] = json.dumps(self.dims)
        params['feature_cols'] = json.dumps(self.feature_cols)
        params['label_cols'] = json.dumps(self.label_cols)

        with mlflow.start_run(run_name=self.job_name, tags=tags, nested=nested) as run:
            for init, fold in product(range(self.inits), range(self.folds)):
                training_job = TrainingJob(
                    dataset_path=self.dataset_path,
                    dims=self.dims,
                    activation=self.activation,
                    df_name=self.df_name,
                    batch_size=self.batch_size,
                    seed=self.seed,
                    feature_cols=self.feature_cols,
                    label_cols=self.label_cols,
                    init=init,
                    fold=fold,
                    job_name=self.job_name,
                    accelerator=self.accelerator,
                    patience=self.patience,
                    checkpoints_dir=self.checkpoints_dir /
                    f'init_{init}_fold_{fold}',
                    max_epochs=self.max_epochs
                )
                training_job.model_dump_mlflow(nested=True)
                logging.debug(
                    f'Dumped training job for init {init}, fold {fold}.')
            mlflow.log_params(params)

        self.run_id = run.info.run_id
        logging.info(f'Model dumped to MLFlow with run ID: {self.run_id}')
        return self.run_id

    def get_children(self, experiment_name: str, tracking_uri: str | None = None) -> List[mlflow.entities.run.Run]:
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

    def _exec(self, experiment_name: str, tracking_uri: str | None = None):
        children = self.get_children(experiment_name, tracking_uri)

        if not children:
            logging.critical('No child training jobs found.')
            return
        
        child_jobs = []

        for child in children:
            child_run_id = child.info.run_id
            child_job = TrainingJob.from_mlflow(child_run_id)
            logging.info(
                f'Running child training job with: run ID - {child_run_id} - fold - {child_job.fold} - init - {child_job.init}')
            child_job.exec(experiment_name=experiment_name,
                           tracking_uri=tracking_uri,
                           nested=True)
            child_jobs.append(child_job)

        # Select best based on best metric and mode
        best_job = min(
            child_jobs,
            key=lambda job: job.run.data.metrics[self.best_metric] if self.best_metric_mode == 'min' else -job.run.data.metrics[self.best_metric]
        )

        mlflow.log_param("best_job_run_id", best_job.run_id)
        mlflow.log_param("best_job_fold", best_job.fold)
        mlflow.log_param("best_job_init", best_job.init)

        # # Transfer the best model to the main run
        # best_model_path = mlflow.pytorch.get_model_path(
        #     run_id=best_job.run_id,
        #     artifact_path='model'
        # )
        # mlflow.pytorch.log_model(
        #     pytorch_model=best_job.model,
        #     artifact_path='model',
        #     run_id=self.run_id,
        #     signature=best_job.run.data.signature
        # )

        self.completed = True
        mlflow.log_param("completed", True)
        logging.info('K-Fold training jobs completed and logged to MLFlow.')

    def exec(self,
             experiment_name: str,
             tracking_uri: str | None = None,
             nested: bool = False):
        if self.run_id is None:
            raise ValueError("Run ID must be set before running the job.")
        with mlflow.start_run(self.run_id, nested=nested):
            self._exec(experiment_name, tracking_uri)


@app.command(
    help='Create a K-Fold training run for an MLP model.'
)
def create_kfold(
    dataset_path: Path,
    dims: types.DimsType,
    seed: types.SeedType = None,
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
    job_name: types.JobNameType = KFoldTrainingJob.model_fields['job_name'].default,
) -> str:

    set_logger()
    logging.debug('Setting MLFlow tracking URI and experiment name.')
    if seed is None:
        seed = types.seed_factory()

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
        seed=seed,
        feature_cols=feature_cols,
        label_cols=label_cols,
        inits=inits,
        fold_col=fold_col,
        folds=folds,
        job_name=job_name,
        accelerator=accelerator,
        patience=patience,
        checkpoints_dir=checkpoints_dir,
        max_epochs=max_epochs,
    )

    run_id = job.model_dump_mlflow()
    logging.info(f'Created K-Fold training job with run ID: {run_id}')

    return run_id


@app.command(
    help='Train an MLP model using K-Fold cross-validation.'
)
def run_kfold(
    run_id: str | None = None,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
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

    job.exec(experiment_name=experiment_name,
             tracking_uri=tracking_uri)
