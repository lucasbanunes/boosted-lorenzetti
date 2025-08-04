from contextlib import contextmanager
from itertools import product
from tempfile import TemporaryDirectory
import mlflow.entities
import numpy as np
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
    BinaryROC,
    BinaryConfusionMatrix
)
from mlflow.models import infer_signature
import logging
import shutil
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
import polars as pl

from ..dataset.duckdb import DuckDBDataset
from ..metrics import sp_index
from ..log import set_logger
from .. import types
from ..constants import N_RINGS
from ..jobs import MLFlowLoggedJob


class MLPDataset(DuckDBDataset):

    def get_df_from_query(self, query: str, limit: str | None = None):
        X, y = super().get_df_from_query(query, limit)
        norms = X.sum_horizontal().abs()
        norms[norms == 0] = 1
        X[X.columns] = X/norms
        if not self.label_cols:
            y = X
        return X, y


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

    def evaluate_on_data(self,
                         X: pl.DataFrame,
                         y: pl.DataFrame,
                         prefix: str = '',
                         mlflow_log: bool = False
                         ) -> Dict[str, Any]:
        """
        Evaluates the model on the provided data.
        """

        if not prefix.endswith('_') and prefix != '':
            prefix += '_'

        X_tensor = X.to_torch()
        y_tensor = y.to_torch()
        thresholds = torch.linspace(0, 1, 300)
        tp = torch.empty((len(thresholds),), dtype=torch.int32)
        fp = torch.empty((len(thresholds),), dtype=torch.int32)
        tn = torch.empty((len(thresholds),), dtype=torch.int32)
        fn = torch.empty((len(thresholds),), dtype=torch.int32)

        with torch.no_grad():
            self.eval()
            logits = self(X_tensor)

        for i, thresh in enumerate(thresholds):
            cm = BinaryConfusionMatrix(threshold=float(thresh))(logits, y_tensor)
            tp[i] = cm[1, 1]
            fp[i] = cm[0, 1]
            tn[i] = cm[0, 0]
            fn[i] = cm[1, 0]

        eval_df = pd.DataFrame({
            'thresholds': thresholds.numpy(),
            'tp': tp.numpy(),
            'fp': fp.numpy(),
            'tn': tn.numpy(),
            'fn': fn.numpy()
        })
        eval_df['tpr'] = eval_df['tp'] / (eval_df['tp'] + eval_df['fn'])
        eval_df['fpr'] = eval_df['fp'] / (eval_df['fp'] + eval_df['tn'])
        eval_df['acc'] = (eval_df['tp'] + eval_df['tn']) / len(y_tensor)
        eval_df['sp'] = sp_index(
            eval_df['tpr'].values,
            eval_df['fpr'].values,
            backend='numpy')
        sp_max_idx = eval_df['sp'].argmax()

        metrics = {
            col_name: eval_df[col_name].tolist()
            for col_name in eval_df.columns
        }

        with TemporaryDirectory() as tmp_dir:
            df_path = Path(tmp_dir) / f'{prefix}eval_df.csv'
            eval_df.to_csv(df_path, index=False)
            mlflow.log_artifact(str(df_path))

        if mlflow_log:
            mlflow.log_metric(f'{prefix}max_sp', eval_df['sp'].iloc[sp_max_idx])
            mlflow.log_metric(f'{prefix}max_sp_threshold', eval_df['thresholds'].iloc[sp_max_idx])
            mlflow.log_metric(f'{prefix}max_sp_fpr', eval_df['fpr'].iloc[sp_max_idx])
            mlflow.log_metric(f'{prefix}max_sp_tpr', eval_df['tpr'].iloc[sp_max_idx])
            mlflow.log_metric(f'{prefix}max_sp_acc', eval_df['acc'].iloc[sp_max_idx])
            roc_auc = np.trapezoid(eval_df['tpr'].values, eval_df['fpr'].values)
            mlflow.log_metric(f'{prefix}roc_auc', roc_auc)

        metrics['max_sp'] = eval_df['sp'].iloc[sp_max_idx]
        metrics['max_sp_fpr'] = eval_df['fpr'].iloc[sp_max_idx]
        metrics['max_sp_tpr'] = eval_df['tpr'].iloc[sp_max_idx]
        metrics['max_sp_acc'] = eval_df['acc'].iloc[sp_max_idx]
        metrics['max_sp_threshold'] = eval_df['thresholds'].iloc[sp_max_idx]
        metrics['roc_auc'] = np.trapezoid(
            eval_df['tpr'].values,
            eval_df['fpr'].values
        )

        if mlflow_log:
            roc_curve_artifact = f'{prefix}roc_curve.html'
            fig = px.line(
                eval_df.sort_values('fpr'),
                x='fpr',
                y='tpr',
            )
            fig.update_layout(
                title=f'ROC Curve (AUC {metrics["roc_auc"]:.2f})',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate'
            )
            mlflow.log_figure(fig, roc_curve_artifact)

        if mlflow_log:
            tpr_fpr_artifact = f'{prefix}tpr_fpr.html'
            fig = px.line(
                eval_df.sort_values('thresholds'),
                x='thresholds',
                y=['tpr', 'fpr'],
            )
            fig.update_layout(
                title='TPR and FPR vs Thresholds',
                xaxis_title='Thresholds',
                yaxis_title='Rate',
                legend_title='Rate Type'
            )
            mlflow.log_figure(fig, tpr_fpr_artifact)

        return metrics


class TrainingJob(MLFlowLoggedJob):
    db_path: Path
    train_query: str
    dims: types.DimsType
    val_query: str | None = None
    test_query: str | None = None
    predict_query: str | None = None
    label_col: str | None = 'label'
    activation: types.ActivationType = 'relu'
    batch_size: types.BatchSizeType = 32
    run_id: types.MLFlowRunId = None
    name: types.JobNameType = 'MLP Training Job'
    accelerator: types.AcceleratorType = 'cpu'
    patience: types.PatienceType = 10
    checkpoints_dir: types.CheckpointsDirType = Path('checkpoints/')
    max_epochs: types.MaxEpochsType = 3

    def to_mlflow(self,
                  nested: bool = False,
                  tags: List[str] = [],
                  extra_tags: Dict[str, Any] = {}) -> str:
        extra_tags['model'] = 'MLP'
        return super().to_mlflow(nested=nested,
                                 tags=tags,
                                 extra_tags=extra_tags)

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None = None):

        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)
        datamodule = MLPDataset(
            db_path=self.db_path,
            train_query=self.train_query,
            val_query=self.val_query,
            test_query=self.test_query,
            predict_query=self.predict_query,
            label_cols=self.label_col,
            batch_size=self.batch_size)
        datamodule.log_to_mlflow()
        class_weights = datamodule.get_class_weights(
            how='balanced'
        )
        mlflow.log_param("class_weights", class_weights)
        model = MLP(dims=self.dims,
                    class_weights=class_weights,
                    activation=self.activation)
        sample_X, _ = datamodule.get_df_from_query(self.train_query, limit=10)
        with torch.no_grad():
            model.eval()
            output = model(sample_X.to_torch())
            signature = infer_signature(
                model_input=sample_X.to_pandas(
                    use_pyarrow_extension_array=True),
                model_output=output.numpy()  # Convert to numpy for signature
            )
        model.train()
        logging.info('Model initialized and example input processed.')
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

        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=1,
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=False,
        )
        logging.info('Starting training process...')
        fit_start = datetime.now(timezone.utc)
        mlflow.log_metric('fit_start', fit_start.timestamp())
        trainer.fit(model, datamodule=datamodule)
        fit_end = datetime.now(timezone.utc)
        mlflow.log_metric('fit_end', fit_end.timestamp())
        mlflow.log_metric(
            "fit_duration", (fit_end - fit_start).total_seconds())
        logging.info('Training completed.')

        best_model = MLP.load_from_checkpoint(
            checkpoint.best_model_path
        )
        log_path = tmp_dir / 'model.ckpt'
        shutil.copy(checkpoint.best_model_path, log_path)
        mlflow.log_artifact(log_path)
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            name="ml_model",
            signature=signature,
        )
        onnx_path = tmp_dir / 'model.onnx'
        best_model.to_onnx(onnx_path, export_params=True)
        mlflow.log_artifact(str(onnx_path))
        logging.info('Training completed and model logged to MLFlow.')
        shutil.rmtree(str(self.checkpoints_dir))

        logging.info('Evaluating best model on train dataset.')
        train_X, train_y = datamodule.train_df()
        model.evaluate_on_data(
            X=train_X,
            y=train_y,
            prefix='train',
            mlflow_log=True
        )
        del train_X, train_y  # Free memory

        if self.val_query:
            logging.info('Evaluating best model on validation dataset.')
            val_X, val_y = datamodule.val_df()
            model.evaluate_on_data(
                X=val_X,
                y=val_y,
                prefix='val',
                mlflow_log=True
            )
            del val_X, val_y  # Free memory

        if self.test_query:
            logging.info('Evaluating best model on test dataset.')
            test_X, test_y = datamodule.test_df()
            model.evaluate_on_data(
                X=test_X,
                y=test_y,
                prefix='test',
                mlflow_log=True
            )
            del test_X, test_y  # Free memory

        if self.predict_query:
            logging.info('Evaluating best model on prediction dataset.')
            predict_X, predict_y = datamodule.predict_df()
            model.evaluate_on_data(
                X=predict_X,
                y=predict_y,
                prefix='predict',
                mlflow_log=True
            )
            del predict_X, predict_y  # Free memory

        end_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', end_start)
        mlflow.log_metric("exec_duration", end_start - exec_start)

    def as_metric_dict(self) -> Dict[str, Any]:
        """
        Converts the metrics of the training job to a dictionary format.
        """
        metric_dict = {
            'run_id': self.run_id,
            **self.metrics
        }
        metric_dict.update(self.tags)
        return metric_dict

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
    db_path: Path,
    train_query: str,
    dims: types.DimsType,
    val_query: str | None = None,
    test_query: str | None = None,
    predict_query: str | None = None,
    label_col: str | None = TrainingJob.model_fields['label_col'].default,
    activation: types.ActivationType = TrainingJob.model_fields['activation'].default,
    batch_size: types.BatchSizeType = TrainingJob.model_fields['batch_size'].default,
    name: types.JobNameType = TrainingJob.model_fields['name'].default,
    accelerator: types.AcceleratorType = TrainingJob.model_fields['accelerator'].default,
    patience: types.PatienceType = TrainingJob.model_fields['patience'].default,
    checkpoints_dir: types.CheckpointsDirType = TrainingJob.model_fields[
        'checkpoints_dir'].default,
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
        max_epochs=max_epochs
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
    db_path: Path
    table_name: str
    dims: types.DimsType
    best_metric: types.BestMetricType
    best_metric_mode: types.BestMetricModeType
    rings_col: str = 'rings'
    label_col: str = 'label'
    fold_col: str = 'fold'
    activation: types.ActivationType = TrainingJob.model_fields['activation'].default
    batch_size: types.BatchSizeType = TrainingJob.model_fields['batch_size'].default
    inits: types.InitsType = 5
    folds: types.FoldType = 5
    run_id: types.MLFlowRunId = None
    name: types.JobNameType = 'MLP K-Fold Training Job'
    accelerator: types.AcceleratorType = TrainingJob.model_fields['accelerator'].default
    patience: types.PatienceType = TrainingJob.model_fields['patience'].default
    checkpoints_dir: types.CheckpointsDirType = TrainingJob.model_fields[
        'checkpoints_dir'].default
    max_epochs: types.MaxEpochsType = TrainingJob.model_fields['max_epochs'].default

    def get_val_query(self, val_fold: int) -> str:
        """
        Generates the validation query for a specific fold.
        """
        query_template = "SELECT {feature_cols_str}, {label_col} FROM {table_name} WHERE {fold_col} = {fold} AND {fold_col} >= 0;"
        feature_cols_str = ', '.join(
            [f'{self.rings_col}[{i+1}]' for i in range(N_RINGS)])
        return query_template.format(
            feature_cols_str=feature_cols_str,
            label_col=self.label_col,
            table_name=self.table_name,
            fold_col=self.fold_col,
            fold=val_fold
        )

    def get_train_query(self, val_fold: int) -> str:
        """
        Generates the training query for a specific fold.
        """
        query_template = "SELECT {feature_cols_str}, {label_col} FROM {table_name} WHERE {fold_col} != {fold};"
        feature_cols_str = ', '.join(
            [f'{self.rings_col}[{i+1}]' for i in range(N_RINGS)])
        return query_template.format(
            feature_cols_str=feature_cols_str,
            label_col=self.label_col,
            table_name=self.table_name,
            fold_col=self.fold_col,
            fold=val_fold
        )

    def get_test_query(self) -> str:
        """
        Generates the test query for a specific fold.
        """
        query_template = "SELECT {feature_cols_str}, {label_col} FROM {table_name};"
        feature_cols_str = ', '.join(
            [f'{self.rings_col}[{i+1}]' for i in range(N_RINGS)])
        return query_template.format(
            feature_cols_str=feature_cols_str,
            label_col=self.label_col,
            table_name=self.table_name
        )

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
                    db_path=self.db_path,
                    train_query=self.get_train_query(fold),
                    val_query=self.get_val_query(fold),
                    test_query=self.get_test_query(),
                    dims=self.dims,
                    label_col=self.label_col,
                    activation=self.activation,
                    batch_size=self.batch_size,
                    name=self.name,
                    accelerator=self.accelerator,
                    patience=self.patience,
                    checkpoints_dir=self.checkpoints_dir,
                    max_epochs=self.max_epochs
                )
                extra_tags['init'] = init
                extra_tags['fold'] = fold
                run_id = training_job.to_mlflow(
                    nested=True, tags=tags, extra_tags=extra_tags)
                logging.info(
                    f'Created child training job with run ID: {run_id}')

            logging.info(
                f'K-Fold training job with run ID: {run.info.run_id} completed.')
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

    def log_model(self, run_id: str) -> MLP:
        """
        Logs the model from the specified run ID to MLFlow.
        """
        with self.tmp_artifact_download(run_id, 'model.ckpt') as model_ckpt_path:
            mlflow.log_artifact(str(model_ckpt_path))
            model = MLP.load_from_checkpoint(model_ckpt_path)
            model.eval()
            with torch.no_grad():
                example_output_array: torch.Tensor = model(
                    model.example_input_array)
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model",
                signature=infer_signature(
                    model_input=model.example_input_array.numpy(),
                    model_output=example_output_array.numpy()
                )
            )

        return model

    def evaluate_best_model(self,
                            model: MLP,
                            val_fold: int):

        dataset = MLPDataset(
            db_path=self.db_path,
            train_query=self.get_train_query(val_fold),
            val_query=self.get_val_query(val_fold),
            test_query=self.get_test_query(),
            label_cols=self.label_col,
            batch_size=self.batch_size
        )

        logging.info('Evaluating best model on train dataset.')
        train_X, train_y = dataset.train_df()
        model.evaluate_on_data(
            X=train_X,
            y=train_y,
            prefix='train',
            mlflow_log=True
        )

        logging.info('Evaluating best model on validation dataset.')
        val_X, val_y = dataset.val_df()
        model.evaluate_on_data(
            X=val_X,
            y=val_y,
            prefix='val',
            mlflow_log=True
        )

        logging.info('Evaluating best model on test dataset.')
        test_X, test_y = dataset.test_df()
        model.evaluate_on_data(
            X=test_X,
            y=test_y,
            prefix='test',
            mlflow_log=True
        )

    def exec(self, tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None,
             force: Literal['all', 'error'] | None = None):

        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)
        logging.info(
            f'Starting K-Fold training job with run ID: {self.run_id} in experiment: {experiment_name}')
        children = self.get_children(experiment_name, tracking_uri)
        if not children:
            raise RuntimeError('No child training jobs found.')

        children_jobs = []

        for child in children:
            child_run_id = child.info.run_id
            child_job = TrainingJob.from_mlflow(child_run_id)
            logging.info(
                f'Running child training job with: run ID - {child_run_id} - fold - {child_job.tags["fold"]} - init - {child_job.tags["init"]}')
            child_job.execute(experiment_name=experiment_name,
                              tracking_uri=tracking_uri,
                              nested=True,
                              force=force)
            children_jobs.append(child_job)
        job_metrics_df_artifact = 'job_metrics.csv'
        if self.artifact_exists(job_metrics_df_artifact):
            job_metrics_df = self.load_csv_artifact(job_metrics_df_artifact).sort_values('fold')
            logging.info(
                'Job metrics already exist, skipping metrics calculation.')
        else:
            job_metrics_df, metric_names = TrainingJob.get_metrics_df(
                children_jobs)
            job_metrics_df.sort_values('fold', inplace=True)
            job_metrics_df_path = tmp_dir / job_metrics_df_artifact
            job_metrics_df.to_csv(job_metrics_df_path, index=False)
            mlflow.log_artifact(str(job_metrics_df_path))

        if self.best_metric_mode == 'min':
            best_idx = job_metrics_df[self.best_metric].idxmin()
        elif self.best_metric_mode == 'max':
            best_idx = job_metrics_df[self.best_metric].idxmax()
        else:
            raise ValueError(
                f'Unsupported best metric mode: {self.best_metric_mode}')

        best_metrics = job_metrics_df.loc[best_idx]
        mlflow.log_param("best_job_run_id", best_metrics['run_id'])
        mlflow.log_param("best_job_fold", best_metrics['fold'])
        mlflow.log_param("best_job_init", best_metrics['init'])

        logging.info(
            f'Best job found: run ID - {best_metrics["run_id"]} - fold - {best_metrics["fold"]} - init - {best_metrics["init"]}')

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
        aggregated_metrics = job_metrics_df \
            .groupby('fold').agg(**aggregation_dict) \
            .reset_index() \
            .melt(id_vars='fold',
                  var_name='metric')

        aggrgated_metrics_artifact = 'aggregated_metrics.csv'
        if self.artifact_exists(aggrgated_metrics_artifact):
            aggregated_metrics = self.load_csv_artifact(
                aggrgated_metrics_artifact)
            logging.info(
                'Aggregated metrics already exist, skipping aggregation.')
        else:
            aggregated_metrics_path = tmp_dir / aggrgated_metrics_artifact
            aggregated_metrics.to_csv(aggregated_metrics_path, index=False)
            mlflow.log_artifact(str(aggregated_metrics_path))

        box_plot_artifact = f'{self.best_metric}_box_plot.html'
        if not self.artifact_exists(box_plot_artifact):
            best_metric_label = self.best_metric.replace('_', ' ').capitalize()
            fig = px.box(job_metrics_df, x="fold", y=self.best_metric)
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
            mlflow.log_figure(fig, box_plot_artifact)

        if not self.artifact_exists('model.onnx'):
            self.copy_artifact(best_metrics['run_id'],
                               'model.onnx',
                               dst='model.onnx')

        if not self.artifact_exists('model.ckpt'):
            model = self.log_model(best_metrics['run_id'])
            self.evaluate_best_model(model, val_fold=best_metrics['fold'])

        logging.info('K-Fold training jobs completed and logged to MLFlow.')
        exec_end = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', exec_end)
        mlflow.log_metric("exec_duration", exec_end - exec_start)


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
    db_path: Path,
    table_name: str,
    dims: types.DimsType,
    best_metric: types.BestMetricType,
    best_metric_mode: types.BestMetricModeType,
    rings_col: str = KFoldTrainingJob.model_fields['rings_col'].default,
    label_col: str = KFoldTrainingJob.model_fields['label_col'].default,
    fold_col: str = KFoldTrainingJob.model_fields['fold_col'].default,
    activation: types.ActivationType = KFoldTrainingJob.model_fields['activation'].default,
    batch_size: types.BatchSizeType = KFoldTrainingJob.model_fields['batch_size'].default,
    inits: types.InitsType = KFoldTrainingJob.model_fields['inits'].default,
    folds: types.FoldType = KFoldTrainingJob.model_fields['folds'].default,
    name: types.JobNameType = KFoldTrainingJob.model_fields['name'].default,
    accelerator: types.AcceleratorType = KFoldTrainingJob.model_fields['accelerator'].default,
    patience: types.PatienceType = KFoldTrainingJob.model_fields['patience'].default,
    checkpoints_dir: types.CheckpointsDirType = KFoldTrainingJob.model_fields[
        'checkpoints_dir'].default,
    max_epochs: types.MaxEpochsType = KFoldTrainingJob.model_fields['max_epochs'].default,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',

) -> str:

    set_logger()
    logging.debug('Setting MLFlow tracking URI and experiment name.')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    job = KFoldTrainingJob(
        db_path=db_path,
        table_name=table_name,
        dims=dims,
        best_metric=best_metric,
        best_metric_mode=best_metric_mode,
        rings_col=rings_col,
        label_col=label_col,
        fold_col=fold_col,
        activation=activation,
        batch_size=batch_size,
        inits=inits,
        folds=folds,
        name=name,
        accelerator=accelerator,
        patience=patience,
        checkpoints_dir=checkpoints_dir,
        max_epochs=max_epochs
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
                tracking_uri=tracking_uri,
                force=force)
