from functools import cached_property
from typing import Literal, List, Dict, Any
from pathlib import Path
import duckdb
import mlflow
from pydantic import ConfigDict
from datetime import datetime, timezone
import json
import logging
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import shutil
import pandas as pd
import plotly.graph_objects as go


from .classification import MLPUnstackedDeepONetBinaryClassifier
from .. import jobs
from .. import types
from ..dataset.duckdb import DuckDBDataset


class MLPUnstackedDeepONetTrainingJob(jobs.MLFlowLoggedJob):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: jobs.NameType = 'MLP Unstacked DeepONet Training Job'

    db_path: Path
    table_name: str
    rings_cols: List[str]
    et_col: str
    eta_col: str
    pileup_col: str
    label_col: str
    fold_col: str
    fold: int
    branch_dims: List[int]
    branch_activations: List[str | None]
    trunk_dims: List[int]
    trunk_activations: List[str | None]
    label_col: str | None = 'label'
    learning_rate: float = 1e-3
    batch_size: types.BatchSizeType = 32
    run_id: types.MLFlowRunId = None
    accelerator: types.AcceleratorType = 'cpu'
    patience: types.PatienceType = 10
    checkpoints_dir: types.CheckpointsDirType = Path('checkpoints/')
    max_epochs: types.MaxEpochsType = 3
    monitor: str = 'val_max_sp'

    model: MLPUnstackedDeepONetBinaryClassifier | None = None
    metrics: Dict[str, Any] = {}
    metrics_dfs: Dict[str, pd.DataFrame] = {}
    roc_figs: Dict[str, go.Figure] = {}
    tpr_fpr_figs: Dict[str, go.Figure] = {}

    @cached_property
    def datamodule(self) -> DuckDBDataset:
        with duckdb.connect(self.db_path) as conn:
            relation = conn.table(self.table_name)
            relation = relation.filter(f'{self.fold_col} != {self.fold} AND {self.fold_col} >= 0')
            relation = relation.aggregate(', '.join([
                f'mean({self.et_col}) as mean_cl_et',
                f'stddev_samp({self.et_col}) as stddev_cl_et',
                f'mean(abs({self.eta_col})) as mean_abs_cl_eta',
                f'stddev_samp(abs({self.eta_col})) as stddev_abs_cl_eta',
                f'mean({self.pileup_col}::Float) as mean_cl_pileup',
                f'stddev_samp({self.pileup_col}::Float) as stddev_cl_pileup',
            ]))
            result = relation.df().iloc[0].to_dict()  # noqa: F841 Ignores the unused variable

        return DuckDBDataset(
            db_path=self.db_path,
            train_query=self.train_query,
            val_query=self.val_query,
            test_query=self.test_query,
            predict_query=self.predict_query,
            label_cols=self.label_col,
            batch_size=self.batch_size,
        )

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None,
             force: Literal['all', 'error'] | None = None):

        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)

        self.datamodule.log_to_mlflow()
        class_weights = self.datamodule.get_class_weights(how='balanced')
        mlflow.log_param("class_weights", json.dumps(class_weights))

        self.model = MLPUnstackedDeepONetBinaryClassifier(
            branch_dims=self.branch_dims,
            branch_activations=self.branch_activations,
            trunk_dims=self.trunk_dims,
            trunk_activations=self.trunk_activations,
            class_weights=class_weights
        )
        sample_X, _ = self.datamodule.get_df_from_query(self.train_query, limit=10)
        signature = self.model.mlflow_model_signature(model_input=sample_X)

        train_X, train_y = self.datamodule.train_df()
        train_X = train_X.to_numpy()
        train_y = train_y.to_numpy()
        logging.info('Fitting Kmeans')

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
                monitor=self.monitor,
                patience=self.patience,
                mode="min",
                check_on_train_epoch_end=False
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
        trainer.fit(self.model, datamodule=self.datamodule)
        fit_end = datetime.now(timezone.utc)
        mlflow.log_metric('fit_end', fit_end.timestamp())
        mlflow.log_metric(
            "fit_duration", (fit_end - fit_start).total_seconds())
        logging.info('Training completed.')

        best_model = type(self.model).load_from_checkpoint(
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

        dataloaders = {
            'train': self.datamodule.train_dataloader(),
        }
        if self.datamodule.val_query:
            dataloaders['val'] = self.datamodule.val_dataloader()
        if self.datamodule.test_query:
            dataloaders['test'] = self.datamodule.test_dataloader()
        if self.datamodule.predict_query:
            dataloaders['predict'] = self.datamodule.predict_dataloader()

        for dataset_type, dataloader in dataloaders.items():
            logging.info(f'Evaluating best model on {dataset_type} dataset')
            trainer.test(
                model=self.model,
                dataloaders=dataloader
            )
            metrics, metrics_df, roc_fig, tpr_fpr_fig = \
                self.model.log_test_metrics(tmp_dir, prefix=f'{dataset_type}.')
            self.metrics[dataset_type] = metrics
            self.metrics_dfs[dataset_type] = metrics_df
            self.roc_figs[dataset_type] = roc_fig
            self.tpr_fpr_figs[dataset_type] = tpr_fpr_fig
            self.model.test_metrics.reset()

        end_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', end_start)
        mlflow.log_metric("exec_duration", end_start - exec_start)
