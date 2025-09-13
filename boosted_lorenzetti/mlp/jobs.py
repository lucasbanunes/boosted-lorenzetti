from functools import cached_property
import torch
import lightning as L
from typing import Any, Dict, ClassVar, Literal
import mlflow
from pathlib import Path
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlflow.models import infer_signature
import logging
import shutil
import pandas as pd
from datetime import datetime, timezone
import plotly.graph_objects as go
import json
from itertools import product
import plotly.express as px
import numpy as np
from pydantic import ConfigDict
from typing_extensions import TypedDict


from .dataset import MLPDataset
from .models import MLP
from .. import types
from ..jobs import MLFlowLoggedJob
from .. import mlflow as boosted_mlflow
from ..constants import N_RINGS
from ..utils import flatten_dict


class TrainingJob(MLFlowLoggedJob):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    MODEL_CKPT_PATH: ClassVar[str] = 'model.ckpt'
    METRICS_DF_PATH_FORMAT: ClassVar[str] = '{dataset_type}_metrics.csv'
    ROC_FIG_PATH_FORMAT: ClassVar[str] = '{dataset_type}_roc_curve.html'
    TPR_FPR_FIG_PATH_FORMAT: ClassVar[str] = '{dataset_type}_tpr_fpr_curve.html'

    db_path: types.DbPathOptionField
    train_query: str
    dims: types.DimsFieldType
    val_query: str | None = None
    test_query: str | None = None
    predict_query: str | None = None
    label_col: str = 'label'
    activation: types.ActivationType = 'relu'
    batch_size: types.BatchSizeType = 32
    name: str = 'MLP Training Job'
    accelerator: types.AcceleratorType = 'cpu'
    patience: types.PatienceType = 10
    checkpoints_dir: types.CheckpointsDirType = Path('checkpoints/')
    max_epochs: types.MaxEpochsType = 3
    monitor: types.MonitorOptionField = 'val_max_sp'

    model: MLP | None = None
    metrics: Dict[str, Any] = {}
    metrics_dfs: Dict[str, pd.DataFrame] = {}

    @cached_property
    def datamodule(self) -> MLPDataset:
        return MLPDataset(
            db_path=self.db_path,
            train_query=self.train_query,
            val_query=self.val_query,
            test_query=self.test_query,
            predict_query=self.predict_query,
            label_cols=self.label_col,
            batch_size=self.batch_size)

    def _to_mlflow(self):
        mlflow.log_param('db_path', self.db_path)
        mlflow.log_param('train_query', self.train_query)
        mlflow.log_param('dims', json.dumps(self.dims))
        mlflow.log_param('val_query', self.val_query)
        mlflow.log_param('test_query', self.test_query)
        mlflow.log_param('predict_query', self.predict_query)
        mlflow.log_param('label_col', self.label_col)
        mlflow.log_param('activation', self.activation)
        mlflow.log_param('batch_size', self.batch_size)
        mlflow.log_param('name', self.name)
        mlflow.log_param('accelerator', self.accelerator)
        mlflow.log_param('patience', self.patience)
        mlflow.log_param('checkpoints_dir', str(self.checkpoints_dir))
        mlflow.log_param('max_epochs', self.max_epochs)
        mlflow.log_param('monitor', self.monitor)

    @classmethod
    def _from_mlflow_run(cls, run) -> 'TrainingJob':
        run_id = run.info.run_id
        kwargs = dict(
            db_path=run.data.params['db_path'],
            train_query=run.data.params['train_query'],
            dims=json.loads(run.data.params['dims']),
            val_query=run.data.params['val_query'] if run.data.params['val_query'] != 'None' else None,
            test_query=run.data.params['test_query'] if run.data.params['test_query'] != 'None' else None,
            predict_query=run.data.params['predict_query'] if run.data.params['predict_query'] != 'None' else None,
            label_col=run.data.params['label_col'],
            activation=run.data.params['activation'],
            batch_size=run.data.params['batch_size'],
            name=run.data.params['name'],
            accelerator=run.data.params['accelerator'],
            patience=run.data.params['patience'],
            checkpoints_dir=run.data.params['checkpoints_dir'],
            max_epochs=run.data.params['max_epochs'],
            monitor=run.data.params['monitor']
        )

        if boosted_mlflow.artifact_exists(run_id, cls.MODEL_CKPT_PATH):
            kwargs['model'] = MLP.load_from_checkpoint(
                boosted_mlflow.download_artifact(run_id, cls.MODEL_CKPT_PATH)
            )
        return cls(**kwargs)

    def log_model(self, tmp_dir: Path, checkpoint: ModelCheckpoint | None = None):
        sample_X, _ = self.datamodule.get_df_from_query(
            self.train_query, limit=10)
        with torch.no_grad():
            self.model.eval()
            output = self.model(sample_X.to_torch())
            signature = infer_signature(
                model_input=sample_X.to_pandas(
                    use_pyarrow_extension_array=True),
                model_output=output.numpy()  # Convert to numpy for signature
            )

        if checkpoint is not None:
            ckpt_path = tmp_dir / self.MODEL_CKPT_PATH
            shutil.copy(checkpoint.best_model_path, ckpt_path)
            mlflow.log_artifact(ckpt_path)
        mlflow.pytorch.log_model(
            pytorch_model=self.model,
            name="model",
            signature=signature,
        )
        onnx_path = tmp_dir / 'model.onnx'
        self.model.to_onnx(onnx_path, export_params=True)
        mlflow.log_artifact(str(onnx_path))

    def log_class_weights(self, tmp_dir: Path) -> np.ndarray:
        class_weights = self.datamodule.get_class_weights(how='balanced')
        class_weights_path = tmp_dir / 'class_weights.json'
        with open(class_weights_path, 'w') as f:
            json.dump(class_weights.tolist(), f, indent=4)
        mlflow.log_artifact(str(class_weights_path))
        return class_weights

    def log_metrics(self, prefix: str = ''):
        for dataset_type, metrics in self.metrics.items():
            for key, value in metrics.items():
                mlflow.log_metric(f"{prefix}{dataset_type}.{key}", value)

    def log_metrics_dfs(self, tmp_dir: Path):
        for dataset_type in self.metrics_dfs.keys():
            metrics_df_path = tmp_dir / self.METRICS_DF_PATH_FORMAT.format(dataset_type=dataset_type)
            self.metrics_dfs[dataset_type].to_csv(metrics_df_path, index=False)
            mlflow.log_artifact(str(metrics_df_path))

    def plot_roc(self, dataset_type: str) -> go.Figure:
        roc_fig = px.line(
            self.metrics_dfs[dataset_type].sort_values('fpr'),
            x='fpr',
            y='tpr',
        )
        auc = self.metrics[dataset_type]['roc_auc']
        roc_fig.update_layout(
            title=f'ROC Curve (AUC {auc:.2f})',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )

        return roc_fig

    def log_roc_figs(self):
        for dataset_type in self.metrics_dfs.keys():
            roc_fig = self.plot_roc(dataset_type)
            mlflow.log_figure(roc_fig, self.ROC_FIG_PATH_FORMAT.format(dataset_type=dataset_type))

    def plot_tpr_fpr_plot(self, dataset_type: str) -> go.Figure:
        tpr_fpr_fig = px.line(
            self.metrics_dfs[dataset_type].sort_values('thresholds'),
            x='thresholds',
            y=['tpr', 'fpr'],
        )
        tpr_fpr_fig.update_layout(
            title='TPR and FPR vs Thresholds',
            xaxis_title='Thresholds',
            yaxis_title='Rate',
            legend_title='Rate Type',
            legend=dict(
                title_text='Variable'
            )
        )
        return tpr_fpr_fig

    def log_tpr_fpr_figs(self):
        for dataset_type in self.metrics_dfs.keys():
            tpr_fpr_fig = self.plot_tpr_fpr_plot(dataset_type)
            mlflow.log_figure(tpr_fpr_fig, self.TPR_FPR_FIG_PATH_FORMAT.format(dataset_type=dataset_type))

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None = None,
             force: Literal['all', 'error'] | None = None):

        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)

        logging.info('Logging data module to mlflow')
        self.datamodule.log_to_mlflow()
        class_weights = self.log_class_weights(tmp_dir)
        logging.info('Setting up training')
        self.model = MLP(dims=self.dims,
                         class_weights=class_weights,
                         activation=self.activation)
        self.model.train()

        logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=self.name,
            tracking_uri=tracking_uri,
            run_id=self.id_
        )

        checkpoint = ModelCheckpoint(
            monitor=self.monitor,  # Monitor a validation metric
            dirpath=self.checkpoints_dir,  # Directory to save checkpoints
            filename='best-model-{epoch:02d}-{val_max_sp:.2f}',
            save_top_k=3,
            mode="max",  # Save based on maximum validation accuracy
            save_on_train_epoch_end=False
        )

        callbacks = [
            EarlyStopping(
                monitor=self.monitor,
                patience=self.patience,
                mode="max",
                min_delta=1e-3
            ),
            checkpoint,
        ]
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=1,
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=True,
        )
        logging.info('Starting training process...')
        fit_start = datetime.now(timezone.utc)
        mlflow.log_metric('fit_start', fit_start.timestamp())
        trainer.fit(self.model, datamodule=self.datamodule)
        fit_end = datetime.now(timezone.utc)
        mlflow.log_metric('fit_end', fit_end.timestamp())
        mlflow.log_metric(
            "fit_duration", (fit_end - fit_start).total_seconds())
        self.model = MLP.load_from_checkpoint(
            checkpoint.best_model_path)
        self.log_model(tmp_dir, checkpoint)
        logging.info('Training completed and model logged to MLFlow.')
        shutil.rmtree(str(self.checkpoints_dir))

        logging.info('Evaluating best model on datasets.')
        dataloaders = {
            'train': self.datamodule.train_dataloader(),
        }
        if self.val_query:
            dataloaders['val'] = self.datamodule.val_dataloader()
        if self.test_query:
            dataloaders['test'] = self.datamodule.test_dataloader()
        if self.predict_query:
            dataloaders['predict'] = self.datamodule.predict_dataloader()

        for dataset_type, dataloader in dataloaders.items():
            logging.info(f'Evaluating best model on {dataset_type} dataset')
            trainer.test(
                model=self.model,
                dataloaders=dataloader
            )
            metrics, metrics_df = self.model.compute_test_metrics()

            self.metrics[dataset_type] = metrics
            self.metrics_dfs[dataset_type] = metrics_df
            self.model.test_metrics.reset()

        self.log_metrics()
        self.log_metrics_dfs(tmp_dir)
        self.log_roc_figs()
        self.log_tpr_fpr_figs()
        end_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', end_start)
        mlflow.log_metric("exec_duration", end_start - exec_start)


class KFoldTrainingJobChild(TypedDict):
    init: int
    fold: int
    job: TrainingJob


class KFoldTrainingJob(MLFlowLoggedJob):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    TRAIN_QUERY_TEMPLATE: ClassVar[str] = "SELECT {feature_cols_str}, {label_col} FROM {table_name} WHERE {fold_col} != {fold} AND {fold_col} >= 0;"
    VAL_QUERY_TEMPLATE: ClassVar[str] = "SELECT {feature_cols_str}, {label_col} FROM {table_name} WHERE {fold_col} = {fold} AND {fold_col} >= 0;"
    PREDICT_QUERY_TEMPLATE: ClassVar[str] = "SELECT {feature_cols_str}, {label_col} FROM {table_name} WHERE {fold_col} >= 0;"
    METRICS_PATH: ClassVar[str] = 'kfold_metrics.csv'
    METRICS_DESCRIPTION_PATH: ClassVar[str] = 'kfold_metrics_description.csv'
    BEST_METRIC_BOX_PLOT_PATH: ClassVar[str] = '{best_metric}_box_plot.html'
    POSSIBLE_DATATYPES: ClassVar[list[str]] = ['train', 'val', 'test', 'predict']

    db_path: Path
    ring_col: str
    dims: types.DimsFieldType
    best_metric: types.BestMetricType
    best_metric_mode: types.BestMetricModeType
    fold_col: str = 'fold'
    label_col: str = 'label'
    table_name: str = 'data'
    inits: types.InitsType = 5
    folds: types.FoldsType = 5
    activation: types.ActivationType = 'relu'
    batch_size: types.BatchSizeType = 32
    name: str = 'MLP KFold Training Job'
    accelerator: types.AcceleratorType = 'cpu'
    patience: types.PatienceType = 10
    checkpoints_dir: types.CheckpointsDirType = Path('checkpoints/')
    max_epochs: types.MaxEpochsType = 3
    monitor: types.MonitorOptionField = 'val_max_sp'

    children: list[KFoldTrainingJobChild] = []
    metrics: pd.DataFrame | None = None
    best_job: TrainingJob | None = None
    best_job_metrics: dict[str, Any] = {}
    metrics_description: pd.DataFrame | None = None

    def model_post_init(self, context):

        # if self.n_jobs < 1:
        #     raise ValueError(f'n_jobs must be at least 1, received {self.n_jobs}')

        if not self.children:
            feature_cols_str = ', '.join([f'{self.ring_col}[{i+1}]' for i in range(N_RINGS)])
            for init, fold in product(range(self.inits), range(self.folds)):
                job_checkpoint_dir = self.checkpoints_dir / f'fold_{fold}_init_{init}'
                job_dict = {
                    'init': init,
                    'fold': fold,
                    'job': TrainingJob(
                        db_path=self.db_path,
                        train_query=self.TRAIN_QUERY_TEMPLATE.format(
                            feature_cols_str=feature_cols_str,
                            label_col=self.label_col,
                            table_name=self.table_name,
                            fold_col=self.fold_col,
                            fold=fold
                        ),
                        val_query=self.VAL_QUERY_TEMPLATE.format(
                            feature_cols_str=feature_cols_str,
                            label_col=self.label_col,
                            table_name=self.table_name,
                            fold_col=self.fold_col,
                            fold=fold
                        ),
                        predict_query=self.PREDICT_QUERY_TEMPLATE.format(
                            feature_cols_str=feature_cols_str,
                            label_col=self.label_col,
                            table_name=self.table_name,
                            fold_col=self.fold_col,
                        ),
                        dims=self.dims,
                        label_col=self.label_col,
                        activation=self.activation,
                        batch_size=self.batch_size,
                        name=f'{self.name} - fold {fold} - init {init}',
                        accelerator=self.accelerator,
                        patience=self.patience,
                        checkpoints_dir=job_checkpoint_dir,
                        max_epochs=self.max_epochs,
                        monitor=self.monitor
                    )
                }
                self.children.append(job_dict)
        return super().model_post_init(context)

    def _to_mlflow(self):
        mlflow.log_param('db_path', self.db_path)
        mlflow.log_param('ring_col', self.ring_col)
        mlflow.log_param('dims', json.dumps(self.dims))
        mlflow.log_param('best_metric', self.best_metric)
        mlflow.log_param('best_metric_mode', self.best_metric_mode)
        mlflow.log_param('fold_col', self.fold_col)
        mlflow.log_param('label_col', self.label_col)
        mlflow.log_param('table_name', self.table_name)
        mlflow.log_param('inits', self.inits)
        mlflow.log_param('folds', self.folds)
        mlflow.log_param('activation', self.activation)
        mlflow.log_param('batch_size', self.batch_size)
        mlflow.log_param('name', self.name)
        mlflow.log_param('accelerator', self.accelerator)
        mlflow.log_param('patience', self.patience)
        mlflow.log_param('checkpoints_dir', str(self.checkpoints_dir))
        mlflow.log_param('max_epochs', self.max_epochs)
        mlflow.log_param('monitor', self.monitor)
        for child in self.children:
            tags = {
                'fold': str(child['fold']),
                'init': str(child['init']),
            }
            child['job'].to_mlflow(nested=True, tags=tags)
        # mlflow.log_param('n_jobs', self.n_jobs)

    @classmethod
    def _from_mlflow_run(cls, run) -> 'TrainingJob':
        run_id = run.info.run_id
        kwargs = dict(
            db_path=run.data.params['db_path'],
            ring_col=run.data.params['ring_col'],
            dims=json.loads(run.data.params['dims']),
            best_metric=run.data.params['best_metric'],
            best_metric_mode=run.data.params['best_metric_mode'],
            fold_col=run.data.params['fold_col'],
            label_col=run.data.params['label_col'],
            table_name=run.data.params['table_name'],
            inits=run.data.params['inits'],
            folds=run.data.params['folds'],
            activation=run.data.params['activation'],
            batch_size=run.data.params['batch_size'],
            name=run.data.params['name'],
            accelerator=run.data.params['accelerator'],
            patience=run.data.params['patience'],
            checkpoints_dir=run.data.params['checkpoints_dir'],
            max_epochs=run.data.params['max_epochs'],
            monitor=run.data.params['monitor'],
            # n_jobs=run.data.params['n_jobs']
        )
        if boosted_mlflow.artifact_exists(run_id, cls.METRICS_PATH):
            kwargs['metrics'] = boosted_mlflow.load_mlflow_csv(run_id,
                                                               cls.METRICS_PATH)
        if boosted_mlflow.artifact_exists(run_id, cls.METRICS_DESCRIPTION_PATH):
            kwargs['metrics_description'] = boosted_mlflow.load_mlflow_csv(run_id,
                                                                           cls.METRICS_DESCRIPTION_PATH)

        client = mlflow.MlflowClient()
        children_runs = client.search_runs(
            experiment_ids=[run.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=1000
        )
        kwargs['children'] = []
        for child_run in children_runs:
            kwargs['children'].append(
                {
                    'fold': int(child_run.data.tags['fold']),
                    'init': int(child_run.data.tags['init']),
                    'job': TrainingJob.from_mlflow_run(child_run)
                }
            )
        return cls(**kwargs)

    def make_box_plot(self) -> go.Figure:
        best_metric_label = self.best_metric.replace('_', ' ').capitalize()
        fig = px.box(self.metrics, x="fold", y=self.best_metric)
        fig.update_layout(
            title=f'K-Fold {best_metric_label} Distribution',
            xaxis_title='Fold',
            yaxis_title=best_metric_label
        )
        fig.add_hline(y=self.best_job_metrics[self.best_metric],
                      line_dash="dash",
                      line_color="red",
                      annotation_text='Best',
                      annotation_position="top left")
        return fig

    def log_metrics(self, tmp_dir: Path):
        metrics_path = tmp_dir / self.METRICS_PATH
        self.metrics.to_csv(metrics_path, index=False)
        mlflow.log_artifact(str(metrics_path))
        box_plot_artifact = self.BEST_METRIC_BOX_PLOT_PATH.format(best_metric=self.best_metric)
        fig = self.make_box_plot()
        mlflow.log_figure(fig, box_plot_artifact)

        for key, value in self.best_job_metrics.items():
            if isinstance(value, str):
                mlflow.log_param(f'best_job.{key}', value)
            else:
                mlflow.log_metric(f'best_job.{key}', value)

    def log_metrics_description(self, tmp_dir: Path):
        metrics_description_path = tmp_dir / self.METRICS_DESCRIPTION_PATH
        self.metrics_description.to_csv(metrics_description_path, index=False)
        mlflow.log_artifact(str(metrics_description_path))

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None = None,
             force: Literal['all', 'error'] | None = None):

        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)
        logging.info(
            f'Starting K-Fold training job with run ID: {self.id_} in experiment: {experiment_name}')
        if not self.children:
            raise RuntimeError('No child training jobs found.')

        metrics = []
        # if self.n_jobs == 1:
        for child in self.children:

            child['job'].execute(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                nested=True,
                force=False
            )
            children_metrics = flatten_dict(child['job'].metrics)
            children_metrics['id'] = child['job'].id_
            children_metrics['fold'] = child['fold']
            children_metrics['init'] = child['init']
            metrics.append(children_metrics)
        # else:
        #     def run_child(child: KFoldTrainingJobChild) -> Dict[str, Any]:
        #         child['job'].execute(
        #             experiment_name=experiment_name,
        #             tracking_uri=tracking_uri,
        #             nested=True,
        #             force=False
        #         )
        #         children_metrics = flatten_dict(child['job'].metrics)
        #         children_metrics['id'] = child['job'].id_
        #         children_metrics['fold'] = child['fold']
        #         children_metrics['init'] = child['init']
        #         return children_metrics

        #     metrics = joblib.Parallel(n_jobs=self.n_jobs)(
        #         joblib.delayed(run_child)(child) for child in self.children
        #     )

        self.metrics = pd.DataFrame.from_records(metrics)

        if self.best_metric_mode == 'min':
            best_idx = self.metrics[self.best_metric].idxmin()
        elif self.best_metric_mode == 'max':
            best_idx = self.metrics[self.best_metric].idxmax()
        else:
            raise ValueError(
                f'Unsupported best metric mode: {self.best_metric_mode}')

        self.best_job_metrics = self.metrics.loc[best_idx].to_dict()
        for child in self.children:
            if child['job'].id_ == self.best_job_metrics['id']:
                self.best_job = child['job']
                break

        logging.info(f'Best job found: run ID - {self.best_job_metrics["id"]}'
                     f' - fold - {self.best_job_metrics["fold"]} - init - {self.best_job_metrics["init"]}')

        aggregation_dict = dict()
        for metric in self.metrics.columns:
            if metric in ['id', 'fold', 'init']:
                continue
            aggregation_dict[f'{metric}.mean'] = pd.NamedAgg(
                column=metric, aggfunc='mean')
            aggregation_dict[f'{metric}.std'] = pd.NamedAgg(
                column=metric, aggfunc='std')
            aggregation_dict[f'{metric}.min'] = pd.NamedAgg(
                column=metric, aggfunc='min')
            aggregation_dict[f'{metric}.max'] = pd.NamedAgg(
                column=metric, aggfunc='max')
            aggregation_dict[f'{metric}.median'] = pd.NamedAgg(
                column=metric, aggfunc='median')
        self.metrics_description = self.metrics \
            .groupby('fold').agg(**aggregation_dict) \
            .reset_index() \
            .melt(id_vars='fold',
                  var_name='metric')
        self.log_metrics(tmp_dir)
        self.log_metrics_description(tmp_dir)
        self.best_job.log_class_weights(tmp_dir)
        # self.best_job.log_metrics(prefix='best_job.')
        self.best_job.log_model(tmp_dir)
        self.best_job.log_metrics_dfs(tmp_dir)
        self.best_job.log_roc_figs()
        self.best_job.log_tpr_fpr_figs()

        logging.info('K-Fold training jobs completed and logged to MLFlow.')
        exec_end = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', exec_end)
        mlflow.log_metric("exec_duration", exec_end - exec_start)
