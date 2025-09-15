from functools import cached_property
from typing import Literal, List, Dict, Any, Annotated, ClassVar
from pathlib import Path
import mlflow
from pydantic import ConfigDict, Field
from datetime import datetime, timezone
import json
import logging
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import shutil
import pandas as pd
import typer
from typing_extensions import TypedDict
from itertools import product
import plotly.express as px
import plotly.graph_objects as go
import torch
from mlflow.models import infer_signature


from .dataset import DuckDBDeepONetRingerDataset
from .classification import MLPUnstackedDeepONetBinaryClassifier
from .. import jobs
from .. import types
from .. import mlflow as boosted_mlflow
from ..utils import flatten_dict


TABLE_NAME_OPTION_FIELD_HELP = "Name of the table in the DuckDB database."
TableNameOptionField = Annotated[
    str,
    Field(
        description=TABLE_NAME_OPTION_FIELD_HELP,
        example="data"
    ),
    typer.Option(
        help=TABLE_NAME_OPTION_FIELD_HELP,
    )
]

RING_COL_OPTION_FIELD_HELP = "Name of the ring column in the database."
RingColOptionField = Annotated[
    str,
    Field(
        description=RING_COL_OPTION_FIELD_HELP,
        example="ring"
    ),
    typer.Option(
        help=RING_COL_OPTION_FIELD_HELP,
    )
]

ET_COL_OPTION_FIELD_HELP = "Name of the ET column in the database."
EtColOptionField = Annotated[
    str,
    Field(
        description=ET_COL_OPTION_FIELD_HELP,
        example="et"
    ),
    typer.Option(
        help=ET_COL_OPTION_FIELD_HELP,
    )
]

ETA_COL_OPTION_FIELD_HELP = "Name of the eta column in the database."
EtaColOptionField = Annotated[
    str,
    Field(
        description=ETA_COL_OPTION_FIELD_HELP,
        example="eta"
    ),
    typer.Option(
        help=ETA_COL_OPTION_FIELD_HELP,
    )
]

PILEUP_COL_OPTION_FIELD_HELP = "Name of the pileup column in the database."
PileupColOptionField = Annotated[
    str,
    Field(
        description=PILEUP_COL_OPTION_FIELD_HELP,
        example="pileup"
    ),
    typer.Option(
        help=PILEUP_COL_OPTION_FIELD_HELP,
    )
]

FOLD_OPTION_FIELD_HELP = "Fold number for cross-validation."
FoldOptionField = Annotated[
    int,
    Field(
        description=FOLD_OPTION_FIELD_HELP,
        example=0
    ),
    typer.Option(
        help=FOLD_OPTION_FIELD_HELP,
    )
]

BRANCH_DIMS_OPTION_FIELD_HELP = "List of dimensions for the branch network layers."
BranchDimsField = Annotated[
    List[int],
    Field(
        description=BRANCH_DIMS_OPTION_FIELD_HELP,
        example=[64, 32, 16]
    )
]

BRANCH_ACTIVATIONS_OPTION_FIELD_HELP = "List of activation functions for the branch network layers."
BranchActivationsField = Annotated[
    List[str | None],
    Field(
        description=BRANCH_ACTIVATIONS_OPTION_FIELD_HELP,
        example=["relu", "relu", None]
    ),
]

TRUNK_DIMS_OPTION_FIELD_HELP = "List of dimensions for the trunk network layers."
TrunkDimsField = Annotated[
    List[int],
    Field(
        description=TRUNK_DIMS_OPTION_FIELD_HELP,
        example=[64, 32, 16]
    )
]

TRUNK_ACTIVATIONS_OPTION_FIELD_HELP = "List of activation functions for the trunk network layers."
TrunkActivationsField = Annotated[
    List[str | None],
    Field(
        description=TRUNK_ACTIVATIONS_OPTION_FIELD_HELP,
        example=["relu", "relu", None]
    )
]

FOLD_COL_OPTION_FIELD_HELP = "Name of the fold column in the database."
FoldColOptionField = Annotated[
    str,
    Field(
        description=FOLD_COL_OPTION_FIELD_HELP,
        example="fold"
    ),
    typer.Option(
        help=FOLD_COL_OPTION_FIELD_HELP,
    )
]

LABEL_COL_OPTION_FIELD_HELP = "Name of the label column in the database."
LabelColOptionField = Annotated[
    str,
    Field(
        description=LABEL_COL_OPTION_FIELD_HELP,
        example="label"
    ),
    typer.Option(
        help=LABEL_COL_OPTION_FIELD_HELP,
    )
]


class MLPUnstackedDeepONetTrainingJob(jobs.MLFlowLoggedJob):

    MODEL_CKPT_PATH: ClassVar[str] = 'model.ckpt'
    METRICS_DF_PATH_FORMAT: ClassVar[str] = '{dataset_type}_metrics.csv'
    ROC_FIG_PATH_FORMAT: ClassVar[str] = '{dataset_type}_roc_curve.html'
    TPR_FPR_FIG_PATH_FORMAT: ClassVar[str] = '{dataset_type}_tpr_fpr_curve.html'
    DATASET_TYPES: ClassVar[list[str]] = ['train', 'val', 'test', 'predict']

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: jobs.NameType = 'MLP Unstacked DeepONet Training Job'

    db_path: types.DbPathOptionField
    table_name: TableNameOptionField
    ring_col: RingColOptionField
    et_col: EtColOptionField
    eta_col: EtaColOptionField
    pileup_col: PileupColOptionField
    fold: FoldOptionField
    branch_dims: BranchDimsField
    branch_activations: BranchActivationsField
    trunk_dims: TrunkDimsField
    trunk_activations: TrunkActivationsField
    fold_col: FoldColOptionField = 'fold'
    label_col: LabelColOptionField = 'label'
    learning_rate: types.LearningRateOptionField = 1e-3
    batch_size: types.BatchSizeType = 32
    accelerator: types.AcceleratorType = 'cpu'
    patience: types.PatienceType = 10
    checkpoints_dir: types.CheckpointsDirType = Path('checkpoints/')
    max_epochs: types.MaxEpochsType = 3
    monitor: types.MonitorOptionField = 'val_max_sp'

    model: MLPUnstackedDeepONetBinaryClassifier | None = None
    metrics: Dict[str, Any] = {}
    metrics_dfs: Dict[str, pd.DataFrame] = {}

    @cached_property
    def datamodule(self) -> DuckDBDeepONetRingerDataset:
        return DuckDBDeepONetRingerDataset(
            db_path=self.db_path,
            table_name=self.table_name,
            ring_col=self.ring_col,
            et_col=self.et_col,
            eta_col=self.eta_col,
            pileup_col=self.pileup_col,
            fold_col=self.fold_col,
            fold=self.fold,
            label_col=self.label_col,
            batch_size=self.batch_size
        )

    def _to_mlflow(self, tmp_dir: Path):
        mlflow.log_param('db_path', self.db_path)
        mlflow.log_param('table_name', self.table_name)
        mlflow.log_param('ring_col', self.ring_col)
        mlflow.log_param('et_col', self.et_col)
        mlflow.log_param('eta_col', self.eta_col)
        mlflow.log_param('pileup_col', self.pileup_col)
        mlflow.log_param('fold', self.fold)
        mlflow.log_param('branch_dims', json.dumps(self.branch_dims))
        mlflow.log_param('branch_activations',
                         json.dumps(self.branch_activations))
        mlflow.log_param('trunk_dims', json.dumps(self.trunk_dims))
        mlflow.log_param('trunk_activations',
                         json.dumps(self.trunk_activations))
        mlflow.log_param('fold_col', self.fold_col)
        mlflow.log_param('label_col', self.label_col)
        mlflow.log_param('learning_rate', self.learning_rate)
        mlflow.log_param('batch_size', self.batch_size)
        mlflow.log_param('accelerator', self.accelerator)
        mlflow.log_param('patience', self.patience)
        mlflow.log_param('checkpoints_dir', self.checkpoints_dir)
        mlflow.log_param('max_epochs', self.max_epochs)
        mlflow.log_param('monitor', self.monitor)

    @classmethod
    def _from_mlflow_run(cls, run) -> 'MLPUnstackedDeepONetTrainingJob':
        kwargs = dict(
            db_path=run.data.params['db_path'],
            table_name=run.data.params['table_name'],
            ring_col=run.data.params['ring_col'],
            et_col=run.data.params['et_col'],
            eta_col=run.data.params['eta_col'],
            pileup_col=run.data.params['pileup_col'],
            fold=run.data.params['fold'],
            branch_dims=json.loads(run.data.params['branch_dims']),
            branch_activations=json.loads(
                run.data.params['branch_activations']),
            trunk_dims=json.loads(run.data.params['trunk_dims']),
            trunk_activations=json.loads(run.data.params['trunk_activations']),
            fold_col=run.data.params['fold_col'],
            label_col=run.data.params['label_col'],
            learning_rate=run.data.params['learning_rate'],
            batch_size=run.data.params['batch_size'],
            accelerator=run.data.params['accelerator'],
            patience=run.data.params['patience'],
            checkpoints_dir=run.data.params['checkpoints_dir'],
            max_epochs=run.data.params['max_epochs'],
            monitor=run.data.params['monitor'],
            metrics_dfs={},
        )

        for dataset_type in cls.DATASET_TYPES:
            metrics_path = cls.METRICS_DF_PATH_FORMAT.format(
                dataset_type=dataset_type)
            if boosted_mlflow.artifact_exists(run.info.run_id, metrics_path):
                kwargs['metrics_dfs'][dataset_type] = boosted_mlflow.load_mlflow_csv(
                    run.info.run_id, metrics_path)
        return cls(**kwargs)

    def log_model(self, tmp_dir: Path, checkpoint: ModelCheckpoint | None = None):
        sample_X = self.datamodule.model_signature_df
        scaler_path = tmp_dir / 'scaler_params.json'
        with open(scaler_path, 'w') as f:
            json.dump(self.datamodule.scaler_params, f, indent=4)
        mlflow.log_artifact(scaler_path)
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

    def log_class_weights(self, tmp_dir: Path):
        class_weights = self.datamodule.balanced_class_weights
        class_weights_path = tmp_dir / 'class_weights.json'
        with open(class_weights_path, 'w') as f:
            json.dump(class_weights, f, indent=4)
        return class_weights

    def log_metrics(self, prefix: str = ''):
        for dataset_type, metrics in self.metrics.items():
            for key, value in metrics.items():
                mlflow.log_metric(f"{prefix}{dataset_type}.{key}", value)

    def log_metrics_dfs(self, tmp_dir: Path):
        for dataset_type in self.metrics_dfs.keys():
            metrics_df_path = tmp_dir / \
                self.METRICS_DF_PATH_FORMAT.format(dataset_type=dataset_type)
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
            mlflow.log_figure(roc_fig, self.ROC_FIG_PATH_FORMAT.format(
                dataset_type=dataset_type))

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
            mlflow.log_figure(tpr_fpr_fig, self.TPR_FPR_FIG_PATH_FORMAT.format(
                dataset_type=dataset_type))

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None,
             force: Literal['all', 'error'] | None = None):

        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)
        self.datamodule.log_to_mlflow()
        class_weights = self.log_class_weights(tmp_dir)

        self.model = MLPUnstackedDeepONetBinaryClassifier(
            branch_dims=self.branch_dims,
            branch_activations=self.branch_activations,
            trunk_dims=self.trunk_dims,
            trunk_activations=self.trunk_activations,
            class_weights=class_weights
        )

        logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=self.name,
            tracking_uri=tracking_uri,
            run_id=self.id_
        )

        checkpoint = ModelCheckpoint(
            monitor="val_max_sp",  # Monitor a validation metric
            dirpath=self.checkpoints_dir,  # Directory to save checkpoints
            filename='best-model-{epoch:02d}-{val_max_sp:.2f}',
            save_top_k=3,
            mode="max",  # Save based on minimum validation loss,
            save_on_train_epoch_end=False
        )
        callbacks = [
            EarlyStopping(
                monitor=self.monitor,
                patience=self.patience,
                mode="max",
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
        self.model = MLPUnstackedDeepONetBinaryClassifier.load_from_checkpoint(
            checkpoint.best_model_path
        )
        self.log_model(tmp_dir, checkpoint)
        logging.info('Training completed and model logged to MLFlow.')
        shutil.rmtree(str(self.checkpoints_dir))

        dataloaders = {
            'train': self.datamodule.train_dataloader(),
            'val': self.datamodule.val_dataloader(),
            'predict': self.datamodule.predict_dataloader()
        }

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


class KFoldMLPUnstackedDeepONetJobChild(TypedDict):
    init: int
    fold: int
    job: MLPUnstackedDeepONetTrainingJob


class KFoldMLPUnstackedDeepONetJob(jobs.MLFlowLoggedJob):

    METRICS_PATH: ClassVar[str] = 'kfold_metrics.csv'
    METRICS_DESCRIPTION_PATH: ClassVar[str] = 'kfold_metrics_description.csv'
    BEST_METRIC_BOX_PLOT_PATH: ClassVar[str] = '{best_metric}_box_plot.html'
    POSSIBLE_DATASET_TYPES: ClassVar[list[str]] = [
        'train', 'val', 'test', 'predict']
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: jobs.NameType = 'K-Fold MLP Unstacked DeepONet Job'

    db_path: types.DbPathOptionField
    table_name: TableNameOptionField
    ring_col: RingColOptionField
    et_col: EtColOptionField
    eta_col: EtaColOptionField
    pileup_col: PileupColOptionField
    branch_dims: BranchDimsField
    branch_activations: BranchActivationsField
    trunk_dims: TrunkDimsField
    trunk_activations: TrunkActivationsField
    best_metric: types.BestMetricType
    best_metric_mode: types.BestMetricModeType
    folds: types.FoldsType = 5
    inits: types.InitsType = 1
    fold_col: FoldColOptionField = 'fold'
    label_col: LabelColOptionField = 'label'
    learning_rate: types.LearningRateOptionField = 1e-3
    batch_size: types.BatchSizeType = 32
    accelerator: types.AcceleratorType = 'cpu'
    patience: types.PatienceType = 10
    checkpoints_dir: types.CheckpointsDirType = Path('checkpoints/')
    max_epochs: types.MaxEpochsType = 3
    monitor: types.MonitorOptionField = 'val_max_sp'

    children: list[KFoldMLPUnstackedDeepONetJobChild] = []
    metrics: pd.DataFrame | None = None
    best_job: MLPUnstackedDeepONetTrainingJob | None = None
    best_job_metrics: dict[str, Any] = {}
    metrics_description: pd.DataFrame | None = None

    def model_post_init(self, context):

        if not self.children:
            for init, fold in product(range(self.inits), range(self.folds)):
                job_checkpoint_dir = self.checkpoints_dir / \
                    f'fold_{fold}_init_{init}'
                job_dict = {
                    'init': init,
                    'fold': fold,
                    'job': MLPUnstackedDeepONetTrainingJob(
                        db_path=self.db_path,
                        table_name=self.table_name,
                        ring_col=self.ring_col,
                        et_col=self.et_col,
                        eta_col=self.eta_col,
                        pileup_col=self.pileup_col,
                        fold=fold,
                        branch_dims=self.branch_dims,
                        branch_activations=self.branch_activations,
                        trunk_dims=self.trunk_dims,
                        trunk_activations=self.trunk_activations,
                        fold_col=self.fold_col,
                        label_col=self.label_col,
                        learning_rate=self.learning_rate,
                        batch_size=self.batch_size,
                        accelerator=self.accelerator,
                        patience=self.patience,
                        checkpoints_dir=job_checkpoint_dir,
                        max_epochs=self.max_epochs,
                        monitor=self.monitor
                    )
                }
                self.children.append(job_dict)
        else:
            self.update_metrics_from_children()
            self.update_best_job()
        return super().model_post_init(context)

    def _to_mlflow(self, tmp_dir: Path):
        mlflow.log_param('db_path', self.db_path)
        mlflow.log_param('table_name', self.table_name)
        mlflow.log_param('ring_col', self.ring_col)
        mlflow.log_param('et_col', self.et_col)
        mlflow.log_param('eta_col', self.eta_col)
        mlflow.log_param('pileup_col', self.pileup_col)
        mlflow.log_param('branch_dims', json.dumps(self.branch_dims))
        mlflow.log_param('branch_activations',
                         json.dumps(self.branch_activations))
        mlflow.log_param('trunk_dims', json.dumps(self.trunk_dims))
        mlflow.log_param('trunk_activations',
                         json.dumps(self.trunk_activations))
        mlflow.log_param('best_metric', self.best_metric)
        mlflow.log_param('best_metric_mode', self.best_metric_mode)
        mlflow.log_param('folds', self.folds)
        mlflow.log_param('inits', self.inits)
        mlflow.log_param('fold_col', self.fold_col)
        mlflow.log_param('label_col', self.label_col)
        mlflow.log_param('learning_rate', self.learning_rate)
        mlflow.log_param('batch_size', self.batch_size)
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

    @classmethod
    def _from_mlflow_run(cls, run) -> 'KFoldMLPUnstackedDeepONetJob':
        run_id = run.info.run_id
        kwargs = dict(
            db_path=run.data.params['db_path'],
            table_name=run.data.params['table_name'],
            ring_col=run.data.params['ring_col'],
            et_col=run.data.params['et_col'],
            eta_col=run.data.params['eta_col'],
            pileup_col=run.data.params['pileup_col'],
            branch_dims=json.loads(run.data.params['branch_dims']),
            branch_activations=json.loads(
                run.data.params['branch_activations']),
            trunk_dims=json.loads(run.data.params['trunk_dims']),
            trunk_activations=json.loads(run.data.params['trunk_activations']),
            best_metric=run.data.params['best_metric'],
            best_metric_mode=run.data.params['best_metric_mode'],
            folds=run.data.params['folds'],
            inits=run.data.params['inits'],
            fold_col=run.data.params['fold_col'],
            label_col=run.data.params['label_col'],
            learning_rate=run.data.params['learning_rate'],
            batch_size=run.data.params['batch_size'],
            accelerator=run.data.params['accelerator'],
            patience=run.data.params['patience'],
            checkpoints_dir=Path(run.data.params['checkpoints_dir']),
            max_epochs=run.data.params['max_epochs'],
            monitor=run.data.params['monitor']
        )
        if boosted_mlflow.artifact_exists(run_id, cls.METRICS_PATH):
            logging.debug(f'Loading metrics from {cls.METRICS_PATH}')
            kwargs['metrics'] = boosted_mlflow.load_mlflow_csv(run_id,
                                                               cls.METRICS_PATH)
        if boosted_mlflow.artifact_exists(run_id, cls.METRICS_DESCRIPTION_PATH):
            logging.debug(f'Loading metrics description from {cls.METRICS_DESCRIPTION_PATH}')
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
                    'job': MLPUnstackedDeepONetTrainingJob.from_mlflow_run(child_run)
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
        box_plot_artifact = self.BEST_METRIC_BOX_PLOT_PATH.format(
            best_metric=self.best_metric)
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

    def update_best_job(self) -> MLPUnstackedDeepONetTrainingJob | None:
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
                return

    def update_metrics_from_children(self):
        if not self.children:
            raise RuntimeError('No child jobs to update metrics from.')
        if self.metrics is not None:
            return
        metrics = []
        for child in self.children:
            if not child['job'].metrics:
                raise RuntimeError(
                    f'Child job with fold {child["fold"]} and init {child["init"]} has no metrics.')
            child_metrics = flatten_dict(child['job'].metrics)
            child_metrics['id'] = child['job'].id_
            child_metrics['fold'] = child['fold']
            child_metrics['init'] = child['init']
            metrics.append(child_metrics)
        self.metrics = pd.DataFrame.from_records(metrics)

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

        for child in self.children:

            child['job'].execute(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                nested=True,
                force=force
            )

        self.update_metrics_from_children()
        self.update_best_job()

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
        self.best_job.log_model(tmp_dir)
        self.best_job.log_class_weights(tmp_dir)
        self.best_job.log_metrics_dfs(tmp_dir)
        self.best_job.log_roc_figs()
        self.best_job.log_tpr_fpr_figs()

        logging.info('K-Fold training jobs completed and logged to MLFlow.')
        exec_end = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', exec_end)
        mlflow.log_metric("exec_duration", exec_end - exec_start)
