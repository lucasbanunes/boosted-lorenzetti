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

from .dataset import DuckDBDeepONetRingerDataset
from .classification import MLPUnstackedDeepONetBinaryClassifier
from .. import jobs
from .. import types
from ..dataset.duckdb import DuckDBDataset


DB_PATH_OPTION_FIELD_HELP = "Path to the DuckDB database file."
DbPathOptionField = Annotated[
    Path,
    Field(
        description=DB_PATH_OPTION_FIELD_HELP,
        example="data/database.duckdb"
    ),
    typer.Option(
        help=DB_PATH_OPTION_FIELD_HELP,
    )
]

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

LEARNING_RATE_OPTION_FIELD_HELP = "Learning rate for the optimizer."
LearningRateOptionField = Annotated[
    float,
    Field(
        description=LEARNING_RATE_OPTION_FIELD_HELP,
        example=1e-3
    ),
    typer.Option(
        help=LEARNING_RATE_OPTION_FIELD_HELP,
    )
]


class MLPUnstackedDeepONetTrainingJob(jobs.MLFlowLoggedJob):

    METRICS_DICT_PATH: ClassVar[str] = 'metrics.json'
    METRICS_DF_PATH_FORMAT: ClassVar[str] = '{dataset_type}_metrics.csv'

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: jobs.NameType = 'MLP Unstacked DeepONet Training Job'

    db_path: DbPathOptionField
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
    learning_rate: LearningRateOptionField = 1e-3
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
    def datamodule(self) -> DuckDBDataset:
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

    def _to_mlflow(self):
        mlflow.log_param('db_path', self.db_path)
        mlflow.log_param('table_name', self.table_name)
        mlflow.log_param('ring_col', self.ring_col)
        mlflow.log_param('et_col', self.et_col)
        mlflow.log_param('eta_col', self.eta_col)
        mlflow.log_param('pileup_col', self.pileup_col)
        mlflow.log_param('fold', self.fold)
        mlflow.log_param('branch_dims', json.dumps(self.branch_dims))
        mlflow.log_param('branch_activations', json.dumps(self.branch_activations))
        mlflow.log_param('trunk_dims', json.dumps(self.trunk_dims))
        mlflow.log_param('trunk_activations', json.dumps(self.trunk_activations))
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
            branch_activations=json.loads(run.data.params['branch_activations']),
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
        )
        return cls(**kwargs)

    def log_class_weights(self):
        mlflow.log_param("class_weights", json.dumps(self.datamodule.balanced_class_weights))

    def log_metrics(self, prefix: str = ''):
        for dataset_type, metrics in self.metrics.items():
            for key, value in metrics.items():
                mlflow.log_metric(f"{prefix}{dataset_type}.{key}", value)

    def log_metrics_dfs(self, tmp_dir: Path):
        for dataset_type in self.metrics_dfs.keys():
            metrics_df_path = tmp_dir / self.METRICS_DF_PATH_FORMAT.format(dataset_type=dataset_type)
            self.metrics_dfs[dataset_type].to_csv(metrics_df_path, index=False)
            mlflow.log_artifact(str(metrics_df_path))

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None,
             force: Literal['all', 'error'] | None = None):

        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)
        self.log_class_weights()

        self.model = MLPUnstackedDeepONetBinaryClassifier(
            branch_dims=self.branch_dims,
            branch_activations=self.branch_activations,
            trunk_dims=self.trunk_dims,
            trunk_activations=self.trunk_activations,
            class_weights=self.datamodule.balanced_class_weights
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

        self.model = MLPUnstackedDeepONetBinaryClassifier.load_from_checkpoint(
            checkpoint.best_model_path
        )
        log_path = tmp_dir / 'model.ckpt'
        shutil.copy(checkpoint.best_model_path, log_path)
        mlflow.log_artifact(log_path)
        mlflow.pytorch.log_model(
            pytorch_model=self.model,
            name="model",
        )
        # onnx_path = tmp_dir / 'model.onnx'
        # self.model.to_onnx(onnx_path,
        #                    export_params=True,
        #                    input_sample=self.model.example_input_array,
        #                    opset_version=12,
        #                    do_constant_folding=True,
        #                    input_names=['branch_input', 'trunk_input'],
        #                    output_names=['output'])
        # mlflow.log_artifact(str(onnx_path))
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

        end_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', end_start)
        mlflow.log_metric("exec_duration", end_start - exec_start)
