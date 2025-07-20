import torch
import torch.nn as nn
import lightning as L
from typing import Any, Tuple, List, Literal, Dict
from typer import Typer
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
from sklearn.model_selection import StratifiedKFold

from ..data import tensor_dataset_from_df
from ..constants import N_RINGS
from ..dataset.file_dataset import FileDataset
from ..metrics import sp_index


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
        weights = torch.empty(logits.shape, dtype=torch.float32, device=self.device)
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
        acc, max_sp, roc_auc, max_sp_tpr, max_sp_fpr, _ = self.get_metrics(batch_values)
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
        acc, max_sp, roc_auc, max_sp_tpr, max_sp_fpr, max_sp_thresh = self.get_metrics(metric_values)
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


def parse_list(list_str: str) -> List[Any]:
    return [item.strip() for item in list_str[1:-1].replace("'", '').split(', ')]


app = Typer(
    name='mlp',
    help='Utility for training MLP models on electron classification data.'
)


@app.command(
    help='Create a training run for an MLP model.'
)
def create_training(
    dataset_path: Path,
    dims: List[int],
    activation: str = 'relu',
    df_name: str = 'data',
    batch_size: int = 32,
    seed: int | None = None,
    feature_cols: List[str] = [f'ring_{i}' for i in range(N_RINGS)],
    label_cols: List[str] = ['label'],
    init: int = 0,
    fold: int = 0,
    tracking_uri: str | None = None,
    experiment_name: str = 'boosted-lorenzetti',
    run_id: str | None = None,
    run_name: str = 'MLP Training',
    accelerator: str = 'cpu',
    patience: int = 3,
    checkpoints_dir: Path = Path('checkpoints/')
) -> str:

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if seed is None:
        seed = np.random.randint(np.finfo(np.float64).max)

    tags = {
        'init': init,
        'fold': fold,
        'model': 'MLP',
    }

    params = dict(
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
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_id=run_id,
        run_name=run_name,
        accelerator=accelerator,
        patience=patience,
        checkpoints_dir=str(checkpoints_dir)
    )

    start_run_params = {
        'run_name': run_name,
        'tags': tags,
    }

    with mlflow.start_run(**start_run_params) as run:
        mlflow.log_params(params)

    return run.info.run_id


@app.command(
    help='Train an MLP model on ingested data.'
)
def run_training(
    run_id: str | None = None,
    dataset_path: Path | None = None,
    dims: List[int] | None = None,
    activation: str = 'relu',
    df_name: str = 'data',
    batch_size: int = 32,
    seed: int | None = None,
    feature_cols: List[str] = [f'ring_{i}' for i in range(N_RINGS)],
    label_cols: List[str] = ['label'],
    init: int = 0,
    fold: int = 0,
    tracking_uri: str | None = None,
    experiment_name: str = 'boosted-lorenzetti',
    run_name: str = 'MLP Training',
    accelerator: str = 'cpu',
    patience: int = 3,
    checkpoints_dir: Path = Path('checkpoints/')
):
    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if seed is None:
        seed = np.random.randint(np.iinfo(np.int64).max)

    if run_id:
        client = mlflow.MlflowClient(
            tracking_uri=tracking_uri,
        )
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f'run_id = "{run_id}"',
        )
        if not runs:
            raise ValueError(
                f'Run with ID {run_id} not found in experiment {experiment_name}.')
        params = runs[0].data.params
        start_run_params = dict(run_id=run_id)

        dataset_path = Path(params['dataset_path'])
        dims = [int(dim) for dim in parse_list(params['dims'])]
        activation = params['activation']
        df_name = params['df_name']
        batch_size = int(params['batch_size'])
        seed = int(params['seed'])
        feature_cols = parse_list(params['feature_cols'])
        label_cols = parse_list(params['label_cols'])
        init = int(params['init'])
        fold = int(params['fold'])
        run_name = runs[0].info.run_name
        accelerator = params['accelerator']

    else:
        if dataset_path is None:
            raise ValueError('dataset_path must be provided if run_id is not.')

        if dims is None:
            raise ValueError('dims must be provided if run_id is not.')

        tags = {
            'init': init,
            'fold': fold,
            'model': 'MLP',
        }
        start_run_params = {
            'run_name': run_name,
            'tags': tags,
            'run_id': run_id
        }

        params = dict(
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
            run_id=run_id,
            run_name=run_name,
            accelerator=accelerator,
            checkpoints_dir=str(checkpoints_dir)
        )

    file_dataset = FileDataset(dataset_path)
    load_cols = feature_cols + label_cols + ['fold', 'id']
    data_df = file_dataset.get_df(df_name,
                                  load_cols=load_cols)
    mlflow_dataset = mlflow.data.from_pandas(
        data_df,
        source=file_dataset.get_df_path(df_name),
        targets='label'
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y = data_df[label_cols].values.flatten()
    for i, (train_idx, val_idx) in enumerate(cv.split(y, y)):
        if i < fold:
            continue
        break  # Only process the specified fold
    train_y = data_df.loc[train_idx, label_cols].values.flatten()
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0., 1.]),
        y=train_y
    ).tolist()
    params['class_weights'] = class_weights
    model = MLP(dims=dims,
                class_weights=class_weights,
                activation=activation)
    with torch.no_grad():
        model.eval()
        model_input = data_df.iloc[:5][feature_cols].values
        output = model(torch.from_numpy(model_input.astype(np.float32)))
        signature = infer_signature(
            model_input=model_input,
            model_output=output.numpy()  # Convert to numpy for signature
        )
    train_dataset = tensor_dataset_from_df(
        data_df,
        feature_cols=feature_cols,
        label_cols=label_cols,
        idx=train_idx
    )
    val_dataset = tensor_dataset_from_df(
        data_df,
        feature_cols=feature_cols,
        label_cols=label_cols,
        idx=val_idx
    )

    del data_df  # Free memory
    datamodule = L.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
    )

    logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        run_id=run_id
    )
    checkpoint = ModelCheckpoint(
        monitor="val_loss",  # Monitor a validation metric
        dirpath=checkpoints_dir,  # Directory to save checkpoints
        filename="best-model",
        save_top_k=1,
        mode="min",  # Save based on minimum validation loss
    )
    callbacks = [
        EarlyStopping(
            monitor="val_max_sp",
            patience=patience,
            mode="min"
        ),
        checkpoint
    ]

    with mlflow.start_run(**start_run_params):
        mlflow.log_input(
            mlflow_dataset,
            context='training'
        )
        mlflow.log_params(params)

        trainer = L.Trainer(
            max_epochs=3,
            accelerator=accelerator,
            devices=1,
            logger=logger,
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule=datamodule)

        mlflow.pytorch.log_model(
            pytorch_model=MLP.load_from_checkpoint(
                checkpoint.best_model_path),  # Load the best model
            artifact_path="model",  # Specify the artifact path,
            signature=signature,  # Log the model signature
        )
