import lightning as L
import torch
import torch.nn as nn
from typing import Tuple, List
import pandas as pd
from torchmetrics import MetricCollection
from pathlib import Path
import mlflow
import plotly.express as px

from ..metrics import MaxSPMetrics, BCEWithLogitsLossMetric
from ..mlp.models import build_mlp


class UnstackedDeepONetBinaryClassifier(L.LightningModule):
    def __init__(self,
                 branch_net: nn.Module,
                 trunk_net: nn.Module,
                 class_weights: list[float] | None = None,
                 learning_rate: float = 1e-3,
                 ):

        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
        self.example_input_array = [
            self.branch_net.example_input_array,
            self.trunk_net.example_input_array
        ]
        self.example_input_df = None
        self.learning_rate = learning_rate

        if class_weights is None:
            self.class_weights = torch.FloatTensor([1.0, 1.0])
        else:
            self.class_weights = torch.FloatTensor(class_weights)

        self.train_metrics = MetricCollection({
            'max_sp': MaxSPMetrics(
                thresholds=100
            )
        })
        self.val_metrics = self.train_metrics.clone()
        self.test_metrics = MetricCollection({
            'max_sp': MaxSPMetrics(
                thresholds=100
            ),
            'loss': BCEWithLogitsLossMetric()
        })

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vecdot(
            self.branch_net(branch_input),
            self.trunk_net(trunk_input)
        )

    def reset_metrics(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()

    def get_loss(self, logits: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        weights = torch.empty(
            y.shape, dtype=torch.float32, device=self.device)
        weights[y == 1] = self.class_weights[1]
        weights[y == 0] = self.class_weights[0]
        return torch.mean(weights*self.loss_func(logits, y))

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: torch.Tensor):
        branch_input, trunk_input, y = batch
        logits = self(branch_input, trunk_input)
        loss = self.get_loss(logits, y.float())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        prob = torch.sigmoid(logits)
        batch_values = self.train_metrics(prob, y)['max_sp']
        self.log('train_acc', batch_values[0], on_epoch=True, prog_bar=True)
        self.log("train_max_sp", batch_values[1], on_epoch=True, prog_bar=True)
        self.log("train_roc_auc", batch_values[2], on_epoch=True, prog_bar=True)
        self.log("train_max_sp_fpr", batch_values[3], on_epoch=True)
        self.log("train_max_sp_tpr", batch_values[4], on_epoch=True)
        self.log("train_max_sp_tp", batch_values[5], on_epoch=True)
        self.log("train_max_sp_tn", batch_values[6], on_epoch=True)
        self.log("train_max_sp_fp", batch_values[7], on_epoch=True)
        self.log("train_max_sp_fn", batch_values[8], on_epoch=True)
        self.log("train_max_sp_thresh", batch_values[9], on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        # Reset metrics after each epoch
        self.train_metrics.reset()

    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: torch.Tensor):
        branch_input, trunk_input, y = batch
        logits = self(branch_input, trunk_input)
        loss = self.get_loss(logits, y.float())
        prob = torch.sigmoid(logits)
        self.val_metrics.update(prob, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        metric_values = self.val_metrics.compute()['max_sp']
        self.log("val_acc", metric_values[0], on_epoch=True, prog_bar=True)
        self.log("val_max_sp", metric_values[1], on_epoch=True, prog_bar=True)
        self.log("val_roc_auc", metric_values[2], on_epoch=True, prog_bar=True)
        self.log("val_max_sp_fpr", metric_values[3], on_epoch=True)
        self.log("val_max_sp_tpr", metric_values[4], on_epoch=True)
        self.log("val_max_sp_tp", metric_values[5], on_epoch=True)
        self.log("val_max_sp_tn", metric_values[6], on_epoch=True)
        self.log("val_max_sp_fp", metric_values[7], on_epoch=True)
        self.log("val_max_sp_fn", metric_values[8], on_epoch=True)
        self.log("val_max_sp_thresh", metric_values[9], on_epoch=True)
        # Reset metrics after logging
        self.val_metrics.reset()

    def test_step(self,
                  batch: Tuple[torch.Tensor, torch.Tensor],
                  batch_idx: torch.Tensor):
        branch_input, trunk_input, y = batch
        logits = self(branch_input, trunk_input)
        loss = self.get_loss(logits, y.float())
        prob = torch.sigmoid(logits)
        self.test_metrics.update(prob, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return optimizer

    def log_test_metrics(self,
                         tmp_dir: Path,
                         prefix: str = 'test.'):
        if tmp_dir is None:
            tmp_dir = Path("tmp")

        # Log metrics to MLflow
        computed_metrics = self.test_metrics.compute()
        loss = computed_metrics['loss']
        acc, sp, auc, fpr, tpr, tp, tn, fp, fn, thresh = \
            computed_metrics['max_sp']
        metrics = {
            'loss': loss,
            'acc': acc,
            'sp': sp,
            'fpr': fpr,
            'tpr': tpr,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'thresh': thresh
        }
        mlflow.log_metric(f"{prefix}loss", loss)
        mlflow.log_metric(f"{prefix}max_sp_acc", acc)
        mlflow.log_metric(f"{prefix}max_sp_sp", sp)
        mlflow.log_metric(f"{prefix}auc", auc)
        mlflow.log_metric(f"{prefix}max_sp_fpr", fpr)
        mlflow.log_metric(f"{prefix}max_sp_tpr", tpr)
        mlflow.log_metric(f"{prefix}max_sp_tp", tp)
        mlflow.log_metric(f"{prefix}max_sp_tn", tn)
        mlflow.log_metric(f"{prefix}max_sp_fp", fp)
        mlflow.log_metric(f"{prefix}max_sp_fn", fn)
        mlflow.log_metric(f"{prefix}max_sp_thresh", thresh)

        df = pd.DataFrame.from_dict(self.test_metrics['max_sp'].compute_arrays())
        df_path = tmp_dir / f'{prefix}metrics.csv'
        df.to_csv(df_path, index=False)
        mlflow.log_artifact(str(df_path))

        roc_curve_artifact = f'{prefix}roc_curve.html'
        roc_fig = px.line(
            df.sort_values('fpr'),
            x='fpr',
            y='tpr',
        )
        roc_fig.update_layout(
            title=f'ROC Curve (AUC {auc:.2f})',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        mlflow.log_figure(roc_fig, roc_curve_artifact)

        tpr_fpr_artifact = f'{prefix}tpr_fpr.html'
        tpr_fpr_fig = px.line(
            df.sort_values('thresholds'),
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
        mlflow.log_figure(tpr_fpr_fig, tpr_fpr_artifact)

        return metrics, df, roc_fig, tpr_fpr_fig


class MLPUnstackedDeepONetBinaryClassifier(UnstackedDeepONetBinaryClassifier):

    def __init__(self,
                 branch_dims: List[int],
                 branch_activations: List[str | None],
                 trunk_dims: List[int],
                 trunk_activations: List[str | None],
                 class_weights: list[float] | None = None,
                 learning_rate: float = 1e-3):
        self.save_hyperparameters()
        super().__init__(
            branch_net=build_mlp(
                dims=branch_dims, activations=branch_activations),
            trunk_net=build_mlp(
                dims=trunk_dims, activations=trunk_activations),
            class_weights=class_weights,
            learning_rate=learning_rate
        )
