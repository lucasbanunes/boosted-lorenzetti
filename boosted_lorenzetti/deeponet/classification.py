import lightning as L
import torch
import torch.nn as nn
from typing import Tuple, List
import pandas as pd
from torchmetrics import MetricCollection

from ..metrics import MaxSPMetrics, BCELossMetric
from ..mlp.models import build_mlp


class UnstackedDeepONetBinaryClassifier(L.LightningModule):
    def __init__(self,
                 branch_net: nn.Module,
                 trunk_net: nn.Module,
                 branch_input_dim: int,
                 class_weights: list[float] | None = None,
                 learning_rate: float = 1e-3,
                 ):

        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.branch_input_dim = branch_input_dim
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
        self.example_input_array = torch.concat([
            self.branch_net.example_input_array.reshape(1, -1),
            self.trunk_net.example_input_array.reshape(1, -1)
        ], dim=1)
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
            'loss': BCELossMetric()
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exporting to onnx doesn't support torch.linalg.vecdot
        product = (self.branch_net(x[..., :self.branch_input_dim]) *
                   self.trunk_net(x[..., self.branch_input_dim:]))
        return torch.sum(product, dim=-1)

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
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: torch.Tensor):
        x, y = batch
        logits = self(x)
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
        x, y = batch
        logits = self(x)
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
        x, y = batch
        logits = self(x)
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

    def compute_test_metrics(self):

        # Log metrics to MLflow
        computed_metrics = self.test_metrics.compute()
        loss = computed_metrics['loss']
        acc, sp, auc, fpr, tpr, tp, tn, fp, fn, thresh = \
            computed_metrics['max_sp']
        metrics = {
            'loss': float(loss),
            'max_sp_acc': float(acc),
            'max_sp': float(sp),
            'max_sp_fpr': float(fpr),
            'max_sp_tpr': float(tpr),
            'max_sp_tp': int(tp),
            'max_sp_tn': int(tn),
            'max_sp_fp': int(fp),
            'max_sp_fn': int(fn),
            'max_sp_thresh': float(thresh),
            'roc_auc': float(auc)
        }

        df = pd.DataFrame.from_dict(self.test_metrics['max_sp'].compute_arrays())

        return metrics, df


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
            learning_rate=learning_rate,
            branch_input_dim=branch_dims[0]
        )
