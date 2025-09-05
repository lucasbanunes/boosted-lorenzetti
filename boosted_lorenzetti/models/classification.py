import lightning as L
import torch
import torch.nn as nn
from typing import Any, Dict, Tuple
import polars as pl

from ..metrics import sp_index
from ..binary_classification import evaluate_on_data


class BinaryCrossEntropyClassifier(L.LightningModule):
    def __init__(self,
                 branch_net: nn.Module,
                 trunk_net: nn.Module,
                 class_weights: list[float] | None = None,
                 learning_rate: float = 1e-3
                 ):

        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
        self.example_input_array = [
            self.branch_nets[0].example_input_array,
            self.trunk_net.example_input_array
        ]
        self.learning_rate = learning_rate

        if class_weights is None:
            self.class_weights = torch.FloatTensor([1.0, 1.0])
        else:
            self.class_weights = torch.FloatTensor(class_weights)

    def forward(self, u, y):
        return torch.dot(self.branch_net(u), self.trunk_net(y))

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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

        with torch.no_grad():
            self.eval()
            logits = self(X_tensor)
        self.train()

        return evaluate_on_data(
            y_pred=logits,
            y_true=y_tensor,
            thresholds=thresholds,
            prefix=prefix,
            mlflow_log=mlflow_log
        )
