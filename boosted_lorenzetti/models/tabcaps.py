# Part of the code extracted from https://github.com/WhatAShot/TabCaps/tree/main

from contextlib import contextmanager
from itertools import product
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Annotated, Any, Set, Tuple, List, Literal, Dict
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy
import typer
import mlflow
from pathlib import Path
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlflow.models import infer_signature
import logging
import shutil
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
import polars as pl
from qhoptim.pyt import QHAdam

from ..metrics import (
    sp_index,
    MultiThresholdBinaryConfusionMatrix
)
from ..log import set_logger
from .. import types
from ..constants import N_RINGS
from ..jobs import MLFlowLoggedJob
from ..binary_classification import evaluate_on_data
from .mlp import MLPDataset


class TabCapsDataset(MLPDataset):
    pass


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, pred, labels):
        b = labels.shape[0]
        pred = F.softmax(pred, dim=-1)
        left = F.relu(0.9 - pred, inplace=True) ** 2
        right = F.relu(pred - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = (margin_loss.sum()) / b
        return margin_loss


"""
Other possible implementations:
https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py
https://github.com/msobroza/SparsemaxPytorch/blob/master/mnist/sparsemax.py
https://github.com/vene/sparse-structured-attention/blob/master/pytorch/torchsparseattn/sparsemax.py
"""


# credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax

        Returns
        -------
        output : torch.Tensor
            same shape as input

        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction._threshold_and_support(
            input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold

        Parameters
        ----------
        input: torch.Tensor
            any dimension
        dim : int
            dimension along which to apply the sparsemax

        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor

        """

        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class Entmax15Function(Function):
    """
    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim

        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val  # same numerical stability trick as for softmax
        input = input / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)

        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt ** 2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean ** 2)
        delta = (1 - ss) / rho

        # NOTE this is not exactly the same as in reference algo
        # Fortunately it seems the clamped values never wrongly
        # get selected by tau <= sorted_z. Prove this!
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size


class Entmoid15(Function):
    """ A highly optimized equivalent of lambda x: Entmax15([x, 0]) """

    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(F.relu(8 - input ** 2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * F.relu(tau - input, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


entmax15 = Entmax15Function.apply
entmoid15 = Entmoid15.apply


class Entmax15(nn.Module):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Entmax15, self).__init__()

    def forward(self, input):
        return entmax15(input, self.dim)


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class GBN(nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=256):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)
        # return self.bn(x)


class LearnableLocality(nn.Module):
    def __init__(self, input_dim, n_path):
        super(LearnableLocality, self).__init__()
        self.weight = nn.Parameter(torch.rand((n_path, input_dim)))
        # self.smax = Sparsemax(dim=-1)
        self.smax = Entmax15(dim=-1)

    def forward(self, x):
        mask = self.smax(self.weight)
        masked_x = torch.einsum('nd,bd->bnd', mask, x)  # [B, n_path, D]
        return masked_x


class AbstractLayer(nn.Module):
    def __init__(self, base_input_dim, base_output_dim, n_path, virtual_batch_size=256):
        super(AbstractLayer, self).__init__()
        self.masker = LearnableLocality(
            input_dim=base_input_dim, n_path=n_path)
        self.fc = nn.Conv1d(base_input_dim * n_path, 2 * n_path *
                            base_output_dim, kernel_size=1, groups=n_path)
        initialize_glu(self.fc, input_dim=base_input_dim * n_path,
                       output_dim=2 * n_path * base_output_dim)
        self.n_path = n_path
        self.base_output_dim = base_output_dim
        self.bn = GBN(2 * base_output_dim * n_path, virtual_batch_size)
        # self.bn = nn.LayerNorm(2 * base_output_dim * n_path)

    def forward(self, x):
        b = x.size(0)
        x = self.masker(x)  # [B, D] -> [B, n_path, D]
        # [B, n_path, D] -> [B, n_path * D, 1] -> [B, n_path * (2 * D'), 1]
        x = self.fc(x.view(b, -1, 1))
        x = self.bn(x.squeeze())
        chunks = x.chunk(self.n_path, 1)  # n_path * [B, 2 * D', 1]
        x = [F.relu(torch.sigmoid(x_[:, :self.base_output_dim]) * x_[:,
                    self.base_output_dim:]) for x_ in chunks]  # n_path * [B, D', 1]
        # x = torch.cat(x, dim=1).squeeze()
        return sum(x)


class PrimaryCapsuleGenerator(nn.Module):
    def __init__(self, num_feature, capsule_dim):
        super(PrimaryCapsuleGenerator, self).__init__()
        self.num_feature = num_feature
        self.capsule_dim = capsule_dim
        self.fc = nn.Parameter(torch.randn(num_feature, capsule_dim))

    def forward(self, x):
        out = torch.einsum('bm,md->bmd', x, self.fc)
        out = torch.cat([out, x[:, :, None]], dim=-1)

        return out.transpose(-1, -2)


class InferCapsule(nn.Module):
    def __init__(self, in_capsule_num, out_capsule_num, in_capsule_size, out_capsule_size, n_leaves):
        super(InferCapsule, self).__init__()
        self.in_capsule_num = in_capsule_num
        self.out_capsule_num = out_capsule_num
        self.routing_dim = out_capsule_size
        self.route_weights = nn.Parameter(torch.randn(
            in_capsule_num, out_capsule_num, in_capsule_size, out_capsule_size))
        self.smax = Entmax15(dim=-2)
        self.thread = nn.Parameter(torch.rand(
            1, in_capsule_num, out_capsule_num), requires_grad=True)
        self.routing_leaves = nn.Parameter(
            torch.rand(n_leaves, out_capsule_size))
        self.ln = nn.LayerNorm(out_capsule_size)

    @staticmethod
    def js_similarity(x, x_m):
        # x1, x2 : B, M, N, L
        dis = torch.mean((x - x_m) ** 2, dim=-1)  # B, M, N
        return dis

    def new_routing(self, priors):
        leave_hash = F.normalize(self.routing_leaves, dim=-1)
        votes = torch.sigmoid(torch.einsum(
            'ld, bmnd->bmnl', leave_hash, priors))
        mean_cap = votes.mean(dim=1, keepdim=True)  # B, 1, N, L
        dis = self.js_similarity(votes, mean_cap)
        weight = F.relu(self.thread ** 2 - dis)
        prob = torch.softmax(weight, dim=-2)  # B, M, N

        next_caps = torch.sum(prob[:, :, :, None] * priors, dim=1)
        return self.ln(next_caps)

    def forward(self, x):
        weights = self.smax(self.route_weights)
        priors = torch.einsum('bmd,mndt->bmnt', x, weights)
        outputs = self.new_routing(priors)

        return outputs


class CapsuleEncoder(nn.Module):
    def __init__(self, input_dim, out_capsule_num, init_dim, primary_capsule_dim, digit_capsule_dim, n_leaves):
        super(CapsuleEncoder, self).__init__()
        self.input_dim = input_dim
        self.init_fc = nn.Linear(input_dim, init_dim)
        digit_input_dim = init_dim + input_dim
        self.guass_primary_capsules = PrimaryCapsuleGenerator(
            digit_input_dim, primary_capsule_dim)
        self.digit_capsules = InferCapsule(in_capsule_num=primary_capsule_dim + 1, out_capsule_num=out_capsule_num,
                                           in_capsule_size=digit_input_dim, out_capsule_size=digit_capsule_dim,
                                           n_leaves=n_leaves)
        self.ln = nn.LayerNorm(digit_input_dim)

    def forward(self, x):
        init_x = self.init_fc(x)  # x: B, D'
        x = self.guass_primary_capsules(torch.cat([x, init_x], dim=1))
        x = self.ln(x)
        x = self.digit_capsules(x)  # x: B, N, T
        return x


class CapsuleClassifier(nn.Module):
    def __init__(self, input_dim, num_class, out_capsule_num, init_dim, primary_capsule_dim, digit_capsule_dim, n_leaves):
        super(CapsuleClassifier, self).__init__()
        self.net = CapsuleEncoder(
            input_dim, out_capsule_num, init_dim, primary_capsule_dim, digit_capsule_dim, n_leaves)
        self.head = head(num_class)

    def forward(self, x):
        x = self.net(x)
        out = self.head(x)
        return out


class ReconstructCapsNet(nn.Module):
    def __init__(self, input_dim, num_class, out_capsule_num, init_dim, primary_capsule_dim, digit_capsule_dim, n_leaves):
        super(ReconstructCapsNet, self).__init__()
        self.encoder = CapsuleEncoder(
            input_dim, out_capsule_num, init_dim, primary_capsule_dim, digit_capsule_dim, n_leaves)
        self.num_class = num_class
        self.sub_class = out_capsule_num // num_class
        self.digit_capsule_dim = digit_capsule_dim
        self.head = head(num_class)

        self.decoder = nn.Sequential(
            CapsuleDecoder_BasicBlock(
                digit_capsule_dim * self.sub_class, 32, 3),
            nn.Linear(32, input_dim)
        )

    def forward(self, x, y=None):
        hidden = self.encoder(x)
        pred = self.head(hidden)
        y = y.repeat(1, (hidden.shape[1] // self.num_class)
                     ).view(y.shape[0], -1, self.num_class)
        # [B, out_capsule_num, num_class]
        hidden = hidden.view(
            hidden.shape[0], -1, self.num_class, self.digit_capsule_dim)
        hidden = (hidden * y[:, :, :, None]).sum(dim=2)
        hidden = hidden.view(hidden.shape[0], -1)
        rec = self.decoder(hidden)
        return pred, rec


class head(nn.Module):
    def __init__(self, num_class):
        super(head, self).__init__()
        self.num_class = num_class

    def forward(self, x):
        x = (x ** 2).sum(dim=-1) ** 0.5
        x = x.view(x.shape[0], self.num_class, -1)
        if self.training:
            x = F.dropout(x, p=0.2)
        out = torch.sum(x, dim=-1)
        return out


class CapsuleDecoder_BasicBlock(nn.Module):
    def __init__(self, input_dim, base_outdim, n_path):
        super(CapsuleDecoder_BasicBlock, self).__init__()
        self.conv1 = AbstractLayer(input_dim, base_outdim // 2, n_path)
        self.conv2 = AbstractLayer(
            input_dim + base_outdim // 2, base_outdim, n_path)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = torch.cat([x, out1], dim=-1)
        out = self.conv2(out1)
        return out


class TabCaps(L.LightningModule):
    def __init__(self, input_dim, num_class, out_capsule_num, init_dim, primary_capsule_dim, digit_capsule_dim, n_leaves):
        super().__init__()
        self.save_hyperparameters()
        self.train_metrics = MetricCollection({
            'cm': MultiThresholdBinaryConfusionMatrix(),
            'acc': BinaryAccuracy()
        })
        self.val_metrics = self.train_metrics.clone()
        self.test_metrics = self.train_metrics.clone()
        self.model = CapsuleClassifier(
            input_dim=input_dim,
            num_class=num_class,
            out_capsule_num=out_capsule_num,
            init_dim=init_dim,
            primary_capsule_dim=primary_capsule_dim,
            digit_capsule_dim=digit_capsule_dim,
            n_leaves=n_leaves
        )
        self.loss_func = MarginLoss()
        self.example_input_array = torch.randn(
            5, input_dim, dtype=torch.float32)

    def forward(self, x):
        return self.model(x)

    def predict_proba(self, x):
        """
        Predict probabilities for the input data.
        """
        return self.forward(x)

    def get_metrics(self, collection_dict: Dict[str, Any]
                    ) -> Tuple[float,
                               float,
                               float,
                               float,
                               float]:
        """
        Extracts metrics from the collection dictionary.
        """
        fpr, tpr, tp, tn, fp, fn, thresh = collection_dict['cm']
        sp = sp_index(
            tpr,
            fpr,
            backend='torch'
        )
        max_sp_idx = torch.argmax(sp)
        auc = torch.trapezoid(tpr, fpr)
        return (
            collection_dict['acc'],
            tp[max_sp_idx],
            tn[max_sp_idx],
            fp[max_sp_idx],
            fn[max_sp_idx],
            sp[max_sp_idx],
            auc,
            fpr[max_sp_idx],
            tpr[max_sp_idx],
            thresh[max_sp_idx]
        )

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: torch.Tensor):
        x, y = batch
        prob = self(x)
        loss = self.loss_func(prob, y.float())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        batch_values = self.train_metrics(prob, y)
        acc, tp, tn, fp, fn, max_sp, roc_auc, max_sp_fpr, max_sp_tpr, _ = self.get_metrics(
            batch_values)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        self.log("train_max_sp_tp", tp, on_epoch=True, prog_bar=True)
        self.log("train_max_sp_tn", tn, on_epoch=True, prog_bar=True)
        self.log("train_max_sp_fp", fp, on_epoch=True, prog_bar=True)
        self.log("train_max_sp_fn", fn, on_epoch=True, prog_bar=True)
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
        prob = self(x)
        loss = self.loss_func(prob, y.float())
        self.val_metrics.update(prob, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        metric_values = self.val_metrics.compute()
        acc, tp, tn, fp, fn, max_sp, roc_auc, max_sp_fpr, max_sp_tpr, max_sp_thresh = self.get_metrics(
            metric_values)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log("val_max_sp_tp", tp, on_epoch=True, prog_bar=True)
        self.log("val_max_sp_tn", tn, on_epoch=True, prog_bar=True)
        self.log("val_max_sp_fp", fp, on_epoch=True, prog_bar=True)
        self.log("val_max_sp_fn", fn, on_epoch=True, prog_bar=True)
        self.log("val_max_sp", max_sp, on_epoch=True, prog_bar=True)
        self.log("val_roc_auc", roc_auc, on_epoch=True, prog_bar=True)
        self.log("val_max_sp_fpr", max_sp_fpr, on_epoch=True)
        self.log("val_max_sp_tpr", max_sp_tpr, on_epoch=True)
        self.log("val_max_sp_thresh", max_sp_thresh, on_epoch=True)
        # Reset metrics after logging
        self.val_metrics.reset()

    def configure_optimizers(self):
        # Originally it uses QHAdam from here
        # https://github.com/facebookresearch/qhoptim
        optimizer = QHAdam(self.parameters(), lr=1e-3)
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


class TrainingJob(MLFlowLoggedJob):
    db_path: Path
    init_dim: int
    primary_capsule_dim: int
    digit_capsule_dim: int
    n_leaves: int
    train_query: str
    val_query: str | None = None
    test_query: str | None = None
    predict_query: str | None = None
    label_col: str | None = 'label'
    batch_size: types.BatchSizeType = 32
    run_id: types.MLFlowRunId = None
    name: str = 'TabCaps Training Job'
    accelerator: types.AcceleratorType = 'cpu'
    patience: types.PatienceType = 10
    checkpoints_dir: types.CheckpointsDirType = Path('checkpoints/')
    max_epochs: types.MaxEpochsType = 3

    def to_mlflow(self,
                  nested: bool = False,
                  tags: List[str] = [],
                  extra_tags: Dict[str, Any] = {}) -> str:
        extra_tags['model'] = 'TabCaps'
        return super().to_mlflow(nested=nested,
                                 tags=tags,
                                 extra_tags=extra_tags)

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None = None):

        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)
        datamodule = TabCapsDataset(
            db_path=self.db_path,
            train_query=self.train_query,
            val_query=self.val_query,
            test_query=self.test_query,
            predict_query=self.predict_query,
            label_cols=self.label_col,
            batch_size=self.batch_size,
            cache=True)
        datamodule.log_to_mlflow()
        class_weights = datamodule.get_class_weights(
            how='balanced'
        )
        mlflow.log_param("class_weights", class_weights)
        model = TabCaps(input_dim=len(datamodule.feature_cols),
                        num_class=1,
                        out_capsule_num=1,
                        init_dim=self.init_dim,
                        primary_capsule_dim=self.primary_capsule_dim,
                        digit_capsule_dim=self.digit_capsule_dim,
                        n_leaves=self.n_leaves
                        )
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

        best_model = TabCaps.load_from_checkpoint(
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
    name='tabcaps',
    help='Utility for training TabCaps models on electron classification data.'
)


@app.command(
    help='Create a training run for an TabCaps model.'
)
def create_training(
    db_path: Path,
    train_query: str,
    init_dim: int,
    primary_capsule_dim: int,
    digit_capsule_dim: int,
    n_leaves: int,
    val_query: str | None = None,
    test_query: str | None = None,
    predict_query: str | None = None,
    label_col: str | None = TrainingJob.model_fields['label_col'].default,
    batch_size: types.BatchSizeType = TrainingJob.model_fields['batch_size'].default,
    name: str = TrainingJob.model_fields['name'].default,
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
        init_dim=init_dim,
        primary_capsule_dim=primary_capsule_dim,
        digit_capsule_dim=digit_capsule_dim,
        n_leaves=n_leaves,
        val_query=val_query,
        test_query=test_query,
        predict_query=predict_query,
        label_col=label_col,
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
    help='Train an TabCaps model on ingested data.'
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
    init_dim: int
    primary_capsule_dim: int
    digit_capsule_dim: int
    n_leaves: int
    best_metric: types.BestMetricType
    best_metric_mode: types.BestMetricModeType
    rings_col: str = 'rings'
    label_col: str = 'label'
    fold_col: str = 'fold'
    batch_size: types.BatchSizeType = TrainingJob.model_fields['batch_size'].default
    inits: types.InitsType = 5
    folds: types.FoldType = 5
    run_id: types.MLFlowRunId = None
    name: str = 'TabCaps K-Fold Training Job'
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
        query_template = "SELECT {feature_cols_str}, {label_col} FROM {table_name} WHERE {fold_col} != {fold} AND {fold_col} >= 0;"
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
        query_template = "SELECT {feature_cols_str}, {label_col} FROM {table_name} WHERE {fold_col} >= 0;"
        feature_cols_str = ', '.join(
            [f'{self.rings_col}[{i+1}]' for i in range(N_RINGS)])
        return query_template.format(
            feature_cols_str=feature_cols_str,
            label_col=self.label_col,
            table_name=self.table_name,
            fold_col=self.fold_col
        )

    @contextmanager
    def to_mlflow_context(self,
                          nested: bool = False,
                          tags: List[str] = [],
                          extra_tags: Dict[str, Any] = {}):
        extra_tags['model'] = 'TabCaps'
        with super().to_mlflow_context(nested=nested, tags=tags, extra_tags=extra_tags) as run:
            logging.info(
                f'Creating K-Fold training job with run ID: {run.info.run_id}')
            for init, fold in product(range(self.inits), range(self.folds)):
                job_checkpoint_dir = self.checkpoints_dir / \
                    f'fold_{fold}_init_{init}'
                job_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                job_name = f'{self.name} - fold {fold} - init {init}'

                training_job = TrainingJob(
                    db_path=self.db_path,
                    init_dim=self.init_dim,
                    primary_capsule_dim=self.primary_capsule_dim,
                    digit_capsule_dim=self.digit_capsule_dim,
                    n_leaves=self.n_leaves,
                    train_query=self.get_train_query(fold),
                    val_query=self.get_val_query(fold),
                    test_query=self.get_test_query(),
                    label_col=self.label_col,
                    batch_size=self.batch_size,
                    name=job_name,
                    accelerator=self.accelerator,
                    patience=self.patience,
                    checkpoints_dir=job_checkpoint_dir,
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

    def log_model(self, run_id: str) -> TabCaps:
        """
        Logs the model from the specified run ID to MLFlow.
        """
        with self.tmp_artifact_download(run_id, 'model.ckpt') as model_ckpt_path:
            mlflow.log_artifact(str(model_ckpt_path))
            model = TabCaps.load_from_checkpoint(model_ckpt_path)
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
                            model: TabCaps,
                            val_fold: int):

        dataset = TabCapsDataset(
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
            job_metrics_df = self.load_csv_artifact(
                job_metrics_df_artifact).sort_values('fold')
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
    help='Create a K-Fold training run for an TabCaps model.'
)
def create_kfold(
    db_path: Annotated[Path, typer.Option(
        help='Path to the DuckDB database file.'
    )],
    table_name: Annotated[str, typer.Option(
        help='Name of the DuckDB table containing the dataset.'
    )],
    init_dim: Annotated[int, typer.Option()],
    primary_capsule_dim: Annotated[int, typer.Option()],
    digit_capsule_dim: Annotated[int, typer.Option()],
    n_leaves: Annotated[int, typer.Option()],
    best_metric: types.BestMetricType,
    best_metric_mode: types.BestMetricModeType,
    rings_col: str = KFoldTrainingJob.model_fields['rings_col'].default,
    label_col: str = KFoldTrainingJob.model_fields['label_col'].default,
    fold_col: str = KFoldTrainingJob.model_fields['fold_col'].default,
    batch_size: types.BatchSizeType = KFoldTrainingJob.model_fields['batch_size'].default,
    inits: types.InitsType = KFoldTrainingJob.model_fields['inits'].default,
    folds: types.FoldType = KFoldTrainingJob.model_fields['folds'].default,
    name: str = KFoldTrainingJob.model_fields['name'].default,
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
        init_dim=init_dim,
        primary_capsule_dim=primary_capsule_dim,
        digit_capsule_dim=digit_capsule_dim,
        n_leaves=n_leaves,
        best_metric=best_metric,
        best_metric_mode=best_metric_mode,
        rings_col=rings_col,
        label_col=label_col,
        fold_col=fold_col,
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
    help='Train an TabCaps model using K-Fold cross-validation.'
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
