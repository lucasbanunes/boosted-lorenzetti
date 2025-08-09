# Code extracted from https://github.com/WhatAShot/TabCaps/tree/main


import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Dict, Any, Tuple
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy

from ..metrics import (
    sp_index,
    MultiThresholdBinaryConfusionMatrix
)


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
        # self.smax = sparsemax.Sparsemax(dim=-1)
        self.smax = sparsemax.Entmax15(dim=-1)

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
        self.smax = sparsemax.Entmax15(dim=-2)
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
        self.example_input_array = torch.randn(5, input_dim, dtype=torch.float32)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
