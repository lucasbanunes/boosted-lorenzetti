from numbers import Number
import numpy as np
from typing import Literal
import torch

TORCHMETRICS_FPR_INDEX = 0
TORCHMETRICS_TPR_INDEX = 1


def sp_index(tpr: Number,
             fpr: Number,
             backend: Literal['numpy', 'torch']
             ) -> Number:
    """
    Calculate the Specificity-Positive index (SP) given true positive rate (TPR)
    and false positive rate (FPR).

    Parameters
    ----------
    tpr : Number
        True Positive Rate (TPR)
    fpr : Number
        False Positive Rate (FPR)
    backend : Literal['numpy', 'torch']
        The backend to use for calculations, either 'numpy' or 'torch'.

    Returns
    -------
    Number
        Specificity-Positive index (SP)
    """
    match backend:
        case 'numpy':
            return np.sqrt(
                np.sqrt(tpr*(1-fpr)) *
                ((tpr + (1-fpr))/2)
            )
        case 'torch':
            return torch.sqrt(
                torch.sqrt(tpr*(1-fpr)) *
                ((tpr + (1-fpr))/2)
            )
        case _:
            raise ValueError("Unsupported backend. Use 'numpy' or 'torch'.")


def weighted_mean(data: np.ndarray,
                  weights: np.ndarray,
                  axis: int = 0):
    return np.sum(data * weights, axis=axis) / np.sum(weights, axis=axis)


def weighted_std(data: np.ndarray,
                 weights: np.ndarray,
                 axis: int = 0):
    mean = weighted_mean(data, weights, axis)
    non_zero_weights = np.sum(weights != 0, axis=axis)
    denominator = ((non_zero_weights - 1) *
                   np.sum(weights, axis=axis) /
                   non_zero_weights)
    return np.sqrt(np.sum(weights * (data - mean)**2, axis=axis) / denominator)


def roc_curve(y_true,
              y_score,
              thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 300).reshape(1, -1)
    elif len(thresholds.shape) == 1:
        thresholds = thresholds.reshape(1, -1)

    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_score.shape) == 1:
        y_score = y_score.reshape(-1, 1)
    n_positives = np.sum(y_true == 1)
    true_positives = np.sum(
        np.logical_and(y_true == 1, y_score >= thresholds),
        axis=0)
    tpr = true_positives / n_positives
    n_negatives = np.sum(y_true == 0)
    false_positives = np.sum(
        np.logical_and(y_true == 0, y_score >= thresholds),
        axis=0)
    fpr = false_positives / n_negatives
    argsort = np.argsort(fpr)
    tpr = tpr[argsort]
    fpr = fpr[argsort]
    return tpr, fpr, thresholds.flatten()
