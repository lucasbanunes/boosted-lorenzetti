from numbers import Number
import numpy as np


def sp_index(tpr: Number,
             fpr: Number):
    return np.sqrt(
        np.sqrt(tpr*(1-fpr)) *
        ((tpr + (1-fpr))/2)
    )


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
