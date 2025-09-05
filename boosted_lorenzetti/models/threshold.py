from typing import Tuple
import numpy as np
from abc import ABC, abstractmethod


class Threshold(ABC):

    @abstractmethod
    def compute(self, X):
        raise NotImplementedError()

    def apply(self, X, thresholded: np.ndarray) -> np.ndarray:
        return self.compute(X) > thresholded

    def apply_compute(self, X, thresholded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        computed = self.compute(X)
        return computed, computed > thresholded


class LinearThreshold(Threshold):
    def __init__(self,
                 coef: np.ndarray,
                 bias: np.ndarray):
        self.coef = coef
        self.bias = bias

    def compute(self, X):
        return np.dot(X, self.coef) + self.bias
