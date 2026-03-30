"""Predictive coding neuron — learns by minimizing prediction error."""

import numpy as np
from .base import Neuron


class PredictiveCodingNeuron(Neuron):
    """A neuron that represents a predictive model of its preferred input.

    Activation is inversely proportional to prediction error:
    high activation = input matches prediction (low surprise).
    Learns by moving weights toward inputs (reducing future error).
    """

    def __init__(self, n_dim: int, label: int = -1, weights: np.ndarray = None):
        super().__init__(n_dim, label)
        if weights is not None:
            self.w = weights.copy()
        else:
            self.w = np.random.randn(n_dim) * 0.1

    def activate(self, x: np.ndarray) -> float:
        error = np.linalg.norm(x - self.w)
        return float(1.0 / (1.0 + error))

    def update(self, x: np.ndarray, lr: float):
        prediction_error = x - self.w
        self.w += lr * prediction_error

    def similarity(self, x: np.ndarray) -> float:
        error = np.linalg.norm(x - self.w)
        denom = np.linalg.norm(x) + np.linalg.norm(self.w) + 1e-9
        return float(1.0 - error / denom)

    def get_weights(self) -> np.ndarray:
        return self.w.copy()

    def copy(self, noise: float = 0.005) -> "PredictiveCodingNeuron":
        new = PredictiveCodingNeuron(self.n_dim, self.label, self.w)
        new.w += np.random.randn(self.n_dim) * noise
        return new

    def state(self) -> dict:
        d = super().state()
        d["w_norm"] = float(np.linalg.norm(self.w))
        return d
