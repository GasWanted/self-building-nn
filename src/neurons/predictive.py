"""Predictive coding neuron — learns by minimizing prediction error."""

import numpy as np
from .base import Neuron


class PredictiveCodingNeuron(Neuron):
    """A neuron that represents a predictive model of its preferred input.

    Uses cosine similarity for activation/similarity (compatible with
    whitened PCA space). Learns by moving weights toward inputs
    (reducing prediction error).
    """

    def __init__(self, n_dim: int, label: int = -1, weights: np.ndarray = None):
        super().__init__(n_dim, label)
        if weights is not None:
            self.w = weights.copy()
        else:
            self.w = np.random.randn(n_dim) * 0.1

    def _cosine_sim(self, x: np.ndarray) -> float:
        nw = np.linalg.norm(self.w)
        nx = np.linalg.norm(x)
        if nw < 1e-9 or nx < 1e-9:
            return 0.0
        return float(np.dot(self.w, x) / (nw * nx))

    def activate(self, x: np.ndarray) -> float:
        return max(0.0, self._cosine_sim(x))

    def update(self, x: np.ndarray, lr: float):
        prediction_error = x - self.w
        self.w += lr * prediction_error

    def similarity(self, x: np.ndarray) -> float:
        return self._cosine_sim(x)

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
