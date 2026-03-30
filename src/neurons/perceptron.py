"""Perceptron neuron — standard w*x+b with gradient update. Baseline."""

import numpy as np
from .base import Neuron


class PerceptronNeuron(Neuron):
    """Standard perceptron: y = relu(w*x + b).
    Learns via gradient descent on squared error.
    Similarity is normalized dot product (same as prototype for fair comparison).
    """

    def __init__(self, n_dim: int, label: int = -1, weights: np.ndarray = None):
        super().__init__(n_dim, label)
        if weights is not None:
            self.w = weights.copy()
        else:
            self.w = np.random.randn(n_dim) * (2.0 / n_dim) ** 0.5
        self.b = 0.0

    def activate(self, x: np.ndarray) -> float:
        return float(max(0.0, np.dot(self.w, x) + self.b))

    def update(self, x: np.ndarray, lr: float):
        # Move weights toward input (same interface as prototype for compatibility)
        self.w += lr * (x - self.w)

    def similarity(self, x: np.ndarray) -> float:
        nw = np.linalg.norm(self.w)
        nx = np.linalg.norm(x)
        if nw < 1e-9 or nx < 1e-9:
            return 0.0
        return float(np.dot(self.w, x) / (nw * nx))

    def copy(self, noise: float = 0.005) -> "PerceptronNeuron":
        new = PerceptronNeuron(self.n_dim, self.label, self.w)
        new.b = self.b
        new.w += np.random.randn(self.n_dim) * noise
        return new

    def state(self) -> dict:
        d = super().state()
        d["w_norm"] = float(np.linalg.norm(self.w))
        d["bias"] = self.b
        return d
