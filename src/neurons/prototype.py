"""Prototype neuron — competitive learning via cosine similarity."""

import numpy as np
from .base import Neuron


class PrototypeNeuron(Neuron):
    """A neuron that stores a prototype weight vector and matches inputs
    via cosine similarity. Learns by moving its weight toward inputs
    (competitive/Hebbian absorption).
    """

    def __init__(self, n_dim: int, label: int = -1, weights: np.ndarray = None):
        super().__init__(n_dim, label)
        if weights is not None:
            self.w = weights.copy()
        else:
            self.w = np.random.randn(n_dim) * 0.1

    def activate(self, x: np.ndarray) -> float:
        return max(0.0, self.similarity(x))

    def update(self, x: np.ndarray, lr: float):
        self.w += lr * (x - self.w)

    def similarity(self, x: np.ndarray) -> float:
        nw = np.linalg.norm(self.w)
        nx = np.linalg.norm(x)
        if nw < 1e-9 or nx < 1e-9:
            return 0.0
        return float(np.dot(self.w, x) / (nw * nx))

    def copy(self, noise: float = 0.005) -> "PrototypeNeuron":
        new = PrototypeNeuron(self.n_dim, self.label, self.w)
        new.w += np.random.randn(self.n_dim) * noise
        return new

    def state(self) -> dict:
        d = super().state()
        d["w_norm"] = float(np.linalg.norm(self.w))
        return d
