"""Dendritic neuron — multiple input compartments with on-demand growth."""

import numpy as np
from .base import Neuron


class DendriticNeuron(Neuron):
    """A neuron with multiple dendrites that grow on demand.

    Starts with 1 dendrite. Adds more when absorbing inputs that don't
    match any existing dendrite (similarity < dendrite_threshold).
    Max dendrites capped at max_dendrites.
    """

    def __init__(self, n_dim: int, label: int = -1, n_dendrites: int = 4):
        super().__init__(n_dim, label)
        self.max_dendrites = n_dendrites
        self.dendrite_weights = [np.random.randn(n_dim) * 0.1]
        self._best_dendrite = 0
        self.dendrite_threshold = 0.5

    def _dendrite_sim(self, x: np.ndarray, w: np.ndarray) -> float:
        nw = np.linalg.norm(w)
        nx = np.linalg.norm(x)
        if nw < 1e-9 or nx < 1e-9:
            return 0.0
        return float(np.dot(w, x) / (nw * nx))

    def activate(self, x: np.ndarray) -> float:
        sims = [self._dendrite_sim(x, w) for w in self.dendrite_weights]
        self._best_dendrite = int(np.argmax(sims))
        return max(0.0, sims[self._best_dendrite])

    def update(self, x: np.ndarray, lr: float):
        best_sim = self._dendrite_sim(x, self.dendrite_weights[self._best_dendrite])
        if best_sim < self.dendrite_threshold and len(self.dendrite_weights) < self.max_dendrites:
            self.dendrite_weights.append(x.copy())
        else:
            w = self.dendrite_weights[self._best_dendrite]
            self.dendrite_weights[self._best_dendrite] = w + lr * (x - w)

    def similarity(self, x: np.ndarray) -> float:
        sims = [self._dendrite_sim(x, w) for w in self.dendrite_weights]
        self._best_dendrite = int(np.argmax(sims))
        return sims[self._best_dendrite]

    def get_weights(self) -> np.ndarray:
        return self.dendrite_weights[self._best_dendrite].copy()

    def copy(self, noise: float = 0.005) -> "DendriticNeuron":
        new = DendriticNeuron(self.n_dim, self.label, self.max_dendrites)
        new.dendrite_weights = [w.copy() + np.random.randn(self.n_dim) * noise
                                for w in self.dendrite_weights]
        new.dendrite_threshold = self.dendrite_threshold
        return new

    def state(self) -> dict:
        d = super().state()
        d["max_dendrites"] = self.max_dendrites
        d["n_active_dendrites"] = len(self.dendrite_weights)
        d["best_dendrite"] = self._best_dendrite
        return d
