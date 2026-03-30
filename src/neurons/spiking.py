"""Spiking neuron — Leaky Integrate-and-Fire with STDP learning."""

import numpy as np
from .base import Neuron


class SpikingNeuron(Neuron):
    """LIF neuron with spike-timing-dependent plasticity.

    Membrane potential accumulates input, fires when threshold exceeded.
    STDP: pre-before-post strengthens, post-before-pre weakens.
    Runs n_ticks internal timesteps per activate() call.
    """

    def __init__(self, n_dim: int, label: int = -1, weights: np.ndarray = None,
                 v_thresh: float = 1.0, tau: float = 0.9, t_ref: int = 3,
                 n_ticks: int = 5):
        super().__init__(n_dim, label)
        if weights is not None:
            self.w = weights.copy()
        else:
            self.w = np.random.randn(n_dim) * 0.1
        self.v = 0.0
        self.v_thresh = v_thresh
        self.tau = tau
        self.t_ref = t_ref
        self.n_ticks = n_ticks
        self.steps_since_spike = t_ref  # start ready to fire
        self._last_spiked = False

    def activate(self, x: np.ndarray) -> float:
        """Run n_ticks timesteps. Return 1.0 if any spike occurred, 0.0 otherwise."""
        spiked = False
        for _ in range(self.n_ticks):
            if self.steps_since_spike < self.t_ref:
                self.v = 0.0
                self.steps_since_spike += 1
            else:
                self.v = self.tau * self.v + float(np.dot(self.w, x))
                if self.v >= self.v_thresh:
                    spiked = True
                    self.v = 0.0
                    self.steps_since_spike = 0
                else:
                    self.steps_since_spike += 1
        self._last_spiked = spiked
        return 1.0 if spiked else 0.0

    def update(self, x: np.ndarray, lr: float):
        """STDP-inspired update."""
        if self._last_spiked:
            self.w += lr * x
        else:
            self.w -= lr * 0.5 * x
        norm = np.linalg.norm(self.w)
        if norm > 10.0:
            self.w = self.w / norm * 10.0

    def similarity(self, x: np.ndarray) -> float:
        nw = np.linalg.norm(self.w)
        nx = np.linalg.norm(x)
        if nw < 1e-9 or nx < 1e-9:
            return 0.0
        return float(np.clip(np.dot(self.w, x) / (nw * nx), -1.0, 1.0))

    def get_weights(self) -> np.ndarray:
        return self.w.copy()

    def copy(self, noise: float = 0.005) -> "SpikingNeuron":
        new = SpikingNeuron(self.n_dim, self.label, self.w,
                           self.v_thresh, self.tau, self.t_ref, self.n_ticks)
        new.w += np.random.randn(self.n_dim) * noise
        new.v = 0.0
        return new

    def state(self) -> dict:
        d = super().state()
        d["v"] = self.v
        d["v_thresh"] = self.v_thresh
        d["last_spiked"] = self._last_spiked
        return d
