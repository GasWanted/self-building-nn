"""Abstract neuron interface. All neuron types implement this."""

from abc import ABC, abstractmethod
import numpy as np


class Neuron(ABC):
    """Base class for all neuron types in the self-building network.

    A neuron has a weight vector and supports four operations:
    activate, update, similarity, and copy (mitosis).
    """

    def __init__(self, n_dim: int, label: int = -1):
        self.n_dim = n_dim
        self.label = label
        self.n_fired = 0
        self.last_fire = 0
        self.age = 0

    @abstractmethod
    def activate(self, x: np.ndarray) -> float:
        """Forward pass. Return scalar activation for input x."""

    @abstractmethod
    def update(self, x: np.ndarray, lr: float):
        """Learn from input x at learning rate lr."""

    @abstractmethod
    def similarity(self, x: np.ndarray) -> float:
        """How well does this neuron match input x. Returns value in [-1, 1]."""

    @abstractmethod
    def copy(self, noise: float = 0.005) -> "Neuron":
        """Duplicate self with small noise to break symmetry (mitosis)."""

    def fire(self, step: int):
        """Record that this neuron fired at the given step."""
        self.n_fired += 1
        self.last_fire = step

    def tick(self):
        """Increment age by one step."""
        self.age += 1

    def state(self) -> dict:
        """Introspection for benchmarks and visualization."""
        return {
            "type": self.__class__.__name__,
            "n_dim": self.n_dim,
            "label": self.label,
            "n_fired": self.n_fired,
            "last_fire": self.last_fire,
            "age": self.age,
        }
