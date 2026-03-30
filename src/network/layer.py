"""Layer — ordered collection of neurons with growth/prune operations."""

import numpy as np
from src.neurons.base import Neuron
from src.network.inhibition import apply_inhibition


class Layer:
    """A layer of neurons, all the same type.

    Tracks activation statistics for the growth engine signals.
    Supports width growth (duplicate neuron), pruning, and full duplication (mitosis).
    """

    _uid = 0

    def __init__(self, neurons: list = None, winner_fraction: float = 0.3,
                 inhibition_factor: float = 0.1):
        self.id = Layer._uid
        Layer._uid += 1
        self.neurons = neurons or []
        self.last_activations = np.array([])
        self.activation_variance = 0.0
        self.step = 0
        self.winner_fraction = winner_fraction
        self.inhibition_factor = inhibition_factor

    @property
    def size(self) -> int:
        return len(self.neurons)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Activate all neurons, return activation vector."""
        self.step += 1
        acts = np.array([n.activate(x) for n in self.neurons])
        acts = apply_inhibition(acts, self.winner_fraction, self.inhibition_factor)
        self.last_activations = acts
        self.activation_variance = float(np.var(acts)) if len(acts) > 1 else 0.0
        for i, a in enumerate(acts):
            self.neurons[i].tick()
            if a > 0:
                self.neurons[i].fire(self.step)
        return acts

    def similarities(self, x: np.ndarray) -> np.ndarray:
        """Return similarity scores for all neurons without side effects."""
        return np.array([n.similarity(x) for n in self.neurons])

    def best_match(self, x: np.ndarray) -> tuple[int, float]:
        """Return (index, similarity) of the best-matching neuron."""
        sims = self.similarities(x)
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])

    def grow(self, n: int = 1, noise: float = 0.005):
        """Add n neurons by duplicating the most active existing neuron."""
        if not self.neurons:
            return
        # Pick neuron with highest fire count
        best = max(self.neurons, key=lambda n: n.n_fired)
        for _ in range(n):
            self.neurons.append(best.copy(noise))

    def prune(self, step: int, prune_age: int, prune_window: int, min_size: int):
        """Remove neurons that haven't fired. Respect min_size unless all are dead."""
        if self.size <= min_size:
            # At minimum size — only remove if ALL neurons are dead
            all_dead = all(
                n.n_fired == 0 and n.age > prune_age for n in self.neurons
            )
            if all_dead:
                self.neurons.clear()
            return

        remove = []
        for i, n in enumerate(self.neurons):
            if n.n_fired == 0 and n.age > prune_age:
                remove.append(i)
            elif n.n_fired > 0 and (step - n.last_fire) > prune_window:
                remove.append(i)

        # Don't prune below min_size
        max_removable = self.size - min_size
        remove = remove[:max_removable]

        for i in sorted(remove, reverse=True):
            self.neurons.pop(i)

    def duplicate(self, noise: float = 0.005) -> "Layer":
        """Copy the entire layer (mitosis). Returns new layer with copied neurons."""
        new_neurons = [n.copy(noise) for n in self.neurons]
        return Layer(new_neurons, self.winner_fraction, self.inhibition_factor)

    def update_neurons(self, x: np.ndarray, lr: float, label: int = -1):
        """Update the best-matching neuron (competitive learning)."""
        idx, sim = self.best_match(x)
        self.neurons[idx].update(x, lr)
        if label >= 0:
            self.neurons[idx].label = label

    def dead_count(self) -> int:
        return sum(1 for n in self.neurons if n.n_fired == 0)

    def label_distribution(self) -> dict[int, int]:
        dist = {}
        for n in self.neurons:
            dist[n.label] = dist.get(n.label, 0) + 1
        return dist

    def state(self) -> dict:
        return {
            "id": self.id,
            "size": self.size,
            "dead": self.dead_count(),
            "activation_variance": self.activation_variance,
            "neurons": [n.state() for n in self.neurons],
        }
