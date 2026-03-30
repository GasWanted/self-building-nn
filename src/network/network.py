"""Network — self-building neural network with growth engine."""

import numpy as np
from src.neurons.base import Neuron
from src.network.layer import Layer
from src.signals.error import should_grow_width
from src.signals.information import should_grow_depth


class Network:
    """Self-building neural network.

    Starts with input -> hidden layers -> output.
    Grows width (new neurons) and depth (new layers) at runtime,
    driven by error and information signals.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        neuron_factory,
        initial_hidden_size: int = 16,
        n_initial_layers: int = 2,
        growth_interval: int = 50,
        depth_check_interval: int = 500,
        merge_threshold: float = 0.80,
        split_threshold: float = 0.35,
        depth_threshold: float = 0.90,
        prune_age: int = 3000,
        prune_window: int = 3000,
        min_layer_size: int = 4,
        growth_noise: float = 0.005,
        max_layers: int = 20,
        max_neurons_per_layer: int = 512,
        lr: float = 0.04,
        ach_decay: float = 0.55,
    ):
        self.n_input = n_input
        self.n_output = n_output
        self.neuron_factory = neuron_factory
        self.lr = lr

        # Growth parameters
        self.growth_interval = growth_interval
        self.depth_check_interval = depth_check_interval
        self.merge_threshold = merge_threshold
        self.merge_threshold_0 = merge_threshold
        self.split_threshold = split_threshold
        self.depth_threshold = depth_threshold
        self.prune_age = prune_age
        self.prune_window = prune_window
        self.min_layer_size = min_layer_size
        self.growth_noise = growth_noise
        self.max_layers = max_layers
        self.max_neurons_per_layer = max_neurons_per_layer
        self.ach = 1.0
        self.ach_decay = ach_decay

        # Build initial hidden layers
        self.layers: list[Layer] = []
        for _ in range(n_initial_layers):
            neurons = [
                neuron_factory(n_input) for _ in range(initial_hidden_size)
            ]
            self.layers.append(Layer(neurons))

        # Output layer: one neuron per class
        out_neurons = [neuron_factory(n_input, label=c) for c in range(n_output)]
        self.output_layer = Layer(out_neurons)

        # Stats
        self.step = 0
        self.n_width_grows = 0
        self.n_depth_grows = 0
        self.n_prunes = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers. Returns output activations."""
        h = x
        for layer in self.layers:
            acts = layer.forward(h)
            # Pass through: use activation-weighted combination of neuron weights
            # as input to next layer. This gives a representation, not just scalars.
            if layer.size > 0:
                weights = np.array([n.get_weights() for n in layer.neurons])
                act_sum = acts.sum()
                if act_sum > 1e-9:
                    h = (acts[:, None] * weights).sum(axis=0) / act_sum
                # If nothing activated, h passes through unchanged

        out = self.output_layer.forward(h)
        return out

    def predict(self, x: np.ndarray) -> int:
        """Predict class label for input x."""
        # Weighted vote across all layers + output
        votes = np.zeros(self.n_output)

        h = x
        for layer in self.layers:
            sims = layer.similarities(h)
            for n, s in zip(layer.neurons, sims):
                if s > 0.05 and 0 <= n.label < self.n_output:
                    votes[n.label] += s
            # Propagate
            acts = layer.forward(h)
            if layer.size > 0:
                weights = np.array([n.get_weights() for n in layer.neurons])
                act_sum = acts.sum()
                if act_sum > 1e-9:
                    h = (acts[:, None] * weights).sum(axis=0) / act_sum

        # Output layer vote
        out_sims = self.output_layer.similarities(h)
        for n, s in zip(self.output_layer.neurons, out_sims):
            if s > 0.05 and 0 <= n.label < self.n_output:
                votes[n.label] += s

        return int(np.argmax(votes))

    def learn(self, x: np.ndarray, label: int):
        """One learning step: forward, update, check growth."""
        self.step += 1

        h = x
        for layer in self.layers:
            # Find best match and update
            idx, sim = layer.best_match(h)
            best = layer.neurons[idx]

            if sim >= self.merge_threshold:
                best.update(h, self.lr)
                best.fire(self.step)
                best.label = label
            elif sim < self.split_threshold:
                # Check width growth
                if self.step % self.growth_interval == 0:
                    self._check_width_growth(layer, h, label, sim)
            else:
                best.update(h, self.lr * 0.25)
                best.fire(self.step)

            # Propagate
            acts = layer.forward(h)
            if layer.size > 0:
                weights = np.array([n.get_weights() for n in layer.neurons])
                act_sum = acts.sum()
                if act_sum > 1e-9:
                    h = (acts[:, None] * weights).sum(axis=0) / act_sum

        # Update output layer
        self.output_layer.update_neurons(h, self.lr, label)

        # Periodic checks
        if self.step % self.depth_check_interval == 0:
            self._check_depth_growth()

        if self.step % 500 == 0:
            self._prune()

    def _check_width_growth(self, layer: Layer, x: np.ndarray, label: int, sim: float):
        """Add a neuron to a layer if error signal says so."""
        if layer.size >= self.max_neurons_per_layer:
            return
        if should_grow_width(sim, self.split_threshold):
            layer.grow(1, self.growth_noise)
            # Set the new neuron's initial position and label
            layer.neurons[-1].update(x, 1.0)  # lr=1.0 snaps weight to x
            layer.neurons[-1].label = label
            self.n_width_grows += 1

    def _check_depth_growth(self):
        """Insert a new layer if information signal says so."""
        if len(self.layers) >= self.max_layers:
            return

        # Check adjacent pairs
        for i in range(len(self.layers) - 1):
            la = self.layers[i]
            lb = self.layers[i + 1]
            if la.size == 0 or lb.size == 0:
                continue

            if should_grow_depth(
                la.last_activations, lb.last_activations, self.depth_threshold
            ):
                new_layer = la.duplicate(self.growth_noise)
                self.layers.insert(i + 1, new_layer)
                self.n_depth_grows += 1
                break  # One insertion per check

    def _prune(self):
        """Remove dead neurons and empty layers."""
        dead_before = self.dead_neurons()

        for layer in self.layers:
            layer.prune(self.step, self.prune_age, self.prune_window, self.min_layer_size)

        # Remove empty layers (but keep at least 1 hidden layer)
        self.layers = [l for l in self.layers if l.size > 0]
        if not self.layers:
            neurons = [self.neuron_factory(self.n_input) for _ in range(self.min_layer_size)]
            self.layers = [Layer(neurons)]

        self.n_prunes += dead_before - self.dead_neurons()

    def tighten(self):
        """Tighten merge threshold (ACh decay). Called after sleep."""
        self.ach *= self.ach_decay
        self.merge_threshold = self.merge_threshold_0 + (0.92 - self.merge_threshold_0) * (1 - self.ach)

    def sleep_replay(self, buffer: list[tuple[np.ndarray, int]], n_steps: int = 1000):
        """Interleaved replay from hippocampal buffer."""
        rng = np.random.default_rng()
        for _ in range(n_steps):
            z, label = buffer[rng.integers(len(buffer))]
            self.learn(z, label)
        self.tighten()

    # --- Introspection ---

    def total_neurons(self) -> int:
        return sum(l.size for l in self.layers) + self.output_layer.size

    def dead_neurons(self) -> int:
        return sum(l.dead_count() for l in self.layers)

    def depth(self) -> int:
        return len(self.layers)

    def topology(self) -> list[int]:
        """Return list of layer sizes: [input, h1, h2, ..., output]."""
        return [self.n_input] + [l.size for l in self.layers] + [self.n_output]

    def state(self) -> dict:
        return {
            "step": self.step,
            "topology": self.topology(),
            "total_neurons": self.total_neurons(),
            "dead_neurons": self.dead_neurons(),
            "n_width_grows": self.n_width_grows,
            "n_depth_grows": self.n_depth_grows,
            "n_prunes": self.n_prunes,
            "merge_threshold": self.merge_threshold,
            "ach": self.ach,
            "layers": [l.state() for l in self.layers],
        }
