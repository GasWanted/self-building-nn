"""Network — self-building neural network with growth engine."""

import numpy as np
from src.neurons.base import Neuron
from src.network.layer import Layer
from src.network.propagation import refine
from src.network.connections import ConnectionTracker
from src.signals.error import should_grow_width, should_grow_depth_v2


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
        initial_hidden_size: int = 8,
        n_initial_layers: int = 2,
        growth_interval: int = 3,
        depth_check_interval: int = 10,
        merge_threshold: float = 0.85,
        split_threshold: float = 0.50,
        stagnation_threshold: float = 0.30,
        depth_patience: int = 5,
        depth_decay: float = 0.99,
        prune_age: int = 3000,
        prune_window: int = 3000,
        min_layer_size: int = 4,
        growth_noise: float = 0.005,
        max_layers: int = 10,
        max_neurons_per_layer: int = 512,
        lr: float = 0.04,
        lr_refine: float = 0.3,
        ach_decay: float = 0.55,
    ):
        self.n_input = n_input
        self.n_output = n_output
        self.neuron_factory = neuron_factory
        self.lr = lr
        self.lr_refine = lr_refine

        # Growth parameters
        self.growth_interval = growth_interval
        self.depth_check_interval = depth_check_interval
        self.merge_threshold = merge_threshold
        self.merge_threshold_0 = merge_threshold
        self.split_threshold = split_threshold
        self.stagnation_threshold = stagnation_threshold
        self.depth_patience = depth_patience
        self.depth_decay = depth_decay
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

        # Depth growth tracking
        self.layer_wrong_counts = [0.0] * n_initial_layers

        # Skip connection tracking
        self.conn_tracker = ConnectionTracker(n_initial_layers)
        self.connection_check_interval = 200
        self.connection_similarity_threshold = 0.3
        self.connection_prune_window = 1000
        self._conn_usage = {}

        # Stats
        self.step = 0
        self.n_width_grows = 0
        self.n_depth_grows = 0
        self.n_prunes = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers. Returns output activations."""
        representations = {-1: x}
        h = x
        for i, layer in enumerate(self.layers):
            if i > 0 and self.conn_tracker.n_layers > i:
                h = self.conn_tracker.blend_inputs(i, representations)
            h = refine(h, layer, self.lr_refine)
            representations[i] = h
        out = self.output_layer.forward(h)
        return out

    def predict(self, x: np.ndarray) -> int:
        """Predict class label for input x."""
        # Weighted vote across all layers + output
        votes = np.zeros(self.n_output)

        representations = {-1: x}
        h = x
        for i, layer in enumerate(self.layers):
            if i > 0 and self.conn_tracker.n_layers > i:
                h = self.conn_tracker.blend_inputs(i, representations)
            sims = layer.similarities(h)
            for n, s in zip(layer.neurons, sims):
                if s > 0.05 and 0 <= n.label < self.n_output:
                    votes[n.label] += s
            h = refine(h, layer, self.lr_refine)
            representations[i] = h

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

            h = refine(h, layer, self.lr_refine)

        # Update output layer
        self.output_layer.update_neurons(h, self.lr, label)

        # Periodic checks
        if self.step % self.depth_check_interval == 0:
            pred = self.predict(x)
            self._check_depth_growth(pred == label)

        if self.step % self.connection_check_interval == 0:
            self._check_connections()

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

    def _check_connections(self):
        """Check if non-adjacent layers provide useful signal."""
        if len(self.layers) < 3:
            return
        for target_idx in range(2, len(self.layers)):
            layer = self.layers[target_idx]
            for source_idx in range(target_idx - 2, -1, -1):
                source_layer = self.layers[source_idx]
                if source_layer.last_output is None:
                    continue
                sims = layer.similarities(source_layer.last_output)
                avg_sim = float(np.mean(sims[sims > 0])) if np.any(sims > 0) else 0
                if avg_sim > self.connection_similarity_threshold:
                    current = self.conn_tracker.get_sources(target_idx)
                    if source_idx not in current:
                        self.conn_tracker.add_connection(target_idx, source_idx, 0.5)
                    self._conn_usage[(target_idx, source_idx)] = self.step
        to_remove = []
        for (t, s), last_used in self._conn_usage.items():
            if self.step - last_used > self.connection_prune_window:
                to_remove.append((t, s))
        for t, s in to_remove:
            self.conn_tracker.remove_connection(t, s)
            del self._conn_usage[(t, s)]

    def _check_depth_growth(self, prediction_correct: bool):
        """Check depth growth using prediction-error signal."""
        if len(self.layers) >= self.max_layers:
            return

        while len(self.layer_wrong_counts) < len(self.layers):
            self.layer_wrong_counts.append(0.0)

        for i, layer in enumerate(self.layers):
            self.layer_wrong_counts[i] *= self.depth_decay

            if layer.last_input is None or layer.last_output is None:
                continue

            input_norm = np.linalg.norm(layer.last_input) + 1e-9
            diff_norm = np.linalg.norm(layer.last_output - layer.last_input)
            transformation_ratio = diff_norm / input_norm

            if not prediction_correct and transformation_ratio < self.stagnation_threshold:
                self.layer_wrong_counts[i] += 1.0

            if should_grow_depth_v2(
                transformation_ratio, self.stagnation_threshold,
                self.layer_wrong_counts[i], self.depth_patience
            ):
                new_layer = layer.duplicate(self.growth_noise)
                self.layers.insert(i + 1, new_layer)
                self.layer_wrong_counts.insert(i + 1, 0.0)
                self.conn_tracker.rebuild(len(self.layers))
                self.n_depth_grows += 1
                break

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
        self.conn_tracker.rebuild(len(self.layers))

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
