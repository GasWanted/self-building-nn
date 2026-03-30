"""GPU-accelerated self-building network."""

import torch
import numpy as np
from src.network.fast_layer import FastLayer
from src.signals.error import should_grow_width, should_grow_depth_v2


class FastNetwork:
    """Self-building neural network on GPU.

    Same growth engine as Network but all operations are vectorized torch tensors.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int = 10,
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
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        self.lr_refine = lr_refine
        self.growth_interval = growth_interval
        self.depth_check_interval = depth_check_interval
        self.merge_threshold = merge_threshold
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

        # Build layers
        self.layers = [
            FastLayer(n_input, initial_hidden_size, str(self.device))
            for _ in range(n_initial_layers)
        ]

        # Output layer
        self.output_layer = FastLayer(n_input, n_output, str(self.device))
        for c in range(n_output):
            self.output_layer.labels[c] = c

        # Depth tracking
        self.layer_wrong_counts = [0.0] * n_initial_layers

        # Stats
        self.step = 0
        self.n_width_grows = 0
        self.n_depth_grows = 0
        self.n_prunes = 0

    def _refine(self, x: torch.Tensor, layer: FastLayer) -> torch.Tensor:
        """Vectorized same-space propagation."""
        layer.last_input = x.clone()
        acts = layer.forward(x)
        act_sum = acts.sum()

        if act_sum < 1e-9:
            layer.last_output = x.clone()
            return x.clone()

        # Weighted shift toward prototypes: all in one matrix op
        shifts = layer.W - x.unsqueeze(0)  # (n_neurons, n_dim)
        weighted_shift = (acts.unsqueeze(1) * shifts).sum(dim=0) / act_sum

        result = x + self.lr_refine * weighted_shift
        layer.last_output = result.clone()
        return result

    def predict(self, x_np: np.ndarray) -> int:
        """Predict class label."""
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        votes = torch.zeros(self.n_output, device=self.device)

        h = x
        for layer in self.layers:
            sims = layer.similarities(h)
            # Vectorized voting: scatter_add similarities to label buckets
            valid = (sims > 0.05) & (layer.labels >= 0) & (layer.labels < self.n_output)
            if valid.any():
                valid_sims = sims[valid]
                valid_labels = layer.labels[valid]
                votes.scatter_add_(0, valid_labels, valid_sims)
            h = self._refine(h, layer)

        # Output layer vote
        out_sims = self.output_layer.similarities(h)
        valid = (out_sims > 0.05) & (self.output_layer.labels >= 0) & (self.output_layer.labels < self.n_output)
        if valid.any():
            votes.scatter_add_(0, self.output_layer.labels[valid], out_sims[valid])

        return int(votes.argmax())

    def learn(self, x_np: np.ndarray, label: int):
        """One learning step."""
        self.step += 1
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)

        h = x
        for layer in self.layers:
            idx, sim = layer.best_match(h)

            if sim >= self.merge_threshold:
                layer.W[idx] += self.lr * (h - layer.W[idx])
                layer.fire_counts[idx] += 1
                layer.last_fire[idx] = self.step
                layer.labels[idx] = label
            elif sim < self.split_threshold:
                if self.step % self.growth_interval == 0:
                    if layer.size < self.max_neurons_per_layer:
                        info_gain = 1.0 - sim
                        if info_gain > 0.02:
                            layer.grow(h, label, self.growth_noise)
                            self.n_width_grows += 1
            else:
                layer.W[idx] += (self.lr * 0.25) * (h - layer.W[idx])
                layer.fire_counts[idx] += 1
                layer.last_fire[idx] = self.step

            h = self._refine(h, layer)

        # Update output layer
        self.output_layer.update_best(h, label, self.lr)

        # Depth check
        if self.step % self.depth_check_interval == 0:
            pred = self.predict(x_np)
            self._check_depth_growth(pred == label)

        # Prune
        if self.step % 500 == 0:
            self._prune()

    def _check_depth_growth(self, prediction_correct: bool):
        if len(self.layers) >= self.max_layers:
            return
        while len(self.layer_wrong_counts) < len(self.layers):
            self.layer_wrong_counts.append(0.0)

        for i, layer in enumerate(self.layers):
            self.layer_wrong_counts[i] *= self.depth_decay
            if layer.last_input is None or layer.last_output is None:
                continue
            diff = (layer.last_output - layer.last_input).norm()
            in_norm = layer.last_input.norm() + 1e-9
            ratio = float(diff / in_norm)

            if not prediction_correct and ratio < self.stagnation_threshold:
                self.layer_wrong_counts[i] += 1.0

            if should_grow_depth_v2(ratio, self.stagnation_threshold,
                                    self.layer_wrong_counts[i], self.depth_patience):
                new_layer = layer.duplicate(self.growth_noise)
                self.layers.insert(i + 1, new_layer)
                self.layer_wrong_counts.insert(i + 1, 0.0)
                self.n_depth_grows += 1
                break

    def _prune(self):
        dead_before = self.dead_neurons()
        for layer in self.layers:
            layer.prune(self.step, self.prune_age, self.prune_window, self.min_layer_size)
        self.layers = [l for l in self.layers if l.size > 0]
        if not self.layers:
            self.layers = [FastLayer(self.n_input, self.min_layer_size, str(self.device))]
        self.n_prunes += dead_before - self.dead_neurons()

    def sleep_replay(self, buffer: list[tuple[np.ndarray, int]], n_steps: int = 1000):
        rng = np.random.default_rng()
        for _ in range(n_steps):
            z, label = buffer[rng.integers(len(buffer))]
            self.learn(z, label)

    def total_neurons(self) -> int:
        return sum(l.size for l in self.layers) + self.output_layer.size

    def dead_neurons(self) -> int:
        return sum(l.dead_count() for l in self.layers)

    def depth(self) -> int:
        return len(self.layers)

    def topology(self) -> list[int]:
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
        }
