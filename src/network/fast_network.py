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

    def _refine_batch(self, X: torch.Tensor, layer: FastLayer) -> torch.Tensor:
        """Batch propagation: X is (batch, n_dim) -> (batch, n_dim)."""
        # (batch, n_neurons)
        sims = layer.similarities_batch(X)
        acts = torch.relu(sims)

        # Lateral inhibition per sample
        if layer.size > 1:
            k = max(1, int(layer.size * layer.winner_fraction))
            if k < layer.size:
                topk_vals, _ = torch.topk(acts, k, dim=1)
                threshold = topk_vals[:, -1:] # (batch, 1)
                mask = acts >= threshold
                acts = torch.where(mask, acts, acts * layer.inhibition_factor)

        act_sum = acts.sum(dim=1, keepdim=True).clamp(min=1e-9)  # (batch, 1)

        # (batch, n_neurons, 1) * (1, n_neurons, n_dim) -> weighted shifts
        # More efficient: use einsum
        # shifts = W - X[:, None, :]  would be (batch, n_neurons, n_dim) — too much memory
        # Instead: weighted_proto = (acts @ W) / act_sum, then shift = weighted_proto - X
        weighted_proto = (acts @ layer.W) / act_sum  # (batch, n_dim)
        shift = weighted_proto - X
        return X + self.lr_refine * shift

    def predict_batch(self, X_np: np.ndarray) -> np.ndarray:
        """Batch predict: X_np is (batch, n_dim) -> (batch,) int labels."""
        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        batch = X.shape[0]
        votes = torch.zeros(batch, self.n_output, device=self.device)

        h = X
        for layer in self.layers:
            sims = layer.similarities_batch(h)  # (batch, n_neurons)
            # Voting: for each sample, accumulate sims into label buckets
            valid_labels = layer.labels.clamp(0, self.n_output - 1)  # (n_neurons,)
            # mask out invalid labels and low sims
            mask = (sims > 0.05) & (layer.labels >= 0).unsqueeze(0) & (layer.labels < self.n_output).unsqueeze(0)
            masked_sims = sims * mask.float()  # (batch, n_neurons)
            # scatter add: for each neuron, add its sim to the label bucket
            label_expanded = valid_labels.unsqueeze(0).expand(batch, -1)  # (batch, n_neurons)
            votes.scatter_add_(1, label_expanded, masked_sims)
            h = self._refine_batch(h, layer)

        # Output layer
        out_sims = self.output_layer.similarities_batch(h)
        valid_labels = self.output_layer.labels.clamp(0, self.n_output - 1)
        mask = (out_sims > 0.05) & (self.output_layer.labels >= 0).unsqueeze(0) & (self.output_layer.labels < self.n_output).unsqueeze(0)
        masked_sims = out_sims * mask.float()
        label_expanded = valid_labels.unsqueeze(0).expand(batch, -1)
        votes.scatter_add_(1, label_expanded, masked_sims)

        return votes.argmax(dim=1).cpu().numpy()

    def learn_batch(self, X_np: np.ndarray, y_np: np.ndarray, batch_size: int = 256):
        """Batch learn with mini-batch processing.

        For each mini-batch:
        1. Batch compute similarities for all samples (GPU parallel)
        2. Classify samples into merge/grow/partial update
        3. Batch update merged neurons
        4. Grow new neurons for novel samples (sequential but rare)
        """
        X_all = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        y_all = torch.tensor(y_np, dtype=torch.long, device=self.device)
        n = X_all.shape[0]

        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            X = X_all[batch_start:batch_end]
            y = y_all[batch_start:batch_end]
            bs = X.shape[0]
            self.step += bs

            h = X  # (bs, n_dim)
            for layer in self.layers:
                # Batch similarities: (bs, n_neurons)
                sims = layer.similarities_batch(h)
                best_sims, best_idxs = sims.max(dim=1)  # (bs,), (bs,)

                # Classify samples
                merge_mask = best_sims >= self.merge_threshold
                grow_mask = (best_sims < self.split_threshold) & ((1.0 - best_sims) > 0.02)
                partial_mask = ~merge_mask & ~grow_mask

                # Batch update for merged samples
                if merge_mask.any():
                    m_idx = best_idxs[merge_mask]  # neuron indices
                    m_h = h[merge_mask]  # input vectors
                    m_labels = y[merge_mask]
                    # Update each winning neuron (some may win multiple times)
                    for j in range(m_idx.shape[0]):
                        ni = int(m_idx[j])
                        layer.W[ni] += self.lr * (m_h[j] - layer.W[ni])
                        layer.fire_counts[ni] += 1
                        layer.last_fire[ni] = self.step
                        layer.labels[ni] = int(m_labels[j])

                # Partial update
                if partial_mask.any():
                    p_idx = best_idxs[partial_mask]
                    p_h = h[partial_mask]
                    for j in range(p_idx.shape[0]):
                        ni = int(p_idx[j])
                        layer.W[ni] += (self.lr * 0.25) * (p_h[j] - layer.W[ni])
                        layer.fire_counts[ni] += 1
                        layer.last_fire[ni] = self.step

                # Grow for novel samples (capped per batch to prevent explosion)
                if grow_mask.any() and layer.size < self.max_neurons_per_layer:
                    g_h = h[grow_mask]
                    g_labels = y[grow_mask]
                    # Limit growth: max 10 new neurons per batch per layer
                    n_grow = min(g_h.shape[0], 10, self.max_neurons_per_layer - layer.size)
                    # Pick the most novel (lowest best_sim)
                    g_sims = best_sims[grow_mask]
                    _, novelty_order = g_sims.sort()
                    for j in range(n_grow):
                        k = int(novelty_order[j])
                        layer.grow(g_h[k], int(g_labels[k]), self.growth_noise)
                        self.n_width_grows += 1

                # Batch refine: propagate to next layer
                h = self._refine_batch(h, layer)

            # Output layer batch update
            out_sims = self.output_layer.similarities_batch(h)
            out_best = out_sims.argmax(dim=1)
            for j in range(bs):
                ni = int(out_best[j])
                self.output_layer.W[ni] += self.lr * (h[j] - self.output_layer.W[ni])
                self.output_layer.labels[ni] = int(y[j])

            # Depth + prune checks (once per batch, not per sample)
            if self.step % (self.depth_check_interval * batch_size) < batch_size:
                # Quick check on last sample
                pred = int(self.output_layer._cosine_sim(h[-1]).argmax())
                self._check_depth_growth(pred == int(y[-1]))

            if self.step % 500 < batch_size:
                self._prune()

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
