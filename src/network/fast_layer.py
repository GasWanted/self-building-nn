"""GPU-accelerated layer using torch tensors."""

import torch
import numpy as np


class FastLayer:
    """A layer that stores all neuron weights as a single GPU tensor.

    All operations are vectorized — no Python loops over neurons.
    """

    _uid = 0

    def __init__(self, n_dim: int, n_neurons: int = 8, device: str = "cuda",
                 winner_fraction: float = 0.3, inhibition_factor: float = 0.1):
        self.id = FastLayer._uid
        FastLayer._uid += 1
        self.n_dim = n_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Weight matrix: (n_neurons, n_dim)
        self.W = torch.randn(n_neurons, n_dim, device=self.device) * 0.1
        self.labels = torch.full((n_neurons,), -1, dtype=torch.long, device=self.device)
        self.fire_counts = torch.zeros(n_neurons, dtype=torch.long, device=self.device)
        self.last_fire = torch.zeros(n_neurons, dtype=torch.long, device=self.device)
        self.ages = torch.zeros(n_neurons, dtype=torch.long, device=self.device)

        self.winner_fraction = winner_fraction
        self.inhibition_factor = inhibition_factor
        self.step = 0

        # Tracked for propagation / depth signal
        self.last_input = None
        self.last_output = None
        self.last_activations = None

    @property
    def size(self) -> int:
        return self.W.shape[0]

    def _cosine_sim(self, x: torch.Tensor) -> torch.Tensor:
        """Batch cosine similarity: (n_neurons,)"""
        # x: (n_dim,), W: (n_neurons, n_dim)
        w_norms = self.W.norm(dim=1).clamp(min=1e-9)  # (n_neurons,)
        x_norm = x.norm().clamp(min=1e-9)
        return (self.W @ x) / (w_norms * x_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute activations with lateral inhibition."""
        self.step += 1
        sims = self._cosine_sim(x)
        acts = torch.relu(sims)

        # Lateral inhibition
        if self.size > 1:
            k = max(1, int(self.size * self.winner_fraction))
            if k < self.size:
                topk_vals, topk_idx = torch.topk(acts, k)
                threshold = topk_vals[-1]
                mask = acts >= threshold
                acts = torch.where(mask, acts, acts * self.inhibition_factor)

        self.last_activations = acts

        # Track firing
        self.ages += 1
        fired = acts > 0
        self.fire_counts += fired.long()
        self.last_fire = torch.where(fired, torch.tensor(self.step, device=self.device), self.last_fire)

        return acts

    def similarities(self, x: torch.Tensor) -> torch.Tensor:
        """Cosine similarities without side effects."""
        return self._cosine_sim(x)

    def similarities_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Batch cosine similarity: X is (batch, n_dim) -> (batch, n_neurons)."""
        w_norms = self.W.norm(dim=1).clamp(min=1e-9)  # (n_neurons,)
        x_norms = X.norm(dim=1).clamp(min=1e-9)  # (batch,)
        # (batch, n_dim) @ (n_dim, n_neurons) -> (batch, n_neurons)
        dots = X @ self.W.t()
        return dots / (x_norms.unsqueeze(1) * w_norms.unsqueeze(0))

    def best_match(self, x: torch.Tensor) -> tuple[int, float]:
        sims = self._cosine_sim(x)
        idx = int(sims.argmax())
        return idx, float(sims[idx])

    def grow(self, x: torch.Tensor, label: int, noise: float = 0.005):
        """Add a neuron positioned at x."""
        new_w = x.unsqueeze(0) + torch.randn(1, self.n_dim, device=self.device) * noise
        self.W = torch.cat([self.W, new_w], dim=0)
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])
        self.fire_counts = torch.cat([self.fire_counts, torch.zeros(1, dtype=torch.long, device=self.device)])
        self.last_fire = torch.cat([self.last_fire, torch.zeros(1, dtype=torch.long, device=self.device)])
        self.ages = torch.cat([self.ages, torch.zeros(1, dtype=torch.long, device=self.device)])

    def prune(self, step: int, prune_age: int, prune_window: int, min_size: int):
        """Remove dead neurons."""
        if self.size <= min_size:
            all_dead = ((self.fire_counts == 0) & (self.ages > prune_age)).all()
            if all_dead:
                # Clear everything
                self.W = self.W[:0]
                self.labels = self.labels[:0]
                self.fire_counts = self.fire_counts[:0]
                self.last_fire = self.last_fire[:0]
                self.ages = self.ages[:0]
            return

        # Find neurons to keep
        never_fired = (self.fire_counts == 0) & (self.ages > prune_age)
        stale = (self.fire_counts > 0) & ((step - self.last_fire) > prune_window)
        remove = never_fired | stale

        # Don't prune below min_size
        n_remove = remove.sum().item()
        max_removable = self.size - min_size
        if n_remove > max_removable:
            # Keep the most recently fired among those marked for removal
            remove_indices = remove.nonzero().squeeze(-1)
            keep_count = n_remove - max_removable
            # Keep the ones with highest last_fire
            _, keep_order = self.last_fire[remove_indices].sort(descending=True)
            keep_these = remove_indices[keep_order[:keep_count]]
            remove[keep_these] = False

        keep = ~remove
        self.W = self.W[keep]
        self.labels = self.labels[keep]
        self.fire_counts = self.fire_counts[keep]
        self.last_fire = self.last_fire[keep]
        self.ages = self.ages[keep]

    def duplicate(self, noise: float = 0.005):
        """Copy this layer (for depth growth)."""
        new = FastLayer(self.n_dim, self.size, str(self.device),
                        self.winner_fraction, self.inhibition_factor)
        new.W = self.W.clone() + torch.randn_like(self.W) * noise
        new.labels = self.labels.clone()
        return new

    def update_best(self, x: torch.Tensor, label: int, lr: float):
        """Update the best-matching neuron."""
        idx, sim = self.best_match(x)
        self.W[idx] += lr * (x - self.W[idx])
        self.labels[idx] = label

    def dead_count(self) -> int:
        return int((self.fire_counts == 0).sum())

    def topology_size(self) -> int:
        return self.size
