"""Skip connection tracking: formation, pruning, input blending."""

import numpy as np


class ConnectionTracker:
    """Tracks which layers connect to which.

    Each layer has a dict of {source_layer_idx: weight}.
    Layer i always has adjacent connection to i-1 (weight=1.0, not removable).
    Skip connections to earlier layers can be added/removed.
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.connections: list[dict[int, float]] = []
        for i in range(n_layers):
            if i == 0:
                self.connections.append({})
            else:
                self.connections.append({i - 1: 1.0})

    def get_sources(self, layer_idx: int) -> dict[int, float]:
        if layer_idx < 0 or layer_idx >= self.n_layers:
            return {}
        return dict(self.connections[layer_idx])

    def add_connection(self, target: int, source: int, weight: float = 0.5):
        if target >= self.n_layers or source >= self.n_layers:
            return
        if source >= target:
            return
        self.connections[target][source] = weight

    def remove_connection(self, target: int, source: int):
        if target >= self.n_layers:
            return
        if source == target - 1:
            return  # don't remove adjacent
        self.connections[target].pop(source, None)

    def blend_inputs(self, target: int, representations: dict[int, np.ndarray]) -> np.ndarray:
        """Blend inputs from all source layers using connection weights."""
        sources = self.connections[target]
        if not sources:
            return representations.get(-1, np.zeros(1))

        total_weight = sum(sources.values())
        if total_weight < 1e-9:
            adj = target - 1
            return representations.get(adj, np.zeros(1))

        result = None
        for src_idx, weight in sources.items():
            if src_idx in representations:
                contrib = weight * representations[src_idx]
                result = contrib if result is None else result + contrib

        if result is None:
            adj = target - 1
            return representations.get(adj, np.zeros(1))

        return result / total_weight

    def rebuild(self, n_layers: int):
        """Rebuild tracker for a new number of layers (after growth/prune)."""
        self.n_layers = n_layers
        self.connections = []
        for i in range(n_layers):
            if i == 0:
                self.connections.append({})
            else:
                self.connections.append({i - 1: 1.0})
