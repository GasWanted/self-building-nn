"""Same-space propagation: iterative refinement in shared feature space."""

import numpy as np
from src.network.layer import Layer


def refine(x: np.ndarray, layer: Layer, lr_refine: float = 0.3) -> np.ndarray:
    """Refine representation x through layer.

    Output = x + lr_refine * weighted_shift_toward_prototypes.
    All layers operate in the same d-dimensional space.
    If no neurons activate, returns x unchanged (passthrough).
    """
    acts = layer.forward(x)
    act_sum = acts.sum()

    if act_sum < 1e-9:
        return x.copy()

    weights = np.array([n.get_weights() for n in layer.neurons])
    # Weighted shift: each neuron pulls x toward its prototype
    shifts = weights - x[None, :]  # (n_neurons, d)
    weighted_shift = (acts[:, None] * shifts).sum(axis=0) / act_sum

    return x + lr_refine * weighted_shift
