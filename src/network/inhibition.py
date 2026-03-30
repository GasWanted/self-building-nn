"""Lateral inhibition: top-k winners suppress the rest."""

import numpy as np


def apply_inhibition(
    activations: np.ndarray,
    winner_fraction: float = 0.3,
    inhibition_factor: float = 0.1,
) -> np.ndarray:
    """Apply lateral inhibition to activation vector.

    Top k neurons keep full activation. Rest are multiplied by inhibition_factor.
    k = max(1, int(len(activations) * winner_fraction))
    """
    n = len(activations)
    if n <= 1:
        return activations.copy()

    result = activations.copy()
    k = max(1, int(n * winner_fraction))

    sorted_indices = np.argsort(activations)[::-1]
    winners = set(sorted_indices[:k].tolist())

    for i in range(n):
        if i not in winners:
            result[i] *= inhibition_factor

    return result
