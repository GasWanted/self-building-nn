"""Information signal — triggers depth growth when adjacent layers are too correlated."""

import numpy as np


def should_grow_depth(
    acts_a: np.ndarray, acts_b: np.ndarray, threshold: float = 0.90
) -> bool:
    """Return True if a new layer should be inserted between layers A and B.

    High correlation between adjacent layer activations means the second
    layer is mostly passing through what the first computed — it's not
    adding representational power. Inserting a duplicate layer gives
    the network more capacity to learn a transformation.

    Uses Pearson correlation. Returns False if either layer has constant
    activations (can't compute correlation).
    """
    if len(acts_a) < 2 or len(acts_b) < 2:
        return False

    # Handle size mismatch: compare what we can
    min_len = min(len(acts_a), len(acts_b))
    a = acts_a[:min_len]
    b = acts_b[:min_len]

    std_a = np.std(a)
    std_b = np.std(b)

    if std_a < 1e-9 or std_b < 1e-9:
        # One layer has constant activation — it IS a bottleneck,
        # but depth won't help. Width growth should handle this.
        return False

    corr = float(np.corrcoef(a, b)[0, 1])
    return abs(corr) > threshold
