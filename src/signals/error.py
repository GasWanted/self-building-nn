"""Error signal — triggers width growth when a layer lacks coverage."""


def should_grow_width(best_similarity: float, split_threshold: float) -> bool:
    """Return True if the layer needs a new neuron.

    Triggered when the best-matching neuron's similarity is below
    the split threshold — the layer doesn't have a good match for
    this input pattern.

    The information gain (1 - best_similarity) must exceed a minimum
    to avoid growing on noise.
    """
    if best_similarity >= split_threshold:
        return False
    info_gain = 1.0 - best_similarity
    return info_gain > 0.02  # minimum novelty to justify a new neuron


def should_grow_depth_v2(
    transformation_ratio: float,
    stagnation_threshold: float,
    wrong_count: float,
    patience: int,
) -> bool:
    """Return True if a layer needs depth growth.

    Triggered when:
    1. Layer is stagnant (transformation_ratio < stagnation_threshold)
    2. AND wrong predictions have accumulated past patience
    """
    is_stagnant = transformation_ratio < stagnation_threshold
    enough_errors = wrong_count > patience
    return is_stagnant and enough_errors
