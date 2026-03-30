"""DA Prototype Selection.

Theorem: Under N(mu_c, I) in whitened feature space,
    argmax_i cosine_sim(z_i, z_bar) = argmax_i p(z_i | mu_hat_c)

The sample most similar to the group mean is the maximum-likelihood
representative of the class. Oracle gap scales O(1/sqrt(K)).
"""

import numpy as np


def da_select(candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Select the most prototypical candidate using group-mean DA.

    Args:
        candidates: (K, d) array of K candidate feature vectors.

    Returns:
        best: the selected prototype vector
        scores: DA scores for all candidates
        best_idx: index of the selected candidate
    """
    if len(candidates) == 1:
        return candidates[0], np.array([1.0]), 0

    mean = candidates.mean(axis=0)
    norm_mean = np.linalg.norm(mean)

    if norm_mean < 1e-9:
        return candidates[0], np.zeros(len(candidates)), 0

    scores = np.array([
        float(np.dot(z, mean) / (np.linalg.norm(z) * norm_mean + 1e-9))
        for z in candidates
    ])

    best_idx = int(np.argmax(scores))
    return candidates[best_idx], scores, best_idx
