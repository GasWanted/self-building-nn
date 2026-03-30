"""Sleep consolidation — interleaved replay from hippocampal buffer to network."""

import numpy as np


def sleep_consolidate(
    network,
    hippocampus: list[tuple[np.ndarray, int]],
    n_steps: int = 1000,
    rng: np.random.Generator = None,
):
    """Replay hippocampal episodes to the network in interleaved order.

    Interleaving is critical: sequential replay (all class A, then all class B)
    causes catastrophic forgetting. Interleaved replay does not, because
    gradient contributions from different classes cancel in expectation.

    After replay, tightens the network's merge threshold (ACh decay).
    """
    if not hippocampus:
        return

    if rng is None:
        rng = np.random.default_rng()

    for _ in range(n_steps):
        z, label = hippocampus[rng.integers(len(hippocampus))]
        network.learn(z, label)

    network.tighten()
