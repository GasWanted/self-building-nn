"""Theta-gamma temporal buffer — sequence encoding without positional embeddings.

A 7-slot circular buffer (f_gamma/f_theta = 40Hz/8Hz = 5, +/-2).
Items pushed sequentially, bigram transitions accumulated.
Position = phase within theta cycle.
"""


class ThetaBuffer:
    """Temporal sequence encoder using bigram co-occurrence."""

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.buf: list[tuple] = []  # (vector, label)
        self.bigrams: dict[tuple[int, int], int] = {}

    def push(self, z, label: int):
        """Add an item to the buffer, recording the bigram transition."""
        if self.buf:
            prev_label = self.buf[-1][1]
            key = (prev_label, label)
            self.bigrams[key] = self.bigrams.get(key, 0) + 1

        self.buf.append((z, label))
        if len(self.buf) > self.capacity:
            self.buf.pop(0)

    def next(self, label: int) -> int | None:
        """Predict the next label given the current one."""
        candidates = {
            k[1]: v for k, v in self.bigrams.items() if k[0] == label
        }
        if not candidates:
            return None
        return max(candidates, key=candidates.get)

    def complete(self, prefix: list[int], n: int = 1) -> list[int]:
        """Extend prefix by n steps using bigram predictions."""
        seq = list(prefix)
        for _ in range(n):
            nxt = self.next(seq[-1])
            if nxt is None:
                break
            seq.append(nxt)
        return seq

    def state(self) -> dict:
        return {
            "capacity": self.capacity,
            "buffer_size": len(self.buf),
            "n_bigrams": len(self.bigrams),
        }
