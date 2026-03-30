"""Smoke tests for benchmark suite — runs on tiny data to verify wiring."""

import numpy as np
import pytest
from src.neurons import PrototypeNeuron
from src.network.network import Network
from src.infrastructure.da_selection import da_select
from src.infrastructure.theta import ThetaBuffer


def make_factory():
    def factory(n_dim, label=-1):
        return PrototypeNeuron(n_dim, label=label)
    return factory


class TestDASelection:
    def test_selects_most_central(self):
        rng = np.random.default_rng(42)
        mean = np.ones(10)
        candidates = mean + rng.standard_normal((5, 10)) * 0.1
        # Make candidate 2 almost exactly the mean
        candidates[2] = mean + rng.standard_normal(10) * 0.001
        best, scores, idx = da_select(candidates)
        assert scores[2] > 0.99  # should be highest

    def test_single_candidate(self):
        c = np.random.randn(1, 10)
        best, scores, idx = da_select(c)
        assert idx == 0


class TestThetaBuffer:
    def test_sequence_completion(self):
        theta = ThetaBuffer()
        # Train sequence 0 -> 1 -> 2 -> 3
        for label in [0, 1, 2, 3]:
            theta.push(np.zeros(10), label)
        result = theta.complete([0, 1, 2], n=1)
        assert result == [0, 1, 2, 3]

    def test_unknown_prefix(self):
        theta = ThetaBuffer()
        theta.push(np.zeros(10), 0)
        result = theta.complete([9], n=1)
        assert result == [9]  # can't extend
