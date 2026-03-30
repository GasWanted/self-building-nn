"""Tests for PredictiveCodingNeuron."""

import numpy as np
from src.neurons.predictive import PredictiveCodingNeuron


class TestPredictiveCodingNeuron:
    def test_create(self):
        n = PredictiveCodingNeuron(10, label=2)
        assert n.n_dim == 10
        assert n.label == 2

    def test_high_activation_when_matched(self):
        n = PredictiveCodingNeuron(10, weights=np.ones(10))
        act = n.activate(np.ones(10))
        assert act > 0.5

    def test_low_activation_when_mismatched(self):
        n = PredictiveCodingNeuron(10, weights=np.ones(10))
        act = n.activate(-np.ones(10) * 10)
        assert act < 0.5

    def test_similarity_high_when_close(self):
        n = PredictiveCodingNeuron(10, weights=np.ones(10))
        sim = n.similarity(np.ones(10))
        assert sim > 0.8

    def test_update_reduces_error(self):
        n = PredictiveCodingNeuron(10, weights=np.zeros(10))
        target = np.ones(10)
        err_before = np.linalg.norm(n.get_weights() - target)
        n.update(target, 0.1)
        err_after = np.linalg.norm(n.get_weights() - target)
        assert err_after < err_before

    def test_copy_breaks_symmetry(self):
        n = PredictiveCodingNeuron(10, label=1, weights=np.ones(10))
        c = n.copy(noise=0.1)
        assert c.label == 1
        assert not np.allclose(c.get_weights(), n.get_weights())

    def test_get_weights_shape(self):
        n = PredictiveCodingNeuron(10)
        assert n.get_weights().shape == (10,)
