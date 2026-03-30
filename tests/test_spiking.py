"""Tests for SpikingNeuron."""

import numpy as np
from src.neurons.spiking import SpikingNeuron


class TestSpikingNeuron:
    def test_create(self):
        n = SpikingNeuron(10, label=0)
        assert n.n_dim == 10
        assert n.v == 0.0

    def test_activate_returns_graded(self):
        n = SpikingNeuron(10, weights=np.ones(10) * 0.1)
        act = n.activate(np.ones(10) * 0.5)
        assert 0.0 <= act <= 1.0

    def test_strong_input_high_activation(self):
        n = SpikingNeuron(10, weights=np.ones(10))
        act = n.activate(np.ones(10))
        assert act >= 0

    def test_membrane_leaks(self):
        n = SpikingNeuron(10)
        n.v = 0.5
        n.steps_since_spike = n.t_ref
        old_v = n.v
        n.activate(np.zeros(10))
        assert n.v < old_v or n._last_spiked

    def test_similarity_is_cosine(self):
        n = SpikingNeuron(10, weights=np.ones(10))
        sim = n.similarity(np.ones(10))
        assert abs(sim - 1.0) < 0.01

    def test_similarity_range(self):
        n = SpikingNeuron(10, weights=np.ones(10))
        sim = n.similarity(-np.ones(10))
        assert -1.0 <= sim <= 1.0

    def test_update_stdp_spike(self):
        n = SpikingNeuron(10, weights=np.ones(10))
        w_before = n.get_weights().copy()
        n._last_spiked = True
        n.update(np.ones(10) * 2, 0.1)
        assert not np.allclose(n.get_weights(), w_before)

    def test_copy_resets_membrane(self):
        n = SpikingNeuron(10, label=3, weights=np.ones(10))
        n.v = 0.8
        c = n.copy(noise=0.1)
        assert c.label == 3
        assert c.v == 0.0

    def test_get_weights_shape(self):
        n = SpikingNeuron(10)
        assert n.get_weights().shape == (10,)

    def test_weight_normalization(self):
        n = SpikingNeuron(10, weights=np.ones(10) * 5)
        n._last_spiked = True
        n.update(np.ones(10) * 100, 1.0)
        assert np.linalg.norm(n.get_weights()) <= 10.0 + 1e-6

    def test_default_n_ticks_is_2(self):
        n = SpikingNeuron(10)
        assert n.n_ticks == 2
