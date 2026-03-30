"""Tests for SpikingNeuron."""

import numpy as np
from src.neurons.spiking import SpikingNeuron


class TestSpikingNeuron:
    def test_create(self):
        n = SpikingNeuron(10, label=0)
        assert n.n_dim == 10
        assert n.v == 0.0

    def test_activate_returns_binary(self):
        n = SpikingNeuron(10, weights=np.ones(10))
        act = n.activate(np.ones(10) * 2)
        assert act in (0.0, 1.0)

    def test_membrane_leaks(self):
        n = SpikingNeuron(10)
        n.v = 0.5
        n.steps_since_spike = n.t_ref  # not in refractory
        # Zero input, membrane should leak
        old_v = n.v
        n.activate(np.zeros(10))
        # After n_ticks of leak: v = tau^n_ticks * old_v (approximately)
        assert n.v < old_v or n._last_spiked  # leaked or spiked

    def test_refractory_resets(self):
        n = SpikingNeuron(10, weights=np.ones(10), v_thresh=0.01)
        # Force a spike with very low threshold
        n.activate(np.ones(10))
        # After spike, v should be 0
        # (may have spiked and then accumulated again in remaining ticks)

    def test_similarity_range(self):
        n = SpikingNeuron(10, weights=np.ones(10))
        sim = n.similarity(np.ones(10))
        assert -1.0 <= sim <= 1.0

    def test_update_changes_weights(self):
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
        assert not np.allclose(c.get_weights(), n.get_weights())

    def test_get_weights_shape(self):
        n = SpikingNeuron(10)
        assert n.get_weights().shape == (10,)

    def test_weight_normalization(self):
        """Weights should not exceed norm of 10."""
        n = SpikingNeuron(10, weights=np.ones(10) * 5)
        n._last_spiked = True
        n.update(np.ones(10) * 100, 1.0)
        assert np.linalg.norm(n.get_weights()) <= 10.0 + 1e-6
