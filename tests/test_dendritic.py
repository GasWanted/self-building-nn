"""Tests for DendriticNeuron."""

import numpy as np
from src.neurons.dendritic import DendriticNeuron


class TestDendriticNeuron:
    def test_create(self):
        n = DendriticNeuron(10, n_dendrites=4, label=3)
        assert n.n_dim == 10
        assert n.n_dendrites == 4
        assert n.label == 3

    def test_similarity_best_dendrite(self):
        n = DendriticNeuron(10, n_dendrites=2)
        n.dendrite_weights[0] = np.ones(10)
        n.dendrite_weights[1] = -np.ones(10)
        sim = n.similarity(np.ones(10))
        assert sim > 0.9

    def test_activate_positive(self):
        n = DendriticNeuron(10, n_dendrites=4)
        n.dendrite_weights[0] = np.ones(10)
        assert n.activate(np.ones(10)) > 0

    def test_update_only_best_dendrite(self):
        n = DendriticNeuron(10, n_dendrites=2)
        n.dendrite_weights[0] = np.ones(10)
        n.dendrite_weights[1] = -np.ones(10)
        w1_before = n.dendrite_weights[1].copy()
        n.similarity(np.ones(10))  # sets _best_dendrite to 0
        n.update(np.ones(10), 0.1)
        np.testing.assert_array_almost_equal(n.dendrite_weights[1], w1_before)

    def test_copy_breaks_symmetry(self):
        n = DendriticNeuron(10, n_dendrites=4, label=5)
        c = n.copy(noise=0.1)
        assert c.label == n.label
        assert c.n_dendrites == n.n_dendrites
        assert not np.allclose(c.dendrite_weights[0], n.dendrite_weights[0])

    def test_get_weights_shape(self):
        n = DendriticNeuron(10, n_dendrites=2)
        w = n.get_weights()
        assert w.shape == (10,)
