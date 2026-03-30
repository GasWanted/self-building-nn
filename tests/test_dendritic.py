"""Tests for DendriticNeuron."""

import numpy as np
from src.neurons.dendritic import DendriticNeuron


class TestDendriticNeuron:
    def test_create_starts_with_one_dendrite(self):
        n = DendriticNeuron(10, n_dendrites=4, label=3)
        assert n.n_dim == 10
        assert n.max_dendrites == 4
        assert len(n.dendrite_weights) == 1
        assert n.label == 3

    def test_similarity_uses_best_dendrite(self):
        n = DendriticNeuron(10, n_dendrites=2)
        n.dendrite_weights[0] = np.ones(10)
        sim = n.similarity(np.ones(10))
        assert sim > 0.9

    def test_activate_positive(self):
        n = DendriticNeuron(10, n_dendrites=4)
        n.dendrite_weights[0] = np.ones(10)
        assert n.activate(np.ones(10)) > 0

    def test_update_grows_dendrite_on_mismatch(self):
        n = DendriticNeuron(10, n_dendrites=4)
        n.dendrite_weights[0] = np.ones(10)
        assert len(n.dendrite_weights) == 1
        different = -np.ones(10)
        n.similarity(different)
        n.update(different, 0.1)
        assert len(n.dendrite_weights) == 2

    def test_update_doesnt_grow_past_max(self):
        n = DendriticNeuron(10, n_dendrites=2)
        n.dendrite_weights = [np.ones(10), -np.ones(10)]
        orthogonal = np.zeros(10); orthogonal[0] = 1.0
        n.similarity(orthogonal)
        n.update(orthogonal, 0.1)
        assert len(n.dendrite_weights) == 2

    def test_update_absorbs_when_matching(self):
        n = DendriticNeuron(10, n_dendrites=4)
        n.dendrite_weights[0] = np.ones(10)
        w_before = n.dendrite_weights[0].copy()
        close_input = np.ones(10) * 1.1
        n.similarity(close_input)
        n.update(close_input, 0.1)
        assert len(n.dendrite_weights) == 1
        assert not np.allclose(n.dendrite_weights[0], w_before)

    def test_copy_preserves_all_dendrites(self):
        n = DendriticNeuron(10, n_dendrites=4, label=5)
        n.dendrite_weights = [np.ones(10), -np.ones(10)]
        c = n.copy(noise=0.1)
        assert c.label == n.label
        assert c.max_dendrites == n.max_dendrites
        assert len(c.dendrite_weights) == 2
        assert not np.allclose(c.dendrite_weights[0], n.dendrite_weights[0])

    def test_get_weights_shape(self):
        n = DendriticNeuron(10, n_dendrites=2)
        w = n.get_weights()
        assert w.shape == (10,)
