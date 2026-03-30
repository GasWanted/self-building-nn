"""Tests for neuron types."""

import numpy as np
import pytest
from src.neurons import PrototypeNeuron, PerceptronNeuron


class TestPrototypeNeuron:
    def test_create(self):
        n = PrototypeNeuron(10, label=3)
        assert n.n_dim == 10
        assert n.label == 3
        assert n.n_fired == 0

    def test_similarity_self(self):
        n = PrototypeNeuron(10, weights=np.ones(10))
        assert n.similarity(np.ones(10)) == pytest.approx(1.0, abs=0.01)

    def test_similarity_orthogonal(self):
        w = np.zeros(10)
        w[0] = 1.0
        n = PrototypeNeuron(10, weights=w)
        x = np.zeros(10)
        x[1] = 1.0
        assert n.similarity(x) == pytest.approx(0.0, abs=0.01)

    def test_activate_positive(self):
        n = PrototypeNeuron(10, weights=np.ones(10))
        assert n.activate(np.ones(10)) > 0

    def test_update_moves_toward(self):
        n = PrototypeNeuron(10, weights=np.zeros(10))
        target = np.ones(10)
        sim_before = n.similarity(target)
        n.update(target, 0.1)
        sim_after = n.similarity(target)
        assert sim_after > sim_before

    def test_copy_breaks_symmetry(self):
        n = PrototypeNeuron(10, label=5, weights=np.ones(10))
        c = n.copy(noise=0.1)
        assert c.label == n.label
        assert not np.allclose(c.w, n.w)

    def test_fire_tracking(self):
        n = PrototypeNeuron(10)
        n.fire(step=42)
        assert n.n_fired == 1
        assert n.last_fire == 42


class TestPerceptronNeuron:
    def test_create(self):
        n = PerceptronNeuron(10, label=0)
        assert n.n_dim == 10

    def test_activate_relu(self):
        n = PerceptronNeuron(10, weights=np.ones(10))
        assert n.activate(np.ones(10)) > 0
        assert n.activate(-np.ones(10) * 100) == 0.0

    def test_copy(self):
        n = PerceptronNeuron(10, label=3, weights=np.ones(10))
        c = n.copy(noise=0.1)
        assert c.label == 3
        assert not np.allclose(c.w, n.w)


class TestNeuronInterface:
    def test_prototype_get_weights(self):
        n = PrototypeNeuron(10, weights=np.ones(10))
        w = n.get_weights()
        assert w.shape == (10,)
        assert np.allclose(w, np.ones(10))

    def test_perceptron_get_weights(self):
        n = PerceptronNeuron(10, weights=np.ones(10))
        w = n.get_weights()
        assert w.shape == (10,)
        assert np.allclose(w, np.ones(10))

    def test_get_weights_returns_copy(self):
        n = PrototypeNeuron(10, weights=np.ones(10))
        w = n.get_weights()
        w[0] = 999
        assert n.get_weights()[0] != 999
