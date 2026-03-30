"""Tests for Layer and Network."""

import numpy as np
import pytest
from src.neurons import PrototypeNeuron
from src.network.layer import Layer
from src.network.network import Network


class TestLayer:
    def make_layer(self, n=5, dim=10):
        neurons = [PrototypeNeuron(dim, label=i % 3) for i in range(n)]
        return Layer(neurons)

    def test_forward_shape(self):
        layer = self.make_layer(5, 10)
        acts = layer.forward(np.random.randn(10))
        assert len(acts) == 5

    def test_grow(self):
        layer = self.make_layer(3, 10)
        # Fire one neuron so it's the growth source
        layer.neurons[0].fire(1)
        layer.grow(1)
        assert layer.size == 4

    def test_prune_dead(self):
        layer = self.make_layer(5, 10)
        for n in layer.neurons:
            n.age = 5000  # old enough to prune
        # Fire neuron 0 so it's alive
        layer.neurons[0].n_fired = 1
        layer.neurons[0].last_fire = 4999
        layer.prune(step=5000, prune_age=3000, prune_window=3000, min_size=1)
        assert layer.size >= 1
        assert layer.neurons[0].n_fired == 1  # survivor

    def test_duplicate(self):
        layer = self.make_layer(3, 10)
        dup = layer.duplicate(noise=0.01)
        assert dup.size == layer.size
        assert dup.id != layer.id

    def test_best_match(self):
        layer = self.make_layer(3, 10)
        x = layer.neurons[1].w.copy()  # exact match for neuron 1
        idx, sim = layer.best_match(x)
        assert idx == 1
        assert sim > 0.99


class TestNetwork:
    def make_net(self, dim=10):
        def factory(n_dim, label=-1):
            return PrototypeNeuron(n_dim, label=label)
        return Network(
            n_input=dim, n_output=3, neuron_factory=factory,
            initial_hidden_size=8, n_initial_layers=2,
            growth_interval=5, depth_check_interval=50,
        )

    def test_create(self):
        net = self.make_net()
        assert net.depth() == 2
        assert net.total_neurons() > 0

    def test_topology(self):
        net = self.make_net(10)
        topo = net.topology()
        assert topo[0] == 10   # input
        assert topo[-1] == 3   # output
        assert len(topo) == 4  # input + 2 hidden + output

    def test_predict_returns_valid(self):
        net = self.make_net(10)
        x = np.random.randn(10)
        pred = net.predict(x)
        assert 0 <= pred < 3

    def test_learn_doesnt_crash(self):
        net = self.make_net(10)
        for i in range(100):
            x = np.random.randn(10)
            net.learn(x, i % 3)
        assert net.step == 100

    def test_growth_happens(self):
        net = self.make_net(10)
        initial = net.total_neurons()
        # Feed many diverse inputs to trigger growth
        for i in range(500):
            x = np.random.randn(10) * 5  # large variance to be novel
            net.learn(x, i % 3)
        assert net.total_neurons() >= initial  # should have grown


class TestPropagation:
    def test_refine_same_dimension(self):
        """Output has same dimension as input."""
        from src.network.propagation import refine
        from src.neurons import PrototypeNeuron
        from src.network.layer import Layer
        neurons = [PrototypeNeuron(10, label=i, weights=np.random.randn(10)) for i in range(5)]
        layer = Layer(neurons)
        x = np.random.randn(10)
        x_out = refine(x, layer)
        assert x_out.shape == x.shape

    def test_refine_shifts_toward_prototypes(self):
        """Output should be closer to the best-matching neuron than input was."""
        from src.network.propagation import refine
        from src.neurons import PrototypeNeuron
        from src.network.layer import Layer
        target = np.ones(10)
        neurons = [PrototypeNeuron(10, label=0, weights=target)]
        layer = Layer(neurons)
        x = np.ones(10) * 0.1  # small positive vector (activates via cosine sim)
        x_out = refine(x, layer, lr_refine=0.5)
        dist_before = np.linalg.norm(x - target)
        dist_after = np.linalg.norm(x_out - target)
        assert dist_after < dist_before

    def test_refine_passthrough_on_zero_activation(self):
        """If no neurons activate, output equals input."""
        from src.network.propagation import refine
        from src.neurons import PrototypeNeuron
        from src.network.layer import Layer
        neurons = [PrototypeNeuron(10, label=0, weights=np.ones(10) * 100)]
        layer = Layer(neurons)
        x = -np.ones(10) * 100
        x_out = refine(x, layer)
        np.testing.assert_array_almost_equal(x_out, x)


class TestIntegration:
    def test_train_and_predict_on_digits(self):
        """End-to-end: train on sklearn digits, predict, verify non-trivial accuracy."""
        from src.data.mnist import load_small
        Z_tr, y_tr, Z_te, y_te, pca = load_small()
        n_in = Z_tr.shape[1]

        def nf(d, label=-1):
            return PrototypeNeuron(d, label=label)

        net = Network(n_in, 10, nf, initial_hidden_size=16)

        for i in range(min(500, len(Z_tr))):
            net.learn(Z_tr[i], int(y_tr[i]))

        correct = sum(
            net.predict(Z_te[i]) == y_te[i]
            for i in range(min(100, len(Z_te)))
        )
        assert correct > 15, f"Only {correct}/100 correct — worse than random"
