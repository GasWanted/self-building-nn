"""Tests for GPU-accelerated fast layer and network."""

import numpy as np
import torch
import pytest
from src.network.fast_layer import FastLayer
from src.network.fast_network import FastNetwork


class TestFastLayer:
    def test_create(self):
        layer = FastLayer(10, 5)
        assert layer.size == 5
        assert layer.W.shape == (5, 10)

    def test_cosine_sim_shape(self):
        layer = FastLayer(10, 5)
        x = torch.randn(10, device=layer.device)
        sims = layer.similarities(x)
        assert sims.shape == (5,)

    def test_forward_with_inhibition(self):
        layer = FastLayer(10, 5, winner_fraction=0.4, inhibition_factor=0.1)
        x = torch.randn(10, device=layer.device)
        acts = layer.forward(x)
        assert acts.shape == (5,)

    def test_grow(self):
        layer = FastLayer(10, 3)
        x = torch.randn(10, device=layer.device)
        layer.grow(x, label=5)
        assert layer.size == 4
        assert layer.labels[-1] == 5

    def test_prune_dead(self):
        layer = FastLayer(10, 5)
        layer.ages[:] = 5000
        layer.fire_counts[0] = 1
        layer.last_fire[0] = 4999
        layer.prune(step=5000, prune_age=3000, prune_window=3000, min_size=1)
        assert layer.size >= 1

    def test_duplicate(self):
        layer = FastLayer(10, 5)
        dup = layer.duplicate()
        assert dup.size == 5
        assert not torch.equal(dup.W, layer.W)  # noise added

    def test_best_match(self):
        layer = FastLayer(10, 3)
        layer.W[1] = torch.ones(10)
        idx, sim = layer.best_match(torch.ones(10, device=layer.device))
        assert idx == 1
        assert sim > 0.9


class TestFastNetwork:
    def test_create(self):
        net = FastNetwork(50)
        assert net.depth() == 2
        assert net.total_neurons() > 0

    def test_predict(self):
        net = FastNetwork(50)
        x = np.random.randn(50)
        pred = net.predict(x)
        assert 0 <= pred < 10

    def test_learn(self):
        net = FastNetwork(50)
        for i in range(100):
            net.learn(np.random.randn(50), i % 10)
        assert net.step == 100

    def test_growth(self):
        net = FastNetwork(50)
        initial = net.total_neurons()
        for i in range(500):
            net.learn(np.random.randn(50) * 3, i % 10)
        assert net.total_neurons() > initial

    def test_topology(self):
        net = FastNetwork(50)
        topo = net.topology()
        assert topo[0] == 50
        assert topo[-1] == 10

    def test_on_real_data(self):
        """Quick smoke test on real digits."""
        from src.data.mnist import load_small
        Z_tr, y_tr, Z_te, y_te, _ = load_small()
        n_in = Z_tr.shape[1]
        net = FastNetwork(n_in)
        for i in range(min(500, len(Z_tr))):
            net.learn(Z_tr[i], int(y_tr[i]))
        correct = sum(net.predict(Z_te[i]) == y_te[i] for i in range(100))
        assert correct > 15, f"Only {correct}/100"
