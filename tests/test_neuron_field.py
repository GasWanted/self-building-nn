"""Tests for NeuronField — per-layer pre-allocated tensor ops."""

import torch
import pytest
from src.field.neuron_field import NeuronField


class TestNeuronField:
    def test_create(self):
        nf = NeuronField(patch_dim=25, capacity=128)
        assert nf.n_alive == 0
        assert nf.capacity == 128

    def test_add_neuron(self):
        nf = NeuronField(patch_dim=25, capacity=128)
        w = torch.randn(25)
        nf.add_neuron(w, pos_x=3, pos_y=5, label=7)
        assert nf.n_alive == 1
        assert nf.labels[0] == 7

    def test_add_multiple(self):
        nf = NeuronField(patch_dim=25, capacity=128)
        for i in range(10):
            nf.add_neuron(torch.randn(25), pos_x=i, pos_y=0, label=i % 10)
        assert nf.n_alive == 10

    def test_batch_cosine_sim(self):
        nf = NeuronField(patch_dim=25, capacity=128)
        w = torch.ones(25)
        nf.add_neuron(w, pos_x=0, pos_y=0, label=0)
        patches = torch.ones(1, 25, device=nf.device)
        indices = torch.tensor([0], device=nf.device)
        sims = nf.batch_cosine_sim(patches, indices)
        assert sims.shape == (1,)
        assert abs(float(sims[0]) - 1.0) < 0.01

    def test_batch_cosine_sim_all(self):
        nf = NeuronField(patch_dim=25, capacity=128)
        nf.add_neuron(torch.ones(25), pos_x=0, pos_y=0, label=0)
        nf.add_neuron(-torch.ones(25), pos_x=1, pos_y=0, label=1)
        patches = torch.ones(3, 25, device=nf.device)
        sims = nf.batch_cosine_sim_all(patches)
        assert sims.shape == (2, 3)  # 2 alive neurons, 3 patches
        assert sims[0, 0] > 0.9  # first neuron matches
        assert sims[1, 0] < -0.9  # second neuron anti-matches

    def test_prune(self):
        nf = NeuronField(patch_dim=25, capacity=128)
        for i in range(5):
            nf.add_neuron(torch.randn(25), pos_x=i, pos_y=0, label=0)
        nf.ages[:5] = 5000
        nf.fire_counts[0] = 10
        nf.prune(prune_age=3000)
        assert nf.n_alive >= 1

    def test_grow_reuses_dead_slot(self):
        nf = NeuronField(patch_dim=25, capacity=4)
        for i in range(4):
            nf.add_neuron(torch.randn(25), pos_x=i, pos_y=0, label=0)
        assert nf.n_alive == 4
        nf.alive[2] = False
        nf.add_neuron(torch.randn(25), pos_x=10, pos_y=10, label=5)
        assert nf.n_alive == 4
        assert nf.pos_x[2] == 10

    def test_expand_capacity(self):
        nf = NeuronField(patch_dim=25, capacity=4)
        for i in range(5):
            nf.add_neuron(torch.randn(25), pos_x=i, pos_y=0, label=0)
        assert nf.capacity >= 5
        assert nf.n_alive == 5

    def test_add_neurons_batch(self):
        nf = NeuronField(patch_dim=25, capacity=128)
        W = torch.randn(5, 25)
        px = torch.arange(5)
        py = torch.zeros(5, dtype=torch.long)
        labels = torch.arange(5)
        nf.add_neurons_batch(W, px, py, labels)
        assert nf.n_alive == 5
