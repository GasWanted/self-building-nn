"""Tests for FieldNetwork — full self-building spatial pipeline."""

import torch
import pytest
from src.field.field_network import FieldNetwork


class TestFieldNetwork:
    def test_create(self):
        net = FieldNetwork()
        assert net.n_layers == 1
        assert net.readout is not None

    def test_forward_returns_features(self):
        net = FieldNetwork()
        image = torch.randn(1, 1, 28, 28, device=net.device)
        features = net.forward(image)
        assert features.dim() == 1
        assert features.shape[0] > 0

    def test_predict(self):
        net = FieldNetwork()
        image = torch.randn(1, 1, 28, 28, device=net.device)
        pred = net.predict(image)
        assert 0 <= pred < 10

    def test_predict_batch(self):
        net = FieldNetwork()
        images = torch.randn(8, 1, 28, 28, device=net.device)
        preds = net.predict_batch(images)
        assert preds.shape == (8,)

    def test_learn_step(self):
        net = FieldNetwork()
        image = torch.randn(1, 1, 28, 28, device=net.device)
        net.learn(image, label=5)
        assert net.step == 1

    def test_train_batch(self):
        net = FieldNetwork()
        images = torch.randn(16, 1, 28, 28, device=net.device)
        labels = torch.randint(0, 10, (16,), device=net.device)
        net.train_batch(images, labels, batch_size=8)
        assert net.step > 0

    def test_growth_happens(self):
        net = FieldNetwork()
        initial = net.total_neurons()
        images = torch.randn(50, 1, 28, 28, device=net.device)
        labels = torch.randint(0, 10, (50,), device=net.device)
        net.train_batch(images, labels)
        # Should have at least as many neurons (may grow)
        assert net.total_neurons() >= initial

    def test_topology(self):
        net = FieldNetwork()
        topo = net.topology()
        assert "layers" in topo
        assert "total_neurons" in topo
        assert "per_layer" in topo
