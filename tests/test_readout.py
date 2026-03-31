"""Tests for gradient-trained readout layer."""

import torch
import pytest
from src.field.readout import Readout


class TestReadout:
    def test_create(self):
        r = Readout(feature_dim=100, n_classes=10)
        assert r.feature_dim == 100
        assert r.n_classes == 10

    def test_forward_shape(self):
        r = Readout(feature_dim=100, n_classes=10)
        features = torch.randn(100, device=r.device)
        logits = r(features)
        assert logits.shape == (10,)

    def test_forward_batch(self):
        r = Readout(feature_dim=100, n_classes=10)
        features = torch.randn(32, 100, device=r.device)
        logits = r(features)
        assert logits.shape == (32, 10)

    def test_predict(self):
        r = Readout(feature_dim=100, n_classes=10)
        features = torch.randn(100, device=r.device)
        pred = r.predict(features)
        assert 0 <= pred < 10

    def test_predict_batch(self):
        r = Readout(feature_dim=100, n_classes=10)
        features = torch.randn(32, 100, device=r.device)
        preds = r.predict_batch(features)
        assert preds.shape == (32,)
        assert (preds >= 0).all() and (preds < 10).all()

    def test_train_step_reduces_loss(self):
        r = Readout(feature_dim=100, n_classes=10, lr=0.1)
        features = torch.randn(32, 100, device=r.device)
        labels = torch.randint(0, 10, (32,), device=r.device)
        loss1 = r.train_step(features, labels)
        loss2 = r.train_step(features, labels)
        assert loss2 < loss1

    def test_resize(self):
        r = Readout(feature_dim=100, n_classes=10)
        r.resize(200)
        assert r.feature_dim == 200
        features = torch.randn(200, device=r.device)
        logits = r(features)
        assert logits.shape == (10,)
