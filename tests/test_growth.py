"""Tests for growth signals."""

import numpy as np
from src.signals.error import should_grow_width, should_grow_depth_v2
from src.signals.information import should_grow_depth


class TestErrorSignal:
    def test_low_sim_triggers_growth(self):
        assert should_grow_width(0.1, split_threshold=0.35) is True

    def test_high_sim_no_growth(self):
        assert should_grow_width(0.8, split_threshold=0.35) is False

    def test_at_threshold_no_growth(self):
        assert should_grow_width(0.35, split_threshold=0.35) is False

    def test_near_one_no_growth(self):
        # similarity ~1 means info gain ~0, below minimum novelty
        assert should_grow_width(0.99, split_threshold=0.35) is False


class TestInformationSignal:
    def test_identical_activations_trigger_depth(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a.copy()
        assert should_grow_depth(a, b, threshold=0.90) is True

    def test_uncorrelated_no_depth(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(100)
        b = rng.standard_normal(100)
        assert should_grow_depth(a, b, threshold=0.90) is False

    def test_constant_activation_no_depth(self):
        a = np.ones(10)
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        assert should_grow_depth(a, b, threshold=0.90) is False

    def test_short_arrays_no_depth(self):
        assert should_grow_depth(np.array([1.0]), np.array([1.0]), 0.9) is False


class TestPredictionErrorDepthSignal:
    def test_stagnant_and_wrong_triggers(self):
        assert should_grow_depth_v2(
            transformation_ratio=0.01, stagnation_threshold=0.05,
            wrong_count=60, patience=50,
        ) is True

    def test_transforming_no_trigger(self):
        assert should_grow_depth_v2(
            transformation_ratio=0.3, stagnation_threshold=0.05,
            wrong_count=100, patience=50,
        ) is False

    def test_stagnant_but_patient(self):
        assert should_grow_depth_v2(
            transformation_ratio=0.01, stagnation_threshold=0.05,
            wrong_count=30, patience=50,
        ) is False

    def test_correct_predictions_no_trigger(self):
        assert should_grow_depth_v2(
            transformation_ratio=0.01, stagnation_threshold=0.05,
            wrong_count=0, patience=50,
        ) is False
