"""Tests for lateral inhibition."""

import numpy as np
from src.network.inhibition import apply_inhibition


class TestLateralInhibition:
    def test_winners_keep_activation(self):
        acts = np.array([0.9, 0.1, 0.8, 0.05, 0.7])
        result = apply_inhibition(acts, winner_fraction=0.4, inhibition_factor=0.1)
        assert result[0] == 0.9
        assert result[2] == 0.8
        assert result[1] < 0.1 * 1.1
        assert result[3] < 0.05 * 1.1

    def test_at_least_one_winner(self):
        acts = np.array([0.5, 0.3])
        result = apply_inhibition(acts, winner_fraction=0.1, inhibition_factor=0.1)
        assert result[0] == 0.5

    def test_all_zero_passthrough(self):
        acts = np.zeros(5)
        result = apply_inhibition(acts, winner_fraction=0.3, inhibition_factor=0.1)
        np.testing.assert_array_equal(result, acts)

    def test_single_neuron_passthrough(self):
        acts = np.array([0.7])
        result = apply_inhibition(acts, winner_fraction=0.3, inhibition_factor=0.1)
        assert result[0] == 0.7

    def test_suppression_factor(self):
        acts = np.array([1.0, 0.5, 0.1])
        result = apply_inhibition(acts, winner_fraction=0.34, inhibition_factor=0.2)
        assert result[0] == 1.0
        assert abs(result[1] - 0.5 * 0.2) < 1e-9
        assert abs(result[2] - 0.1 * 0.2) < 1e-9
