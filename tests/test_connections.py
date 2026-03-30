"""Tests for skip connection tracking."""

import numpy as np
from src.network.connections import ConnectionTracker


class TestConnectionTracker:
    def test_initial_sequential(self):
        ct = ConnectionTracker(n_layers=3)
        sources = ct.get_sources(layer_idx=2)
        assert sources == {1: 1.0}

    def test_add_connection(self):
        ct = ConnectionTracker(n_layers=4)
        ct.add_connection(target=3, source=1, weight=0.5)
        sources = ct.get_sources(3)
        assert 1 in sources
        assert 2 in sources

    def test_remove_connection(self):
        ct = ConnectionTracker(n_layers=3)
        ct.add_connection(target=2, source=0, weight=0.5)
        ct.remove_connection(target=2, source=0)
        sources = ct.get_sources(2)
        assert 0 not in sources
        assert 1 in sources

    def test_cannot_remove_adjacent(self):
        ct = ConnectionTracker(n_layers=3)
        ct.remove_connection(target=2, source=1)
        sources = ct.get_sources(2)
        assert 1 in sources

    def test_blend_inputs(self):
        ct = ConnectionTracker(n_layers=3)
        ct.add_connection(target=2, source=0, weight=0.5)
        representations = {
            0: np.ones(5) * 1.0,
            1: np.ones(5) * 2.0,
        }
        blended = ct.blend_inputs(target=2, representations=representations)
        expected = (0.5 * np.ones(5) * 1.0 + 1.0 * np.ones(5) * 2.0) / 1.5
        np.testing.assert_array_almost_equal(blended, expected)

    def test_layer_0_no_sources(self):
        ct = ConnectionTracker(n_layers=3)
        sources = ct.get_sources(0)
        assert sources == {}
