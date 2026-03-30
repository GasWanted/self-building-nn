"""Test that the growth engine actually triggers on real data distributions."""

import numpy as np
import pytest
from src.neurons import PrototypeNeuron
from src.network.network import Network
from src.data.mnist import load_small


class TestGrowthDynamics:
    @pytest.fixture
    def data(self):
        Z_tr, y_tr, Z_te, y_te, pca = load_small()
        return Z_tr, y_tr

    @pytest.fixture
    def net(self, data):
        Z_tr, y_tr = data
        n_in = Z_tr.shape[1]
        def nf(d, label=-1):
            return PrototypeNeuron(d, label=label)
        return Network(
            n_in, 10, nf,
            initial_hidden_size=8,
            n_initial_layers=2,
            growth_interval=10,
            depth_check_interval=100,
            split_threshold=0.35,
            depth_threshold=0.85,
            prune_age=500,
            prune_window=500,
        )

    def test_width_growth_triggers(self, net, data):
        """Network must add neurons when seeing diverse real data."""
        Z_tr, y_tr = data
        initial_neurons = net.total_neurons()
        for i in range(min(500, len(Z_tr))):
            net.learn(Z_tr[i], int(y_tr[i]))
        assert net.n_width_grows > 0, (
            f"No width growth after 500 steps. "
            f"Neurons: {initial_neurons} -> {net.total_neurons()}"
        )

    def test_topology_changes(self, net, data):
        """Topology should be different from initial after training."""
        Z_tr, y_tr = data
        initial_topo = net.topology()
        for i in range(min(1000, len(Z_tr))):
            net.learn(Z_tr[i], int(y_tr[i]))
        final_topo = net.topology()
        assert initial_topo != final_topo, (
            f"Topology unchanged after 1000 steps: {initial_topo}"
        )

    def test_pruning_reduces_dead_neurons(self, net, data):
        """After training + pruning, dead neurons should be reduced."""
        Z_tr, y_tr = data
        for i in range(min(1000, len(Z_tr))):
            net.learn(Z_tr[i], int(y_tr[i]))
        dead_before = net.dead_neurons()
        # Run multiple prune passes to catch neurons that just aged past threshold
        for _ in range(3):
            net._prune()
        dead_after = net.dead_neurons()
        total = net.total_neurons()
        # Dead neurons should be a small fraction of total after pruning
        dead_frac = dead_after / max(total, 1)
        assert dead_frac < 0.20, (
            f"Too many dead neurons after pruning: {dead_after}/{total} "
            f"({dead_frac:.0%}), before prune: {dead_before}"
        )

    def test_growth_log(self, net, data):
        """Width and depth growth counters should be accessible."""
        Z_tr, y_tr = data
        for i in range(min(1000, len(Z_tr))):
            net.learn(Z_tr[i], int(y_tr[i]))
        state = net.state()
        assert "n_width_grows" in state
        assert "n_depth_grows" in state
        assert "n_prunes" in state
        print(f"\nGrowth log: width={state['n_width_grows']} "
              f"depth={state['n_depth_grows']} prunes={state['n_prunes']}")
        print(f"Topology: {state['topology']}")
