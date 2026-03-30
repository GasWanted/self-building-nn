"""Compare all neuron types through full benchmark suite."""

import sys
from src.neurons import (PrototypeNeuron, PerceptronNeuron, DendriticNeuron,
                         PredictiveCodingNeuron, SpikingNeuron)
from src.network.network import Network
from src.data.mnist import load_small, load_mnist
from src.benchmarks.harness import run_all

NEURON_TYPES = {
    "prototype": PrototypeNeuron,
    "perceptron": PerceptronNeuron,
    "dendritic": DendriticNeuron,
    "predictive": PredictiveCodingNeuron,
    "spiking": SpikingNeuron,
}


def make_factory(neuron_class, n_input):
    def neuron_factory(n_dim, label=-1):
        return neuron_class(n_dim, label=label)
    def network_factory():
        return Network(n_input=n_input, n_output=10, neuron_factory=neuron_factory)
    return network_factory


if __name__ == "__main__":
    data_name = "small"
    for arg in sys.argv[1:]:
        if arg.startswith("--data="):
            data_name = arg.split("=")[1]

    loader = load_mnist if data_name == "mnist" else load_small
    Z_tr, y_tr, Z_te, y_te, pca = loader()
    n_input = Z_tr.shape[1]
    data_loader = lambda: (Z_tr, y_tr, Z_te, y_te, pca)

    print("\n" + "=" * 62)
    print(f"  COMPARISON: All neuron types ({data_name})")
    print("=" * 62)

    all_results = {}
    for name, cls in NEURON_TYPES.items():
        print(f"\n>>> Running {name}...")
        factory = make_factory(cls, n_input)
        all_results[name] = run_all(factory, data_loader, label=name)

    print("\n" + "=" * 62)
    print("  HEAD-TO-HEAD")
    print("=" * 62)
    print(f"\n  {'Neuron':<15} {'B1 Acc':>8} {'B3 BWT':>8} {'B4 Acc':>8} {'B4 Seq':>7} {'Topology'}")
    print(f"  {'-' * 70}")
    for name, r in all_results.items():
        b1 = r["b1"]["accuracy"]
        bwt = r["b3"]["bwt"]
        b4 = r["b4"]["accuracy"]
        seq = r["b4"]["seq_score"]
        topo = r["b1"]["topology"]
        print(f"  {name:<15} {b1:>8.1%} {bwt:>+8.1%} {b4:>8.0%} {seq:>5}/5  {topo}")
