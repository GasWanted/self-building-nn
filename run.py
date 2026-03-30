"""Run the full benchmark suite on sklearn digits."""

from src.neurons import PrototypeNeuron, PerceptronNeuron
from src.network.network import Network
from src.data.mnist import load_small
from src.benchmarks.harness import run_all


def make_factory(neuron_class, n_input):
    def neuron_factory(n_dim, label=-1):
        return neuron_class(n_dim, label=label)
    def network_factory():
        return Network(
            n_input=n_input, n_output=10, neuron_factory=neuron_factory,
        )
    return network_factory


if __name__ == "__main__":
    import sys

    Z_tr, y_tr, Z_te, y_te, pca = load_small()
    n_input = Z_tr.shape[1]

    label = sys.argv[1] if len(sys.argv) > 1 else "prototype"
    neuron_class = PerceptronNeuron if label == "perceptron" else PrototypeNeuron

    factory = make_factory(neuron_class, n_input)
    data_loader = lambda: (Z_tr, y_tr, Z_te, y_te, pca)

    results = run_all(factory, data_loader, label=label)
