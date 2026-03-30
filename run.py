"""Run the full benchmark suite."""

import sys
from src.neurons import (PrototypeNeuron, PerceptronNeuron, DendriticNeuron,
                         PredictiveCodingNeuron, SpikingNeuron)
from src.network.network import Network
from src.network.fast_network import FastNetwork
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
    neuron_name = "prototype"
    data_name = "small"

    for arg in sys.argv[1:]:
        if arg.startswith("--data="):
            data_name = arg.split("=")[1]
        elif arg in NEURON_TYPES:
            neuron_name = arg

    loader = load_mnist if data_name == "mnist" else load_small
    Z_tr, y_tr, Z_te, y_te, pca = loader()
    n_input = Z_tr.shape[1]

    if "--fast" in sys.argv:
        factory = lambda: FastNetwork(n_input)
        neuron_name = "fast"
    else:
        neuron_class = NEURON_TYPES[neuron_name]
        factory = make_factory(neuron_class, n_input)
    data_loader = lambda: (Z_tr, y_tr, Z_te, y_te, pca)

    results = run_all(factory, data_loader, label=f"{neuron_name} ({data_name})")
