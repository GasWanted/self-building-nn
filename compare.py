"""Compare prototype vs perceptron neurons through full benchmark suite."""

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
    Z_tr, y_tr, Z_te, y_te, pca = load_small()
    n_input = Z_tr.shape[1]
    data_loader = lambda: (Z_tr, y_tr, Z_te, y_te, pca)

    print("\n" + "=" * 62)
    print("  COMPARISON: Prototype vs Perceptron")
    print("=" * 62)

    r_proto = run_all(make_factory(PrototypeNeuron, n_input), data_loader, "Prototype")
    r_perc = run_all(make_factory(PerceptronNeuron, n_input), data_loader, "Perceptron")

    print("\n" + "=" * 62)
    print("  HEAD-TO-HEAD")
    print("=" * 62)
    for metric, k1, k2 in [
        ("B1 accuracy", ("b1", "accuracy"), ("b1", "accuracy")),
        ("B3 BWT", ("b3", "bwt"), ("b3", "bwt")),
        ("B4 accuracy", ("b4", "accuracy"), ("b4", "accuracy")),
        ("B4 seq", ("b4", "seq_score"), ("b4", "seq_score")),
    ]:
        v1 = r_proto[k1[0]][k1[1]]
        v2 = r_perc[k2[0]][k2[1]]
        winner = "Proto" if v1 > v2 else ("Perc" if v2 > v1 else "Tie")
        if isinstance(v1, float):
            print(f"  {metric:<20} Proto={v1:.1%}  Perc={v2:.1%}  -> {winner}")
        else:
            print(f"  {metric:<20} Proto={v1}  Perc={v2}  -> {winner}")
