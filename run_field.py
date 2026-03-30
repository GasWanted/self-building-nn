"""Train NeuronField on raw MNIST."""

import torch
import time
import sys
from src.field.field_network import FieldNetwork
from src.field.mnist_raw import load_mnist_raw


def main():
    print("Loading MNIST...", flush=True)
    X_tr, y_tr, X_te, y_te = load_mnist_raw()
    print(f"Train: {X_tr.shape}  Test: {X_te.shape}  Device: {X_tr.device}")

    net = FieldNetwork()
    print(f"Initial topology: {net.topology()}")

    t0 = time.time()

    n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}", flush=True)

        perm = torch.randperm(X_tr.shape[0], device=X_tr.device)
        X_tr_s = X_tr[perm]
        y_tr_s = y_tr[perm]

        net.train_batch(X_tr_s, y_tr_s, batch_size=64)

        t_elapsed = time.time() - t0
        topo = net.topology()
        print(f"  Time: {t_elapsed:.1f}s  Neurons: {topo['total_neurons']}  "
              f"Layers: {topo['layers']}  Width grows: {topo['n_width_grows']}")

        print("  Evaluating...", end=" ", flush=True)
        t1 = time.time()
        n_eval = min(10000, X_te.shape[0])
        preds = net.predict_batch(X_te[:n_eval])
        acc = float((preds == y_te[:n_eval]).float().mean())
        print(f"Acc: {acc:.1%}  ({time.time() - t1:.1f}s)")

    print(f"\nFinal: {acc:.1%}  Total time: {time.time() - t0:.1f}s")
    print(f"Topology: {net.topology()}")


if __name__ == "__main__":
    main()
