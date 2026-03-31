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

    net = FieldNetwork(initial_stride=1, readout_lr=0.003, merge_threshold=0.60)
    print(f"Initial topology: {net.topology()}")

    t0 = time.time()
    best = 0.0

    n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for epoch in range(n_epochs):
        perm = torch.randperm(X_tr.shape[0], device=X_tr.device)
        net.train_batch(X_tr[perm], y_tr[perm], batch_size=512)

        preds = net.predict_batch(X_te)
        acc = float((preds == y_te).float().mean())
        best = max(best, acc)

        # LR decay
        if epoch == n_epochs // 2 or epoch == 3 * n_epochs // 4:
            for pg in net.readout.optimizer.param_groups:
                pg['lr'] *= 0.3
            print(f"  LR decay -> {net.readout.optimizer.param_groups[0]['lr']:.5f}")

        topo = net.topology()
        print(f"Epoch {epoch+1}/{n_epochs}: {acc:.2%} (best={best:.2%})  "
              f"neurons={topo['total_neurons']}  t={time.time()-t0:.0f}s", flush=True)

    print(f"\nBest: {best:.2%}  Total: {time.time() - t0:.0f}s")
    print(f"Topology: {net.topology()}")


if __name__ == "__main__":
    main()
