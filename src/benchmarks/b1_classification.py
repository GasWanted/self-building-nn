"""B1: Full classification — train on all data, measure accuracy and topology."""

import time
import numpy as np


def run_b1(network_factory, Z_tr, y_tr, Z_te, y_te):
    """Train network on full training set, evaluate on test set.

    Returns dict with accuracy, topology, dead neuron count.
    """
    print("\n" + "#" * 62)
    print("  B1: CLASSIFICATION")
    print("#" * 62)

    net = network_factory()
    t0 = time.time()

    checkpoints = {
        int(len(Z_tr) * 0.15),
        int(len(Z_tr) * 0.5),
        len(Z_tr),
    }

    n_eval = min(2000, len(Z_te))

    print(f"\n  {'Step':>7} {'Neurons':>8} {'Layers':>7} {'Dead':>5} {'Acc':>7}  time")
    print("  " + "-" * 50)

    for i in range(len(Z_tr)):
        net.learn(Z_tr[i], int(y_tr[i]))

        if (i + 1) in checkpoints:
            preds = np.array([net.predict(Z_te[j]) for j in range(n_eval)])
            acc = float(np.mean(preds == y_te[:n_eval]))
            print(f"  {i+1:>7} {net.total_neurons():>8} {net.depth():>7} "
                  f"{net.dead_neurons():>5} {acc:>7.1%}  {time.time()-t0:.0f}s")

    # Sleep consolidation
    print("  Sleep replay...", end=" ", flush=True)
    buf = list(zip(Z_tr, y_tr.astype(int)))
    net.sleep_replay(buf, n_steps=2000)
    print("done")

    # Final eval
    preds = np.array([net.predict(Z_te[j]) for j in range(len(Z_te))])
    acc = float(np.mean(preds == y_te))

    topo = net.topology()
    print(f"\n  Final: {acc:.1%}  topology={topo}  dead={net.dead_neurons()}")

    return {
        "accuracy": acc,
        "topology": topo,
        "total_neurons": net.total_neurons(),
        "dead_neurons": net.dead_neurons(),
        "state": net.state(),
    }
