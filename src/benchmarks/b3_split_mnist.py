"""B3: Split MNIST — continual learning, 5 tasks x 2 classes."""

import numpy as np
from src.infrastructure.sleep import sleep_consolidate


def run_b3(network_factory, Z_tr, y_tr, Z_te, y_te):
    """Train on 5 sequential tasks, measure backward transfer (forgetting).

    Tasks: [0,1], [2,3], [4,5], [6,7], [8,9]
    BWT < 0 means catastrophic forgetting.
    """
    print("\n" + "#" * 62)
    print("  B3: SPLIT MNIST — 5 tasks, continual learning")
    print("#" * 62)

    rng = np.random.default_rng(42)
    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def task_data(classes, n=6000):
        mask = np.isin(y_tr, classes)
        return Z_tr[mask][:n], y_tr[mask][:n]

    def eval_tasks(net, n=500):
        accs = []
        for cls in tasks:
            mask = np.isin(y_te, cls)
            Zt, yt = Z_te[mask][:n], y_te[mask][:n]
            preds = [net.predict(Zt[i]) for i in range(len(yt))]
            accs.append(float(np.mean(np.array(preds) == yt)))
        return accs

    net = network_factory()
    hippocampus = []
    accs_after = []

    for ti, cls in enumerate(tasks):
        Zt, yt = task_data(cls)
        perm = rng.permutation(len(Zt))

        for i in perm:
            net.learn(Zt[i], int(yt[i]))
            hippocampus.append((Zt[i].copy(), int(yt[i])))

        # Sleep between tasks
        sleep_consolidate(net, hippocampus, n_steps=1000, rng=rng)

        accs = eval_tasks(net)
        accs_after.append(accs)
        print(f"  After task {ti} {cls}: " +
              " ".join(f"T{j}={a:.0%}" for j, a in enumerate(accs)))

    # Backward transfer
    bwt_vals = []
    for ti in range(4):
        after_own = accs_after[ti][ti]
        final = accs_after[-1][ti]
        bwt_vals.append(final - after_own)
    bwt = float(np.mean(bwt_vals))

    final_accs = accs_after[-1]
    avg = float(np.mean(final_accs))

    print(f"\n  Final: avg={avg:.1%}  BWT={bwt:+.1%}")
    print(f"  Topology: {net.topology()}")

    return {"final_accs": final_accs, "avg": avg, "bwt": bwt, "topology": net.topology()}
