"""B4: Combined task — one-shot + retention + sequence completion."""

import numpy as np
from src.infrastructure.da_selection import da_select
from src.infrastructure.sleep import sleep_consolidate
from src.infrastructure.theta import ThetaBuffer


def run_b4(network_factory, Z_tr, y_tr, Z_te, y_te):
    """One-shot learning with DA selection, retention across days, sequence completion.

    Day 1: classes 0,1,2  |  Day 2: classes 3,4,5  |  Day 3: classes 6,7,8,9
    K=5 candidates per class, DA selects best.
    """
    print("\n" + "#" * 62)
    print("  B4: COMBINED TASK — one-shot + retention + sequence")
    print("#" * 62)

    rng = np.random.default_rng(42)
    K = 5
    n_classes = 10
    day_classes = {1: [0, 1, 2], 2: [3, 4, 5], 3: [6, 7, 8, 9]}

    cands = {c: Z_tr[y_tr == c][:K] for c in range(n_classes)}

    # Build test set
    test_Z, test_y = [], []
    for c in range(n_classes):
        idx = np.where(y_tr == c)[0][K:K + 500]
        test_Z.extend(Z_tr[idx])
        test_y.extend([c] * len(idx))
    test_Z = np.array(test_Z)
    test_y = np.array(test_y, dtype=int)

    net = network_factory()
    hippocampus = []
    theta = ThetaBuffer()
    day_accs = {}

    print(f"\n  K={K} candidates per class\n")
    for day, classes in day_classes.items():
        print(f"  DAY {day}: {classes}")
        for c in classes:
            best_z, scores, idx = da_select(cands[c])
            print(f"    {c}: DA=[{', '.join(f'{s:.2f}' for s in scores)}] -> #{idx}")

            net.learn(best_z, c)
            hippocampus.append((best_z.copy(), c))
            theta.push(best_z, c)

        # Sleep
        sleep_consolidate(net, hippocampus, n_steps=800, rng=rng)

        # Evaluate retention
        for d2, cls2 in day_classes.items():
            if d2 <= day:
                mask = np.isin(test_y, cls2)
                preds = np.array([net.predict(test_Z[i]) for i in range(len(test_y))])
                day_accs[(day, d2)] = float(np.mean(preds[mask] == test_y[mask]))

        print("    " + " ".join(
            f"D{d2}={day_accs[(day, d2)]:.0%}" for d2 in range(1, day + 1)))

    # Overall accuracy
    preds = np.array([net.predict(test_Z[i]) for i in range(len(test_y))])
    overall = float(np.mean(preds == test_y))

    # Sequence completion
    seq_tests = [([0, 1, 2], 3), ([3, 4, 5], 6), ([0, 1], 2),
                 ([6, 7, 8], 9), ([1, 2, 3], 4)]
    seq_ok = 0
    for prefix, expected in seq_tests:
        result = theta.complete(prefix, 1)
        got = result[-1] if len(result) > len(prefix) else None
        ok = got == expected
        seq_ok += ok
        print(f"    {prefix} -> {got}  (expected {expected})  {'Y' if ok else 'N'}")

    print(f"\n  Overall: {overall:.0%}  Seq: {seq_ok}/5  Topology: {net.topology()}")

    return {
        "accuracy": overall,
        "seq_score": seq_ok,
        "day_accs": day_accs,
        "topology": net.topology(),
    }
