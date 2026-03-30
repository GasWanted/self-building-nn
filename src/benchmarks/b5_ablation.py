"""B5: Ablation — contribution of each component."""

import numpy as np
from src.infrastructure.da_selection import da_select
from src.infrastructure.sleep import sleep_consolidate
from src.infrastructure.theta import ThetaBuffer


def run_b5(network_factory, Z_tr, y_tr, Z_te, y_te):
    """Toggle DA, sleep, theta independently. Measure marginal contribution."""
    print("\n" + "#" * 62)
    print("  B5: ABLATION — component contributions")
    print("#" * 62)

    rng = np.random.default_rng(42)
    K = 5
    day_classes = {1: [0, 1, 2], 2: [3, 4, 5], 3: [6, 7, 8, 9]}

    cands = {c: Z_tr[y_tr == c][:K] for c in range(10)}

    test_Z, test_y = [], []
    for c in range(10):
        idx = np.where(y_tr == c)[0][K:K + 500]
        test_Z.extend(Z_tr[idx])
        test_y.extend([c] * len(idx))
    test_Z = np.array(test_Z)
    test_y = np.array(test_y, dtype=int)

    seq_tests = [([0, 1, 2], 3), ([3, 4, 5], 6), ([0, 1], 2),
                 ([6, 7, 8], 9), ([1, 2, 3], 4)]

    def ablation_run(use_da, use_sleep, use_theta):
        net = network_factory()
        hippocampus = []
        theta = ThetaBuffer()

        for day, classes in day_classes.items():
            for c in classes:
                if use_da:
                    best_z = da_select(cands[c])[0]
                else:
                    best_z = cands[c][0]

                net.learn(best_z, c)
                hippocampus.append((best_z.copy(), c))
                if use_theta:
                    theta.push(best_z, c)

            if use_sleep:
                sleep_consolidate(net, hippocampus, n_steps=800, rng=rng)

        preds = np.array([net.predict(test_Z[i]) for i in range(len(test_y))])
        overall = float(np.mean(preds == test_y))

        mask_d1 = np.isin(test_y, [0, 1, 2])
        d1 = float(np.mean(preds[mask_d1] == test_y[mask_d1]))

        seq_ok = 0
        if use_theta:
            for prefix, expected in seq_tests:
                result = theta.complete(prefix, 1)
                got = result[-1] if len(result) > len(prefix) else None
                seq_ok += (got == expected)

        return overall, d1, seq_ok, net.dead_neurons()

    configs = [
        ("Baseline (no DA, no sleep, no theta)", False, False, False),
        ("+ DA selection only", True, False, False),
        ("+ Sleep only", False, True, False),
        ("+ Theta only", False, False, True),
        ("DA + Sleep", True, True, False),
        ("DA + Sleep + Theta  [FULL]", True, True, True),
    ]

    print(f"\n  {'Config':<38} {'Overall':>8} {'Day1':>6} {'Seq':>5} {'Dead':>5}")
    print(f"  {'-' * 65}")

    results = {}
    for name, da, sl, th in configs:
        ov, d1, sq, dead = ablation_run(da, sl, th)
        print(f"  {name:<38} {ov:>8.0%} {d1:>6.0%} {sq:>4}/5 {dead:>5}")
        results[name] = {"overall": ov, "day1": d1, "seq": sq, "dead": dead}

    return results
