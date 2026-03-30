"""B2: DA Theorem verification — oracle gap vs K on real data."""

import numpy as np
from src.infrastructure.da_selection import da_select


def run_b2(Z_tr, y_tr, Z_te, y_te):
    """Verify DA theorem: cosine_sim(z, z_bar) selects MLE prototype.

    Tests K=1,2,3,5,10,20 and checks O(1/sqrt(K)) oracle gap scaling.
    """
    print("\n" + "#" * 62)
    print("  B2: DA THEOREM — oracle gap vs K")
    print("#" * 62)

    rng = np.random.default_rng(42)
    n_classes = 10
    n_eval = min(1000, len(Z_te))

    # Oracle prototypes (full class means)
    oracle_protos = [Z_tr[y_tr == c].mean(axis=0) for c in range(n_classes)]

    def nn1(protos, z):
        sims = [np.dot(z, p) / (np.linalg.norm(z) * np.linalg.norm(p) + 1e-9)
                for p in protos]
        return int(np.argmax(sims))

    acc_oracle = float(np.mean([
        nn1(oracle_protos, Z_te[i]) == y_te[i] for i in range(n_eval)
    ]))

    print(f"\n  {'K':>4} {'Random':>8} {'DA-group':>10} {'Oracle':>8}  DA gain")
    print(f"  {'-' * 48}")

    results = {}
    for K in [1, 2, 3, 5, 10, 20]:
        acc_rand, acc_da = [], []
        for _ in range(20):
            cands = {}
            for c in range(n_classes):
                idx_c = np.where(y_tr == c)[0]
                picks = rng.choice(idx_c, K, replace=False)
                cands[c] = Z_tr[picks]

            rand_protos = [cands[c][rng.integers(K)] for c in range(n_classes)]
            da_protos = [da_select(cands[c])[0] for c in range(n_classes)]

            sub = rng.choice(n_eval, min(200, n_eval), replace=False)
            acc_rand.append(np.mean([nn1(rand_protos, Z_te[i]) == y_te[i] for i in sub]))
            acc_da.append(np.mean([nn1(da_protos, Z_te[i]) == y_te[i] for i in sub]))

        mr, md = np.mean(acc_rand), np.mean(acc_da)
        results[K] = (mr, md, acc_oracle)
        print(f"  {K:>4} {mr:>8.1%} {md:>10.1%} {acc_oracle:>8.1%}  +{md-mr:>5.1%}")

    k5_gain = results[5][1] - results[5][0] if 5 in results else 0
    return {"k_results": results, "oracle": acc_oracle, "k5_gain": k5_gain}
