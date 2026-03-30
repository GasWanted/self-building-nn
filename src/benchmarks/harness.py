"""Benchmark harness — run all B1-B5 on any Network instance."""

import time
import numpy as np
from src.benchmarks.b1_classification import run_b1
from src.benchmarks.b2_da_theorem import run_b2
from src.benchmarks.b3_split_mnist import run_b3
from src.benchmarks.b4_combined import run_b4
from src.benchmarks.b5_ablation import run_b5


def run_all(network_factory, data_loader, label: str = "Network"):
    """Run all benchmarks and print summary.

    Args:
        network_factory: callable() -> Network instance (fresh for each benchmark)
        data_loader: callable() -> (Z_tr, y_tr, Z_te, y_te, pca)
        label: name for this configuration
    """
    print("=" * 62)
    print(f"  BENCHMARK SUITE: {label}")
    print("=" * 62)

    t0 = time.time()
    Z_tr, y_tr, Z_te, y_te, pca = data_loader()

    results = {}
    results["b1"] = run_b1(network_factory, Z_tr, y_tr, Z_te, y_te)
    results["b2"] = run_b2(Z_tr, y_tr, Z_te, y_te)
    results["b3"] = run_b3(network_factory, Z_tr, y_tr, Z_te, y_te)
    results["b4"] = run_b4(network_factory, Z_tr, y_tr, Z_te, y_te)
    results["b5"] = run_b5(network_factory, Z_tr, y_tr, Z_te, y_te)

    total = time.time() - t0

    print("\n" + "=" * 62)
    print(f"  SUMMARY: {label}")
    print("=" * 62)
    print(f"\n  B1 Classification:  {results['b1']['accuracy']:.1%}  "
          f"({results['b1']['topology']})")
    print(f"  B2 DA Theorem:      K=5 gain = {results['b2'].get('k5_gain', 0):+.1%}")
    print(f"  B3 Split MNIST BWT: {results['b3']['bwt']:+.1%}")
    print(f"  B4 Combined:        {results['b4']['accuracy']:.0%}  "
          f"seq={results['b4']['seq_score']}/5")
    print(f"  B5 Ablation:        (see table above)")
    print(f"\n  Total runtime: {total:.1f}s")

    return results
