"""Raw MNIST loader — 28x28 tensors, no PCA."""

import torch
import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist_raw(device: str = "cuda"):
    """Load MNIST as torch tensors on GPU.
    Returns: X_train (60000,1,28,28), y_train (60000,), X_test (10000,1,28,28), y_test (10000,)
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    X_tr = torch.tensor(X[:60000].reshape(-1, 1, 28, 28), device=dev)
    y_tr = torch.tensor(y[:60000], dtype=torch.long, device=dev)
    X_te = torch.tensor(X[60000:].reshape(-1, 1, 28, 28), device=dev)
    y_te = torch.tensor(y[60000:], dtype=torch.long, device=dev)
    return X_tr, y_tr, X_te, y_te
