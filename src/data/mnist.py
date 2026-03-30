"""MNIST loader using sklearn (8x8 digits) or fetching full MNIST."""

import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from sklearn.decomposition import PCA


def load_small(n_pca: int = 50, whiten: bool = True):
    """Load sklearn digits (8x8, 1797 images, 10 classes).
    Returns (X_train, y_train, X_test, y_test) in PCA space.
    """
    digits = load_digits()
    X, y = digits.data.astype(np.float64), digits.target.astype(int)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    n_components = min(n_pca, X_tr.shape[1])
    pca = PCA(n_components=n_components, whiten=whiten)
    Z_tr = pca.fit_transform(X_tr)
    Z_te = pca.transform(X_te)

    return Z_tr, y_tr, Z_te, y_te, pca


def load_mnist(n_pca: int = 50, whiten: bool = True):
    """Load full MNIST (28x28, 70k images, 10 classes).
    Returns (X_train, y_train, X_test, y_test) in PCA space.
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data.astype(np.float64), mnist.target.astype(int)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    X_tr, y_tr = X[:60000], y[:60000]
    X_te, y_te = X[60000:], y[60000:]

    pca = PCA(n_components=n_pca, whiten=whiten)
    Z_tr = pca.fit_transform(X_tr)
    Z_te = pca.transform(X_te)

    return Z_tr, y_tr, Z_te, y_te, pca
