"""
Context feature transform: per-feature z-score, then a PCA basis fitted on the
context and zero-padded to the network's input width.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA


def feature_zscore_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std of X (ddof=0); a std below 1e-8 is replaced by 1."""
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mu, std


def _zero_pad_pca(p: PCA, target_k: int) -> PCA:
    """Grow a k(<target_k)-component PCA to target_k components, in place.

    The extra components are zero rows, so a small context leaves the trailing
    input slots of the network at exactly zero instead of inventing directions it
    cannot estimate. The padded variances only keep the ratios well defined.
    """
    k = p.components_.shape[0]
    if k >= target_k:
        return p
    pad = target_k - k
    nfeat = p.components_.shape[1]
    pad_variance = np.full(pad, float(p.explained_variance_[-1]) * 1e-3,
                           dtype=p.explained_variance_.dtype)
    p.components_ = np.vstack([p.components_, np.zeros((pad, nfeat), dtype=p.components_.dtype)])
    p.explained_variance_ = np.concatenate([p.explained_variance_, pad_variance])
    p.explained_variance_ratio_ = p.explained_variance_ / p.explained_variance_.sum()
    if hasattr(p, "singular_values_"):
        p.singular_values_ = np.concatenate(
            [p.singular_values_, np.zeros(pad, dtype=p.singular_values_.dtype)])
    p.n_components_ = target_k
    return p


def fit_padded_pca(X: np.ndarray, n_comp: int, random_state) -> tuple[Optional[PCA], int]:
    """Fit a PCA on a context of N rows and zero-pad it to n_comp components.

    N rows support at most N - 1 components, so k = min(n_comp, N - 1) are fitted
    and the rest zero-padded. Returns (pca, k), or (None, 0) when N < 3.
    """
    k = min(n_comp, len(X) - 1)
    if k < 2:
        return None, 0
    p = PCA(n_components=k, random_state=random_state).fit(np.asarray(X, dtype=np.float32))
    return _zero_pad_pca(p, n_comp), k


def pca_from_state(state: dict) -> PCA:
    """A transform-ready PCA from a checkpoint's stored basis (arrays or tensors)."""
    arr = {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v))
           for k, v in state.items() if k in ("components_", "mean_", "explained_variance_")}
    pca = PCA(n_components=int(state["n_components"]))
    pca.components_ = arr["components_"]
    pca.mean_ = arr["mean_"]
    pca.explained_variance_ = arr["explained_variance_"]
    pca.explained_variance_ratio_ = pca.explained_variance_ / pca.explained_variance_.sum()
    pca.n_features_in_ = int(state["n_features_in_"])
    pca.n_components_ = int(pca.components_.shape[0])
    return pca
