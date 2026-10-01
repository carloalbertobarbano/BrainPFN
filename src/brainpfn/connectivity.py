"""
Functional-connectivity matrices → BrainPFN input features.

BrainPFN reads each subject as the log-Cholesky vector of its FC matrix: the
matrix is made positive definite (eigenvalues floored at 1e-6), Cholesky-factored
as L L^T, and flattened to [log diag(L), strictly-lower triangle of L (row-major)],
i.e. d * (d + 1) / 2 values — 80,200 for 400 parcels.

The computation runs in float32, as it did on the pretraining data.
"""

from __future__ import annotations

import numpy as np

_EIG_FLOOR = 1e-6


def _logchol(mat: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, _EIG_FLOOR)
    spd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    L = np.linalg.cholesky(spd)
    return np.concatenate([np.log(np.diag(L)), L[np.tril_indices_from(L, k=-1)]])


def fc_to_logchol(fc: np.ndarray) -> np.ndarray:
    """Log-Cholesky vectors of FC matrices.

    Parameters
    ----------
    fc : (d, d) or (n, d, d) array
        Symmetric FC matrices, e.g. Pearson correlations from
        ``nilearn.connectome.ConnectivityMeasure(kind="correlation")``.

    Returns
    -------
    (d * (d + 1) / 2,) or (n, d * (d + 1) / 2) float32 array.
    """
    fc = np.asarray(fc)
    single = fc.ndim == 2
    if single:
        fc = fc[None]
    if fc.ndim != 3 or fc.shape[1] != fc.shape[2]:
        raise ValueError(f"fc must be (d, d) or (n, d, d) square matrices, got shape {fc.shape}.")
    if not np.isfinite(fc).all():
        bad = np.where(~np.isfinite(fc).all(axis=(1, 2)))[0]
        raise ValueError(f"fc contains NaN/inf values (matrices {bad[:10].tolist()}"
                         f"{' ...' if len(bad) > 10 else ''}).")

    d = fc.shape[1]
    out = np.empty((len(fc), d * (d + 1) // 2), dtype=np.float32)
    for i, mat in enumerate(fc):
        try:
            out[i] = _logchol(mat.astype(np.float32))
        except np.linalg.LinAlgError:
            # float32 round-off can leave a near-singular matrix not quite positive
            # definite after flooring; float64 does not.
            out[i] = _logchol(mat.astype(np.float64)).astype(np.float32)
    return out[0] if single else out
