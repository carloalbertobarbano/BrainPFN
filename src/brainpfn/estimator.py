"""
BrainPFN as a scikit-learn regressor.

``fit(X, y)`` stores a labelled context of subjects and fits the feature transform
on it; ``predict(X)`` predicts new subjects in-context, in one forward pass of the
pretrained network. No weight is ever updated.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from brainpfn.connectivity import fc_to_logchol
from brainpfn.features import feature_zscore_stats, fit_padded_pca
from brainpfn.hub import DEFAULT_VARIANT, load_network


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not x.flags.writeable:   # e.g. a broadcast view; torch refuses to wrap those cleanly
        x = x.copy()
    return torch.from_numpy(x).to(device)


class BrainPFN(RegressorMixin, BaseEstimator):
    """In-context regressor from resting-state functional connectivity.

    BrainPFN is a pretrained transformer: ``fit`` does not train it, it only stores
    the labelled subjects (the *context*) the predictions are conditioned on.

    Inputs are Schaefer-400 Pearson-correlation FC matrices, either as an
    ``(n, 400, 400)`` array or already converted with :func:`brainpfn.fc_to_logchol`
    to ``(n, 80200)``.

    Parameters
    ----------
    variant : {"4.7M", "0.8M"} or str, default="4.7M"
        Pretrained model, named by its parameter count. The weights are downloaded
        on the first ``fit`` and cached (see :mod:`brainpfn.hub`). A local path or
        an http(s) URL to a released ``.pt`` file is also accepted.
    calibrate : {"cv", None}, default="cv"
        Corrects the network's tendency to shrink predictions towards the context
        mean on real data (right ranking, too-small spread). ``"cv"`` predicts each
        context subject from the others (K-fold), fits ``y ≈ a * y_hat + b`` on
        those predictions, and applies it in ``predict``. The slope is clipped at 0.
        It costs ``calibrate_folds`` extra forward passes in ``fit``, and is skipped
        with a warning when the context has fewer than ``2 * calibrate_folds``
        subjects. ``None`` returns the raw network output.
    calibrate_folds : int, default=5
        Number of folds for ``calibrate="cv"``.
    random_state : int, default=0
        Seed of the context PCA and of the calibration folds.
    device : str or None, default=None
        Torch device, e.g. ``"cpu"``, ``"cuda"``, ``"mps"``. None picks CUDA when
        available, else CPU.

    Attributes
    ----------
    calib_slope_, calib_intercept_ : float, (n_targets,) array or None
        Calibration fitted by ``calibrate="cv"`` (None when off or skipped).
    n_components_ : int
        Number of PCA components fitted on the context (at most n_context - 1; the
        rest of the network's 100 input slots are zero).
    n_features_in_ : int
        Feature width of the converted input (80200).
    device_ : torch.device
        Device the network runs on.

    Examples
    --------
    >>> model = BrainPFN().fit(fc_context, y_context)   # fc: (n, 400, 400)
    >>> y_pred = model.predict(fc_query)
    """

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        calibrate: Optional[str] = "cv",
        calibrate_folds: int = 5,
        random_state: int = 0,
        device: Optional[str] = None,
    ):
        self.variant = variant
        self.calibrate = calibrate
        self.calibrate_folds = calibrate_folds
        self.random_state = random_state
        self.device = device

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "X_context_")

    # ------------------------------------------------------------------
    # input handling
    # ------------------------------------------------------------------

    def _as_features(self, X) -> np.ndarray:
        """(n, R, R) FC matrices or (n, R(R+1)/2) log-Cholesky vectors -> float32 vectors."""
        n_rois, n_features = self.transform_["n_rois"], self.transform_["n_features"]
        X = np.asarray(X)
        if X.ndim == 3:
            if X.shape[1:] != (n_rois, n_rois):
                raise ValueError(f"FC matrices must be ({n_rois}, {n_rois}), got shape {X.shape}.")
            X = fc_to_logchol(X)
        elif X.ndim != 2:
            raise ValueError(f"X must be (n, {n_rois}, {n_rois}) FC matrices or (n, {n_features}) "
                             f"log-Cholesky vectors, got shape {X.shape}.")
        if X.shape[1] != n_features:
            hint = ""
            if X.shape == (n_rois, n_rois):
                hint = f" To pass a single FC matrix, add a subject axis: fc[None], shape (1, {n_rois}, {n_rois})."
            elif X.shape[1] == n_rois * (n_rois - 1) // 2:
                hint = (" This looks like the lower triangle of the FC matrices; pass the full "
                        f"(n, {n_rois}, {n_rois}) matrices instead.")
            raise ValueError(f"X must be (n, {n_rois}, {n_rois}) FC matrices or (n, {n_features}) "
                             f"log-Cholesky vectors, got shape {X.shape}.{hint}")
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf values.")
        # Entry 0 is log L_00 = 0 for a correlation matrix, but its diagonal value 1 in
        # a plain vectorised matrix (nilearn's vectorize=True), which has the same width.
        if len(X) and np.allclose(X[:, 0], 1.0, atol=1e-3):
            raise ValueError(
                "X looks like vectorised correlation matrices (e.g. nilearn's vectorize=True "
                f"output), not log-Cholesky vectors. Pass the (n, {n_rois}, {n_rois}) matrices, "
                "or convert them with brainpfn.fc_to_logchol.")
        return X

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "BrainPFN":
        """Store the labelled context and fit the feature transform on it.

        Parameters
        ----------
        X : (n, 400, 400) FC matrices or (n, 80200) log-Cholesky vectors.
        y : (n,) or (n, n_targets) scores. NaNs are allowed: each target is then
            predicted from the context subjects that have a label for it.
        """
        if self.calibrate not in (None, "cv"):
            raise ValueError(f"calibrate must be 'cv' or None, got {self.calibrate!r}.")
        if self.calibrate == "cv" and self.calibrate_folds < 2:
            raise ValueError(f"calibrate_folds must be >= 2, got {self.calibrate_folds}.")

        self.device_ = torch.device(self.device if self.device is not None else
                                    ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net_, self.transform_, self.pretrain_pca_ = load_network(self.variant, self.device_)

        X = self._as_features(X)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim not in (1, 2) or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must be (n,) or (n, n_targets) with n={X.shape[0]}, got shape {y.shape}.")
        if len(X) < 2:
            raise ValueError(f"BrainPFN needs at least 2 context subjects, got {len(X)}.")

        n_comp = self.transform_["n_comp"]
        self.ctx_mean_, self.ctx_std_ = None, None
        Xn = X
        if self.transform_["group_norm"]:
            self.ctx_mean_, self.ctx_std_ = feature_zscore_stats(X)
            Xn = (X - self.ctx_mean_) / self.ctx_std_

        # PCA basis refitted on the context; a context of < 3 subjects cannot
        # support one, and falls back to the basis of the pretraining data.
        pca, k = fit_padded_pca(Xn, n_comp, self.random_state)
        if pca is None:
            pca, k = self.pretrain_pca_, n_comp
        self.pca_, self.n_components_ = pca, k

        self.n_features_in_ = X.shape[1]
        self.X_context_ = np.asarray(pca.transform(Xn), dtype=np.float32)
        self.y_context_ = y
        self.calib_slope_, self.calib_intercept_ = None, None
        if self.calibrate == "cv":
            self._fit_cv_calibration(X, y)
        return self

    def _fit_cv_calibration(self, X: np.ndarray, y: np.ndarray) -> None:
        """Per-target affine map y ~ a * y_hat + b from K-fold out-of-fold predictions
        on the context. Each fold refits the whole transform on its training rows."""
        n_folds = self.calibrate_folds
        if len(X) < 2 * n_folds:
            warnings.warn(f"calibrate='cv' needs >= {2 * n_folds} context subjects for "
                          f"{n_folds} folds, got {len(X)}; predictions are left uncalibrated.")
            return
        y2 = y[:, None] if y.ndim == 1 else y
        sub = BrainPFN(variant=self.variant, calibrate=None, random_state=self.random_state,
                       device=self.device)
        oof = np.full(y2.shape, np.nan, dtype=np.float32)
        for tr, va in KFold(n_folds, shuffle=True, random_state=self.random_state).split(X):
            oof[va] = sub.fit(X[tr], y2[tr]).predict(X[va])
        slope = np.ones(y2.shape[1])
        icpt = np.zeros(y2.shape[1])
        for j in range(y2.shape[1]):
            m = np.isfinite(y2[:, j]) & np.isfinite(oof[:, j])
            if m.sum() < 3 or oof[m, j].std() < 1e-12:
                continue   # nothing to fit: leave this target uncalibrated
            a, b = np.polyfit(oof[m, j].astype(np.float64), y2[m, j].astype(np.float64), 1)
            if a < 0:
                a, b = 0.0, float(y2[m, j].mean())
            slope[j], icpt[j] = a, b
        if y.ndim == 1:
            self.calib_slope_, self.calib_intercept_ = float(slope[0]), float(icpt[0])
        else:
            self.calib_slope_, self.calib_intercept_ = slope, icpt

    def transform(self, X) -> np.ndarray:
        """Map subjects into the network's input space (context z-score, then PCA)."""
        check_is_fitted(self, msg="BrainPFN is not fitted; call fit(X_context, y_context) first.")
        X = self._as_features(X)
        if self.ctx_mean_ is not None:
            X = (X - self.ctx_mean_) / self.ctx_std_
        return np.asarray(self.pca_.transform(X), dtype=np.float32)

    # Targets per batched forward in predict(): at most _PREDICT_CHUNK, which bounds
    # the (B, heads, N, N) attention tensors for wide multi-target contexts, and fewer
    # when one target's (N + M, d) input is large. Targets are independent, so
    # chunking does not change the result.
    _PREDICT_CHUNK = 32
    _PREDICT_ELEM_BUDGET = 2 ** 26

    def predict(self, X) -> np.ndarray:
        """Predict new subjects in-context.

        Parameters
        ----------
        X : (m, 400, 400) FC matrices or (m, 80200) log-Cholesky vectors.

        Returns
        -------
        (m,) predictions for a 1-D ``y``, (m, n_targets) for a 2-D ``y``. A target
        with fewer than 2 labelled context subjects is predicted as NaN.
        """
        Xt = self.transform(X)
        y = self.y_context_
        if y.ndim == 1:
            if np.isfinite(y).all():
                out = self._forward(self.X_context_, y, Xt)
            else:
                out = self._predict_columns(Xt, y[:, None])[:, 0]
        else:
            out = self._predict_columns(Xt, y)
        if self.calib_slope_ is not None:
            out = (out * self.calib_slope_ + self.calib_intercept_).astype(np.float32)
        return out

    def _predict_columns(self, Xt: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.empty((len(Xt), y.shape[1]), dtype=np.float32)
        finite = np.isfinite(y)
        full = np.where(finite.all(axis=0))[0]
        n_ctx, d = self.X_context_.shape
        chunk = max(1, min(self._PREDICT_CHUNK,
                           self._PREDICT_ELEM_BUDGET // max(1, (n_ctx + len(Xt)) * d)))
        for start in range(0, len(full), chunk):
            cols = full[start:start + chunk]
            B = len(cols)
            X_ctx_b = np.broadcast_to(self.X_context_, (B,) + self.X_context_.shape)
            X_te_b = np.broadcast_to(Xt, (B,) + Xt.shape)
            out[:, cols] = self._forward(X_ctx_b, y[:, cols].T, X_te_b).T
        for j in np.where(~finite.all(axis=0))[0]:
            m = finite[:, j]
            if m.sum() < 2:
                out[:, j] = np.nan   # no usable context for this target
                continue
            out[:, j] = self._forward(self.X_context_[m], y[m, j], Xt)
        return out

    def _forward(self, X_ctx: np.ndarray, y_ctx: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        """Run the network on inputs already in its feature space: one task
        (N, d), (N,), (M, d) -> (M,), or a batch (B, N, d), (B, N), (B, M, d) -> (B, M)."""
        single = X_ctx.ndim == 2
        if single:
            X_ctx, y_ctx, X_te = X_ctx[None], y_ctx[None], X_te[None]
        dev = self.device_
        with torch.no_grad():
            out = self.net_(_to_tensor(X_ctx, dev), _to_tensor(y_ctx, dev), _to_tensor(X_te, dev))
        out = out.float().cpu().numpy()
        return out[0] if single else out
