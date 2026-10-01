import warnings

import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from brainpfn import BrainPFN


def _fit_predict(variant, X_ctx, y_ctx, X_q, **kw):
    return BrainPFN(variant=variant, device="cpu", **kw).fit(X_ctx, y_ctx).predict(X_q)


def test_matrices_and_vectors_agree(variant, cohort):
    fc, X, y = cohort["fc"], cohort["X"], cohort["y"]
    a = _fit_predict(variant, fc[:40], y[:40], fc[40:])
    b = _fit_predict(variant, X[:40], y[:40], X[40:])
    np.testing.assert_array_equal(a, b)
    assert a.shape == (20,) and a.dtype == np.float32


def test_predicts_signal(variant, cohort):
    X, y = cohort["X"], cohort["y"]
    pred = _fit_predict(variant, X[:40], y[:40], X[40:])
    assert np.corrcoef(pred, y[40:])[0, 1] > 0.3


def test_calibration(variant, cohort):
    X, y = cohort["X"], cohort["y"]
    m = BrainPFN(variant=variant, device="cpu").fit(X[:40], y[:40])
    assert isinstance(m.calib_slope_, float) and m.calib_slope_ >= 0
    raw = BrainPFN(variant=variant, calibrate=None, device="cpu").fit(X[:40], y[:40])
    assert raw.calib_slope_ is None
    np.testing.assert_allclose(m.predict(X[40:]),
                               raw.predict(X[40:]) * m.calib_slope_ + m.calib_intercept_, rtol=1e-5, atol=1e-5)


def test_calibration_skipped_for_small_context(variant, cohort):
    X, y = cohort["X"], cohort["y"]
    with pytest.warns(UserWarning, match="uncalibrated"):
        m = BrainPFN(variant=variant, device="cpu").fit(X[:8], y[:8])
    assert m.calib_slope_ is None


def test_multi_target_with_missing_labels(variant, cohort):
    X, y = cohort["X"], cohort["y"]
    Y = np.stack([y, -y, y], axis=1)
    Y[::3, 1] = np.nan
    Y[1:, 2] = np.nan            # a single label: unpredictable
    pred = _fit_predict(variant, X[:40], Y[:40], X[40:])
    assert pred.shape == (20, 3)
    assert np.isfinite(pred[:, :2]).all() and np.isnan(pred[:, 2]).all()
    single = _fit_predict(variant, X[:40], y[:40], X[40:])
    np.testing.assert_allclose(pred[:, 0], single, rtol=1e-5, atol=1e-5)


def test_tiny_contexts(variant, cohort):
    X, y = cohort["X"], cohort["y"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in (2, 3, 5):
            m = BrainPFN(variant=variant, device="cpu").fit(X[:n], y[:n])
            assert np.isfinite(m.predict(X[40:])).all()
            assert m.n_components_ == (100 if n < 3 else n - 1)
        with pytest.raises(ValueError, match="at least 2"):
            BrainPFN(variant=variant, device="cpu").fit(X[:1], y[:1])


def test_deterministic(variant, cohort):
    X, y = cohort["X"], cohort["y"]
    a = _fit_predict(variant, X[:40], y[:40], X[40:], random_state=3)
    b = _fit_predict(variant, X[:40], y[:40], X[40:], random_state=3)
    np.testing.assert_array_equal(a, b)


def test_sklearn_compatibility(variant, cohort):
    X, y = cohort["X"], cohort["y"]
    m = BrainPFN(variant=variant, device="cpu", calibrate_folds=3)
    assert clone(m).get_params() == m.get_params()
    scores = cross_val_score(m, X, y, cv=3, scoring="r2")
    assert scores.shape == (3,) and np.isfinite(scores).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_mps_matches_cpu(variant, cohort):
    X, y = cohort["X"], cohort["y"]
    cpu = _fit_predict(variant, X[:40], y[:40], X[40:])
    mps = BrainPFN(variant=variant, device="mps").fit(X[:40], y[:40]).predict(X[40:])
    np.testing.assert_allclose(mps, cpu, atol=1e-3 * y.std())


def test_input_errors(variant, cohort):
    fc, X, y = cohort["fc"], cohort["X"], cohort["y"]
    m = BrainPFN(variant=variant, device="cpu")
    with pytest.raises(ValueError, match=r"fc\[None\]"):
        m.fit(fc[0], np.zeros(400))
    il = np.tril_indices(400, k=-1)
    with pytest.raises(ValueError, match="lower triangle"):
        m.fit(fc[:10][:, il[0], il[1]], y[:10])
    iu = np.tril_indices(400)
    with pytest.raises(ValueError, match="vectorize"):
        m.fit(fc[:10][:, iu[0], iu[1]], y[:10])
    with pytest.raises(ValueError, match=r"\(400, 400\)"):
        m.fit(np.zeros((10, 100, 100)), y[:10])
    with pytest.raises(ValueError, match="y must be"):
        m.fit(X[:10], y[:9])
    with pytest.raises(ValueError, match="calibrate"):
        BrainPFN(variant=variant, calibrate="yes").fit(X[:10], y[:10])
