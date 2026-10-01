import numpy as np
import pytest

from brainpfn import fc_to_logchol


def _reference(mat):
    w, V = np.linalg.eigh(mat.astype(np.float64))
    L = np.linalg.cholesky(V @ np.diag(np.maximum(w, 1e-6)) @ V.T)
    return np.concatenate([np.log(np.diag(L)), L[np.tril_indices_from(L, k=-1)]])


def test_shapes(cohort):
    fc = cohort["fc"][:3]
    out = fc_to_logchol(fc)
    assert out.shape == (3, 400 * 401 // 2) and out.dtype == np.float32
    assert fc_to_logchol(fc[0]).shape == (400 * 401 // 2,)
    np.testing.assert_array_equal(fc_to_logchol(fc[0]), out[0])


def test_matches_float64_on_full_rank(cohort):
    mat = cohort["fc"][0]
    np.testing.assert_allclose(fc_to_logchol(mat), _reference(mat), atol=1e-4)


def test_log_diagonal_of_correlation_starts_at_zero(cohort):
    # L_00 = sqrt(C_00) = 1 for a correlation matrix
    np.testing.assert_allclose(cohort["X"][:, 0], 0.0, atol=1e-5)


def test_rank_deficient_matrix():
    rng = np.random.default_rng(0)
    mat = np.corrcoef(rng.standard_normal((100, 400)).T)   # rank 99
    out = fc_to_logchol(mat)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("bad", [np.zeros((3, 4, 5)), np.zeros(10), np.zeros((2, 3, 3, 3))])
def test_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        fc_to_logchol(bad)


def test_rejects_nan(cohort):
    fc = cohort["fc"][:2].copy()
    fc[1, 3, 4] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fc_to_logchol(fc)
