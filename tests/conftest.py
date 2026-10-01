import numpy as np
import pytest

from brainpfn.hub import resolve_variant

N_ROIS = 400


def make_cohort(n_subjects: int, seed: int = 0, n_timepoints: int = 500):
    """Synthetic correlation matrices whose structure depends on a latent score."""
    rng = np.random.default_rng(seed)
    score = rng.standard_normal(n_subjects)
    base = rng.standard_normal((N_ROIS, 8))
    fc = np.empty((n_subjects, N_ROIS, N_ROIS), dtype=np.float32)
    for i in range(n_subjects):
        loadings = base * (1.0 + 0.5 * score[i])
        ts = rng.standard_normal((n_timepoints, 8)) @ loadings.T + rng.standard_normal((n_timepoints, N_ROIS))
        fc[i] = np.corrcoef(ts.T)
    y = (score + 0.3 * rng.standard_normal(n_subjects)).astype(np.float32)
    return fc, y


@pytest.fixture(scope="session")
def cohort():
    from brainpfn import fc_to_logchol
    fc, y = make_cohort(60)
    return {"fc": fc, "X": fc_to_logchol(fc), "y": y}


def _available(variant: str) -> bool:
    try:
        resolve_variant(variant)
        return True
    except (RuntimeError, ValueError, OSError):
        return False


@pytest.fixture(scope="session", params=["4.7M", "0.8M"])
def variant(request):
    if not _available(request.param):
        pytest.skip(f"weights for variant {request.param} are not available")
    return request.param
