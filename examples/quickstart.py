"""
Quickstart: predict a score from functional connectivity with a pretrained BrainPFN.

The data here are synthetic: correlation matrices between 400 regions whose
network structure depends on a hidden score. Replace them with your own
(n_subjects, 400, 400) Schaefer-400 FC matrices.

    python examples/quickstart.py
"""

import numpy as np

from brainpfn import BrainPFN


def synthetic_cohort(n_subjects, n_rois=400, n_timepoints=500, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.standard_normal(n_subjects)
    networks = rng.standard_normal((n_rois, 8))
    fc = np.empty((n_subjects, n_rois, n_rois), dtype=np.float32)
    for i in range(n_subjects):
        signal = rng.standard_normal((n_timepoints, 8)) @ (networks * (1 + 0.5 * score[i])).T
        time_series = signal + rng.standard_normal((n_timepoints, n_rois))
        fc[i] = np.corrcoef(time_series.T)
    y = 100 + 15 * (score + 0.3 * rng.standard_normal(n_subjects))
    return fc, y


fc, y = synthetic_cohort(150)
fc_context, y_context = fc[:100], y[:100]   # labelled subjects
fc_query, y_query = fc[100:], y[100:]       # subjects to predict

model = BrainPFN()                          # downloads the weights on first use
model.fit(fc_context, y_context)
y_pred = model.predict(fc_query)

r = np.corrcoef(y_query, y_pred)[0, 1]
print(f"Context: {len(y_context)} subjects, query: {len(y_query)} subjects")
print(f"Pearson r = {r:.3f}   R² = {model.score(fc_query, y_query):.3f}")
