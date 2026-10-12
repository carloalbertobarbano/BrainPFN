
![BrainPFN](https://raw.githubusercontent.com/carloalbertobarbano/BrainPFN/main/img/brainpfn.svg)

# BrainPFN

BrainPFN predicts behavioural, cognitive or demographic scores from resting-state
functional connectivity (FC) **in context**. You give it a set of labelled
subjects (the *context*) and it predicts new subjects in a single forward pass of
a pretrained transformer. Nothing is trained or fine-tuned on your data, so it is
fast and needs only a few labelled subjects to start with.

BrainPFN follows the scikit-learn API: `fit` stores the context, `predict` predicts.

Paper: [BrainPFN: Amortized Brain-Behaviour Prediction via Prior-Data Fitted
Networks](https://inria.hal.science/hal-05767107) (Barbano & Wassermann, 2026).

## Installation

```bash
pip install brainpfn
```

It needs Python ≥ 3.10, PyTorch, NumPy and scikit-learn. The pretrained weights
are downloaded the first time a model is fitted.

## Quickstart

```python
from brainpfn import BrainPFN

# fc_context: (n_context, 400, 400) FC matrices, y_context: (n_context,) scores
# fc_query:   (n_query, 400, 400) FC matrices of the subjects to predict
model = BrainPFN()
model.fit(fc_context, y_context)
y_pred = model.predict(fc_query)
```

[`examples/quickstart.py`](https://gitlab.inria.fr/cbarbano/brainpfn/-/blob/main/examples/quickstart.py) runs end to end on synthetic
data. [`examples/predict_from_files.py`](https://gitlab.inria.fr/cbarbano/brainpfn/-/blob/main/examples/predict_from_files.py) runs on
your own `.npy` files:

```bash
python examples/predict_from_files.py \
    --context-fc context_fc.npy --context-y context_y.npy \
    --query-fc query_fc.npy [--query-y query_y.npy]
```

## Input data

BrainPFN expects one FC matrix per subject:

- **Parcellation:** Schaefer 2018, 400 parcels. The model was pretrained with the
  parcels in the 7-network order.
- **Connectivity:** Pearson correlation between parcel time series, as computed by
  nilearn's `ConnectivityMeasure(kind="correlation")`.

```python
from nilearn.connectome import ConnectivityMeasure

# time_series: list of (n_timepoints, 400) arrays, one per subject
fc = ConnectivityMeasure(kind="correlation").fit_transform(time_series)  # (n, 400, 400)
```

Pass the full `(n, 400, 400)` matrices, not nilearn's `vectorize=True` output.
Internally each matrix is converted to its log-Cholesky vector (80,200 values).
For large cohorts you can do the conversion once and pass the vectors instead:

```python
from brainpfn import fc_to_logchol

X = fc_to_logchol(fc)    # (n, 80200)
model.fit(X_context, y_context).predict(X_query)
```

## Pretrained models

| `variant` | parameters | download size |
|-----------|-----------:|--------------:|
| `"4.7M"` (default) | 4.7M | 52 MB |
| `"0.8M"` | 0.8M | 36 MB |

```python
model = BrainPFN(variant="0.8M")
```

The weights are cached in `~/.cache/brainpfn`. To use another location, set the
`BRAINPFN_CACHE_DIR` environment variable. `variant` also accepts a local path to
a downloaded `.pt` file, for machines without internet access.

## Options

```python
BrainPFN(
    variant="4.7M",       # pretrained model
    calibrate="cv",       # output calibration, see below; None to disable
    calibrate_folds=5,
    random_state=0,       # seed of the context PCA and of the calibration folds
    device=None,          # "cpu", "cuda", "mps"; default: CUDA if available, else CPU
)
```

**Calibration.** On real data, the raw network output tends to rank subjects
well but shrink their predictions towards the context mean. With
`calibrate="cv"` (the default), `fit` predicts each context subject from the
others (5-fold) and learns a linear correction `y ≈ a · ŷ + b` that `predict` then
applies. This uses the context only, and makes `fit` run 5 extra forward passes.
It is skipped, with a warning, when the context has fewer than
`2 × calibrate_folds` subjects. How well it works depends on the context size:
larger contexts give a more reliable correction.

**Several targets.** `y` can be `(n, n_targets)`; `predict` then returns
`(n_query, n_targets)`. Missing labels (NaN) are allowed: each target is
predicted from the context subjects that have a label for it.

**Context size.** Any context of at least 2 subjects works, but predictions get
better with more context subjects.

**scikit-learn.** BrainPFN is a regular scikit-learn regressor, so it works with
`cross_val_score`, `cross_val_predict`, `GridSearchCV` and so on. Each fold uses its
own training split as the context:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(BrainPFN(), fc, y, cv=5, scoring="r2")
```

## Citation

If you use BrainPFN, please cite the [paper](https://inria.hal.science/hal-05767107):

```bibtex
@article{barbano2026brainpfn,
  title   = {BrainPFN: Amortized Brain-Behaviour Prediction via Prior-Data Fitted Networks},
  author  = {Barbano, Carlo Alberto and Wassermann, Demian},
  journal = {preprint},
  year    = {2026},
  month   = sep,
  url     = {https://inria.hal.science/hal-05767107}
}
```

## License

MIT; see [LICENSE](https://gitlab.inria.fr/cbarbano/brainpfn/-/blob/main/LICENSE).
