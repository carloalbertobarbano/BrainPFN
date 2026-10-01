"""
Predict scores for new subjects from a labelled context, with data read from files.

Inputs are .npy files:
  --context-fc  (n_context, 400, 400) FC matrices, or (n_context, 80200) vectors
                from brainpfn.fc_to_logchol
  --context-y   (n_context,) or (n_context, n_targets) scores; NaN = missing label
  --query-fc    (n_query, 400, 400) FC matrices, or (n_query, 80200) vectors
  --query-y     optional true scores of the query subjects, to report r and R²

    python examples/predict_from_files.py --context-fc ctx_fc.npy --context-y ctx_y.npy \
        --query-fc query_fc.npy --out predictions.npy
"""

import argparse

import numpy as np

from brainpfn import VARIANTS, BrainPFN


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context-fc", required=True)
    parser.add_argument("--context-y", required=True)
    parser.add_argument("--query-fc", required=True)
    parser.add_argument("--query-y", default=None)
    parser.add_argument("--out", default="predictions.npy", help="where to save the predictions")
    parser.add_argument("--variant", default="4.7M",
                        help=f"pretrained model: one of {sorted(VARIANTS)}, or a path to a .pt file")
    parser.add_argument("--no-calibrate", action="store_true", help="return the raw network output")
    parser.add_argument("--device", default=None, help="cpu, cuda or mps (default: cuda if available)")
    args = parser.parse_args()

    fc_context, y_context = np.load(args.context_fc), np.load(args.context_y)
    fc_query = np.load(args.query_fc)

    model = BrainPFN(variant=args.variant, calibrate=None if args.no_calibrate else "cv",
                     device=args.device)
    model.fit(fc_context, y_context)
    y_pred = model.predict(fc_query)
    np.save(args.out, y_pred)
    print(f"Context: {len(fc_context)} subjects | query: {len(fc_query)} subjects | "
          f"variant {args.variant} on {model.device_}")
    print(f"Saved predictions {y_pred.shape} to {args.out}")

    if args.query_y is not None:
        y_true = np.load(args.query_y)
        y_true, y_pred = y_true.reshape(len(y_true), -1), y_pred.reshape(len(y_pred), -1)
        for j in range(y_true.shape[1]):
            m = np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
            t, p = y_true[m, j], y_pred[m, j]
            r = np.corrcoef(t, p)[0, 1]
            r2 = 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
            print(f"target {j}: Pearson r = {r:.4f}   R² = {r2:.4f}   (n = {m.sum()})")


if __name__ == "__main__":
    main()
