"""Per-category PHQ-9 bias correction for the BERT+MLP assessment path.

The fine-tuned regressor is biased per PHQ-9 level: it over-predicts low scores
and under-predicts high ones (regression-to-the-mean shrinkage). When the
regressor sits inside the simulation's feedback loop (its output becomes the
score the next batch of posts is generated from), that level-dependent bias is
an instrument artifact mixed into the well-being drift.

This module estimates the signed bias of the regressor at each PHQ-9 level from
its own held-out test points and exposes it as a lookup table so the simulation
can subtract it at inference, indexed by the agent's *previous* PHQ-9 (the score
the assessed posts were generated at).

Math
----
Given test pairs {(t_i, p_i)}  (t_i = ground-truth label, p_i = raw prediction),
the per-level bias is the mean signed error at that level:

    b(k) = mean_{i : round(t_i)=k} (p_i - t_i)        for k = 0 .. 27

At inference, for an agent with previous PHQ-9 s_prev whose posts the regressor
scored as `pred`, the de-biased score is:

    pred_corrected = pred - b(round(s_prev))

Indexing by s_prev (not by `pred`) removes the systematic offset the instrument
adds at that level while leaving the environment-driven deviation
(pred - g(s_prev)) intact, so social influence still drives the drift. Note this
subtracts a constant offset only; it does not undo the slope compression, so the
environment signal is preserved in direction but not rescaled.

Levels with no test support are filled by linear interpolation over the
populated levels (flat-extended at the ends).
"""

import csv
import os

import numpy as np

N_LEVELS = 28  # PHQ-9 sum-score range is 0..27 inclusive


def compute_bias_table(true, pred, n_levels=N_LEVELS, min_count=1):
    """Per-level signed bias b(k) = mean(pred - true | round(true)=k).

    Args:
        true: iterable of ground-truth PHQ-9 labels.
        pred: iterable of raw regressor predictions (same order/length).
        n_levels: number of PHQ-9 levels (28 → 0..27).
        min_count: minimum test points required for a level to be estimated
            directly; sparser levels are interpolated from the populated ones.

    Returns:
        (bias, counts) where bias is a float array of length n_levels (bias[k]
        is the offset to SUBTRACT from a prediction made at true level k) and
        counts[k] is the number of test points at level k.
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"true/pred length mismatch: {true.shape} vs {pred.shape}")

    levels = np.arange(n_levels)
    bias = np.full(n_levels, np.nan, dtype=float)
    counts = np.zeros(n_levels, dtype=int)

    k = np.clip(np.round(true).astype(int), 0, n_levels - 1)
    for lvl in levels:
        m = k == lvl
        counts[lvl] = int(m.sum())
        if counts[lvl] >= min_count:
            bias[lvl] = float(np.mean(pred[m] - true[m]))

    populated = ~np.isnan(bias)
    if not populated.any():
        # No usable test data — fall back to a no-op (all-zero) correction.
        bias[:] = 0.0
    else:
        # Linear interpolation across gaps; np.interp flat-extends past the ends.
        bias = np.interp(levels, levels[populated], bias[populated])

    return bias, counts


def load_bias_table_from_csv(csv_path, true_col="true_phq9", pred_col=None,
                             n_levels=N_LEVELS, min_count=1):
    """Build a bias table from a per-sample scores CSV.

    Works for both `test_raw_scores.csv` (written by train_BERT_model; columns
    `true_phq9`, `pred_phq9`) and the per-sample output of `eval_bert_on_csv`
    (which adds a `raw_pred` column). When `pred_col` is None the un-rounded
    `raw_pred` column is used if present (better bias estimate, no rounding/clamp
    at the extremes), otherwise `pred_phq9`.

    Returns:
        (bias, counts) — see compute_bias_table.
    """
    true, pred = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        col = pred_col or ("raw_pred" if "raw_pred" in cols else "pred_phq9")
        if true_col not in cols or col not in cols:
            raise ValueError(
                f"{csv_path} missing columns {true_col!r}/{col!r}; found {cols}"
            )
        for row in reader:
            true.append(float(row[true_col]))
            pred.append(float(row[col]))
    if not true:
        raise ValueError(f"No rows read from {csv_path}")
    return compute_bias_table(true, pred, n_levels=n_levels, min_count=min_count)


def apply_bias_correction(pred, s_prev, bias_table):
    """De-bias a single prediction, indexed by the previous PHQ-9.

    Args:
        pred: raw regressor output for this agent.
        s_prev: the agent's previous PHQ-9 (the score its posts were generated at).
        bias_table: array from compute_bias_table / load_bias_table_from_csv,
            or None to disable correction.

    Returns:
        Bias-corrected prediction (float). Returns float(pred) unchanged when
        bias_table is None.
    """
    if bias_table is None or s_prev is None:
        return float(pred)
    k = int(round(float(s_prev)))
    k = max(0, min(len(bias_table) - 1, k))
    return float(pred) - float(bias_table[k])


def save_bias_table(scores_csv, out_path, n_levels=N_LEVELS, min_count=1):
    """Precompute the per-level bias table from a per-sample scores CSV and save it.

    Reads `scores_csv` (true_phq9 + raw_pred/pred_phq9, e.g. a bert-eval seedN.csv),
    aggregates to the per-level offset table, and writes one row per PHQ-9 level:
    `phq9,bias,count`. This is the file the simulation loads at runtime, so the
    aggregation happens once here rather than every run.

    Returns:
        (bias, counts) — see compute_bias_table.
    """
    bias, counts = load_bias_table_from_csv(scores_csv, n_levels=n_levels,
                                            min_count=min_count)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phq9", "bias", "count"])
        for k in range(n_levels):
            writer.writerow([k, round(float(bias[k]), 6), int(counts[k])])
    return bias, counts


def load_bias_table(path, n_levels=N_LEVELS):
    """Load a precomputed `phq9_bias_table.csv` (rows: phq9,bias,count) into an array.

    Returns a float array of length `n_levels` where entry k is the offset to
    SUBTRACT from a prediction made at previous PHQ-9 level k. Missing levels
    default to 0 (no correction).
    """
    bias = np.zeros(n_levels, dtype=float)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "phq9" not in (reader.fieldnames or []) or "bias" not in (reader.fieldnames or []):
            raise ValueError(f"{path} missing 'phq9'/'bias' columns; found {reader.fieldnames}")
        for row in reader:
            k = int(round(float(row["phq9"])))
            if 0 <= k < n_levels:
                bias[k] = float(row["bias"])
    return bias


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Precompute the per-level PHQ-9 bias table.")
    ap.add_argument("--scores", required=True,
                    help="Per-sample scores CSV (true_phq9 + raw_pred/pred_phq9), "
                         "e.g. a bert-eval seedN.csv or a test_raw_scores.csv.")
    ap.add_argument("--out", required=True, help="Output phq9_bias_table.csv path.")
    args = ap.parse_args()
    bias, counts = save_bias_table(args.scores, args.out)
    print(f"[phq9_bias] wrote {args.out}  (n={int(counts.sum())})")
    print(f"[phq9_bias] per-level bias (pred-true): {np.round(bias, 2).tolist()}")
    if (counts == 0).any():
        print(f"[phq9_bias] interpolated levels (no support): "
              f"{[int(k) for k in np.where(counts == 0)[0]]}")


if __name__ == "__main__":
    main()
