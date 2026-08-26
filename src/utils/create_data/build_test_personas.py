"""Build the EXTRA test personas needed to grow the BERT test set past the existing 120.

Takes (persona, PHQ-9) pairs from the canonical eval file, keeping the original
PHQ-9 labels, and selects rows that:
  * come AFTER the first ``--keep`` rows (those are already generated as test posts), and
  * are NOT in the fine-tune training persona file (no train/test leakage).

Writes the next ``--n - --keep`` such rows, so generating posts for them and
appending to the existing test posts yields a clean ``--n``-block test set.

Usage:
    python -m utils.create_data.build_test_personas --n 300 --keep 120 \\
        --out data/finetune/personas_test_extra.csv
"""

import argparse
import os

import pandas as pd

EVAL_FILE = "data/personas_eval_1000_phq9.csv"
TRAIN_FILE = "data/personas_finetune_phq9.csv"


def build(n: int, keep: int, out_path: str) -> None:
    eval_df = pd.read_csv(EVAL_FILE)
    train_personas = set(pd.read_csv(TRAIN_FILE)["persona"].astype(str))

    n_extra = n - keep
    if n_extra <= 0:
        raise SystemExit(f"--n ({n}) must exceed --keep ({keep}).")

    # Candidates: eval rows after the first `keep`, not in the training set.
    rest = eval_df.iloc[keep:]
    clean = rest[~rest["persona"].astype(str).isin(train_personas)]
    if len(clean) < n_extra:
        raise SystemExit(f"only {len(clean)} eval personas after row {keep} are not in "
                         f"training, but {n_extra} needed for an {n}-block test set.")

    extra = clean.head(n_extra)[["persona", "phq9"]]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    extra.to_csv(out_path, index=False)
    print(f"[test-personas] wrote {out_path} ({len(extra)} rows); "
          f"existing {keep} + these {len(extra)} = {n}-block test set, all disjoint from training.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=300, help="Total test blocks wanted.")
    parser.add_argument("--keep", type=int, default=120,
                        help="Existing test blocks already generated (kept as-is).")
    parser.add_argument("--out", default="data/finetune/personas_test_extra.csv")
    args = parser.parse_args()
    build(args.n, args.keep, args.out)


if __name__ == "__main__":
    main()
