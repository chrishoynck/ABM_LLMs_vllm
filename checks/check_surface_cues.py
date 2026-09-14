"""Check whether the PHQ-9 regressor leans on emoji and punctuation.

Re-scores a tweets_with_phq9 CSV after removing emoji, expressive punctuation
(!, ?, ellipsis, hashtags) or both, and reports how much MAE and correlation
change. A large drop means the regressor leans on surface cues; a small drop
means it reads the words. Run: see checks/README.md.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from utils.eval_bert_on_csv import evaluate

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF️]")


def no_emoji(t):
    """Remove emoji from one post and collapse whitespace."""
    return re.sub(r"\s+", " ", EMOJI.sub("", str(t))).strip()


def no_punct(t):
    """Remove hashtags and turn !, ? and ellipses into plain periods."""
    t = re.sub(r"#\w+", "", str(t))
    t = re.sub(r"[!?…]+", ".", t)
    t = re.sub(r"\.{2,}", ".", t)
    return re.sub(r"\s+", " ", t).strip()


VARIANTS = {
    "orig": lambda t: str(t),
    "no_emoji": no_emoji,
    "no_punct": no_punct,
    "stripped": lambda t: no_punct(no_emoji(t)).lower(),
}


def main():
    """Score every variant with the regressor and write one summary row per variant."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regressor", required=True, help="path to regressor.pt")
    ap.add_argument("--csv", required=True, help="tweets_with_phq9 CSV to re-score")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default=None, help="torch device; default: cuda if available")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    src = pd.read_csv(a.csv)
    rows = []
    for name, fn in VARIANTS.items():
        d = src.copy()
        d["tweet"] = d.tweet.map(fn)
        csv_path = os.path.join(a.out_dir, f"{name}.csv")
        d.to_csv(csv_path, index=False)
        evaluate(a.regressor, csv_path, os.path.join(a.out_dir, f"{name}_pred.csv"), device=a.device)
        p = pd.read_csv(os.path.join(a.out_dir, f"{name}_pred.csv"))
        rows.append({"variant": name, "mae": p.abs_error.mean(), "bias": p.signed_bias.mean(),
                     "corr": np.corrcoef(p.raw_pred, p.true_phq9)[0, 1],
                     "mae_low(0-9)": p[p.true_phq9 <= 9].abs_error.mean(),
                     "mae_high(15+)": p[p.true_phq9 >= 15].abs_error.mean()})
    res = pd.DataFrame(rows).round(3)
    print("\n" + res.to_string(index=False))
    res.to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
