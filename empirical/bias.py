"""Per-band assessor bias and MAE on the empirical set, across the regressor seeds of every arm.

Reads the per-user prediction CSVs that `prompt_optimizer --mode bert-eval` writes
(`<eval_root>/<arm>/seed<NN>.csv`), bins users by their true PHQ-9 band and reports
signed bias (pred - true; positive = over-estimate) and MAE per band, as mean and SD
over seeds. Same quantity as `visualization._band_bias_per_seed` on synthetic data.
Run: sbatch empirical/run_empirical.job (step 4).
"""

import argparse
import glob
import os

import pandas as pd

from utils.sensitivity.sa_analyze import BAND_LABELS, phq9_to_band


def arm_band_table(arm_dir: str) -> pd.DataFrame:
    """Per-band bias/MAE (mean, SD over seeds) and n users for one assessor arm.

    Args:
        arm_dir: directory holding seed<NN>.csv prediction files.

    Returns:
        DataFrame indexed by band, empty if the arm has no prediction files.
    """
    paths = [p for p in sorted(glob.glob(os.path.join(arm_dir, "seed*.csv")))
             if not p.endswith("_summary.csv")]
    per_seed = []
    for p in paths:
        d = pd.read_csv(p)
        d["band"] = pd.Categorical(d["true_phq9"].map(phq9_to_band), BAND_LABELS)
        d["err"] = d["pred_phq9"] - d["true_phq9"]
        g = d.groupby("band", observed=False)["err"]
        per_seed.append(pd.DataFrame({"bias": g.mean(), "mae": g.apply(lambda e: e.abs().mean()),
                                      "n": g.size()}))
    if not per_seed:
        return pd.DataFrame()
    s = pd.concat(per_seed, keys=range(len(per_seed)))
    by = s.groupby(level="band", observed=False)
    return pd.DataFrame({"n": by["n"].first(), "bias": by["bias"].mean(), "bias_sd": by["bias"].std(),
                         "mae": by["mae"].mean(), "mae_sd": by["mae"].std(),
                         "n_seeds": len(per_seed)}).reindex(BAND_LABELS)


def main() -> None:
    """Parse args, build the per-arm band tables and write one long CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-root", required=True, help="Directory with one <arm>/ subdir per assessor.")
    parser.add_argument("--out", required=True, help="Output CSV.")
    args = parser.parse_args()

    tables = []
    for arm_dir in sorted(glob.glob(os.path.join(args.eval_root, "*", ""))):
        arm = os.path.basename(os.path.normpath(arm_dir))
        t = arm_band_table(arm_dir)
        if t.empty:
            print(f"[bias] {arm}: no seed*.csv, skipping")
            continue
        print(f"\n[bias] {arm}\n{t.to_string(float_format='%+.2f')}")
        tables.append(t.assign(arm=arm).rename_axis("band").reset_index())

    out = pd.concat(tables)[["arm", "band", "n", "bias", "bias_sd", "mae", "mae_sd", "n_seeds"]]
    out.to_csv(args.out, index=False)
    print(f"\n[bias] -> {args.out}")


if __name__ == "__main__":
    main()
