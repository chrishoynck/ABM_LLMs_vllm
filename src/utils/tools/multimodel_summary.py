"""Generator x estimator summary on each generator's 300-block held-out set.

Reads the per-seed prediction CSVs written by scripts/assessment/run_finetune.sh
(MentalBERT+MLP) and run_llm_assessor_on_heldout.sh (Qwen LLM assessor) and
writes MAE (mean +- SD over seeds), signed bias (pred - true) and per-band MAE:

    data/test_post/method_comparison/multimodel/summary.csv
    data/test_post/method_comparison/multimodel/summary_by_band.csv
    data/test_post/method_comparison/multimodel/table_multimodel.tex
    data/test_post/method_comparison/multimodel/multimodel_mae_bias.png
    data/test_post/method_comparison/multimodel/mae_bias_per_band_finetuned.png

Usage (CPU):
    PYTHONPATH=src .venv_vllm/bin/python -m utils.tools.multimodel_summary
    PYTHONPATH=src .venv_vllm/bin/python -m utils.tools.multimodel_summary --generators qwen gemma4

Missing inputs are skipped with a warning, so it can run while jobs are pending.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from utils.visualization import plot_multimodel_band_bias, plot_multimodel_mae_bias

LABELS = {"qwen": "Qwen3.5-27B", "gemma4": "Gemma-4-31B-it", "kimi": "Kimi-Linear-48B-A3B"}
BANDS = [(0, 4, "Minimal"), (5, 9, "Mild"), (10, 14, "Moderate"),
         (15, 19, "Mod. severe"), (20, 27, "Severe")]
OPT_DIR = "data/test_post/optimized_phq9/Qwen3.5-27B_seed{seed}"


def estimator_files(tag, bert_seeds, prompt_seeds):
    """Estimator name -> per-seed CSVs (columns true_phq9, pred_phq9)."""
    sfx = "" if tag == "qwen" else f"_{tag}"
    held = "human300" if tag == "qwen" else f"{tag}300"
    bert = lambda d: [f"{d}/seed{s}.csv" for s in bert_seeds]
    prompt = lambda sub: [OPT_DIR.format(seed=s) + f"/{sub}/test_raw_scores.csv" for s in prompt_seeds]
    files = {
        "BERT base (Qwen-trained)": bert(f"data/test_post/bert_regression/eval_baseline{sfx}"),
        "BERT fine-tuned (own posts)": bert(f"data/test_post/bert_regression_finetuned{sfx}/eval_finetuned"),
        "LLM prompt (minimal)": prompt(f"minimal_{held}"),
        "LLM prompt (TextGrad)": prompt(f"eval_on_{held}"),
    }
    if tag != "qwen":  # Qwen-fine-tuned regressors transferred to this generator's posts
        files["BERT fine-tuned (Qwen posts)"] = bert(f"data/test_post/bert_regression_finetuned/eval_{tag}300")
    return files


def band(score):
    return next(name for lo, hi, name in BANDS if lo <= score <= hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generators", nargs="+", default=["qwen", "gemma4"])
    ap.add_argument("--bert-seeds", nargs="+", type=int, default=[34, 35, 36, 37, 38])
    ap.add_argument("--prompt-seeds", nargs="+", type=int, default=[23, 24, 25, 32, 33])
    ap.add_argument("--out-dir", default="data/test_post/method_comparison/multimodel")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows, band_rows = [], []
    for tag in args.generators:
        gen = LABELS.get(tag, tag)
        for est, paths in estimator_files(tag, args.bert_seeds, args.prompt_seeds).items():
            per_seed, pooled = [], []
            for p in paths:
                if not os.path.isfile(p):
                    print(f"[skip] {gen} / {est}: missing {p}")
                    continue
                df = pd.read_csv(p)
                err = df["pred_phq9"] - df["true_phq9"]
                per_seed.append({"mae": err.abs().mean(), "bias": err.mean()})
                pooled.append(df.assign(err=err))
            if not per_seed:
                continue
            ps = pd.DataFrame(per_seed)
            rows.append({"generator": gen, "estimator": est, "n_seeds": len(ps),
                         "mae": ps["mae"].mean(), "mae_sd": ps["mae"].std(ddof=0),
                         "bias": ps["bias"].mean(), "bias_sd": ps["bias"].std(ddof=0)})
            pooled = pd.concat(pooled)
            pooled["band"] = pooled["true_phq9"].map(band)
            for b, g in pooled.groupby("band", sort=False):
                band_rows.append({"generator": gen, "estimator": est, "band": b, "n": len(g),
                                  "mae": g["err"].abs().mean(), "bias": g["err"].mean()})

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{args.out_dir}/summary.csv", index=False)
    by_band = pd.DataFrame(band_rows)
    order = [name for _, _, name in BANDS]
    if len(by_band):
        by_band["band"] = pd.Categorical(by_band["band"], order)
        by_band = by_band.sort_values(["generator", "estimator", "band"])
    by_band.to_csv(f"{args.out_dir}/summary_by_band.csv", index=False)
    print(summary.round(2).to_string(index=False))

    # LaTeX table (booktabs, same style as the SI fine-tune table).
    lines = ["\\begin{tabular}{llcc}", "\\toprule",
             "Generator & Estimator & MAE & Bias \\\\", "\\midrule"]
    for gen, g in summary.groupby("generator", sort=False):
        for i, r in enumerate(g.itertuples()):
            name = gen if i == 0 else ""
            lines.append(f"{name} & {r.estimator} & ${r.mae:.2f} \\pm {r.mae_sd:.2f}$ & ${r.bias:+.2f}$ \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    with open(f"{args.out_dir}/table_multimodel.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # Figures live in utils.visualization next to the other MAE/bias plots.
    plot_multimodel_mae_bias(summary, f"{args.out_dir}/multimodel_mae_bias.png")
    plot_multimodel_band_bias(f"{args.out_dir}/mae_bias_per_band_finetuned.png")

    print(f"[done] -> {args.out_dir}/")


if __name__ == "__main__":
    main()
