"""Train/validation score trajectories of the post-generation prompt optimization.

Orange: TextGrad, mean +/- 1 SD across the five seed runs (steps 0-8).
Blue:   the human-in-the-loop run (steps 0-10), one run, so no band.

TextGrad numbers come from the per-seed training_trajectory.csv written by
utils.prompt_optimizer. The human loop kept no trajectory file, so the two
splits are rebuilt from the iteration folders:

  train -> batch_grades.csv   teacher grades of that iteration's 7-agent
                              working batch (posts.csv), the set the human
                              read before editing the prompt
  val   -> test_raw_scores.csv  the 100-persona held-out set

Only iterations 0, 7, 8, 9 and 10 were ever evaluated on the held-out set and
no working batch was ever graded, so missing steps are simply left out of the
blue line (the script prints which ones).

    ./.venv_vllm/bin/python scripts/plotting/plot_post_opt_train_val.py
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

TG_DIR    = "data/test_post/optimized_tweets"
HUMAN_DIR = "data/prompt_optimization_h/qwen27_baseline"
MODEL     = "Qwen3.5-27B"
OUT       = f"{TG_DIR}/{MODEL}_mean_sd_score_across_seeds.png"

TG_COLOR    = "#d96907"
HUMAN_COLOR = "#2e7ebc"
BAND_ALPHA  = 0.22
XMAX        = 10
HUMAN_ITERS = range(11)


def textgrad_agg(split):
    """Mean and SD of the per-step score across the TextGrad seed runs."""
    files = sorted(glob.glob(f"{TG_DIR}/{MODEL}_seed*/training_trajectory.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["split"] == split]
    agg = df.groupby("step")["mean_score"].agg(["mean", "std"]).reset_index()
    return agg.sort_values("step"), df["seed"].nunique()


def hand_scores(path):
    """The per-agent ratings the human wrote into feedback.md, e.g.
    'scores: 7.2, 6.0, 7, ...' (the label is 'test_score' in two files)."""
    for line in open(path, encoding="utf-8"):
        key, _, rest = line.partition(":")
        if key.strip().lower() not in ("scores", "test_score"):
            continue
        return [float(m) for m in re.findall(r"\d+(?:\.\d+)?", rest)]
    return []


def human_series(split):
    """Per-iteration mean score of the human loop; iterations without an
    evaluation are dropped. The working batch falls back to the human's own
    hand ratings when no teacher grading exists."""
    steps, means, missing = [], [], []
    for i in HUMAN_ITERS:
        d = f"{HUMAN_DIR}/iter_{i}"
        if split == "val":
            path = f"{d}/test_raw_scores.csv"
            scores = pd.read_csv(path)["score"].dropna().tolist() if os.path.isfile(path) else []
        elif os.path.isfile(f"{d}/batch_grades.csv"):
            scores = pd.read_csv(f"{d}/batch_grades.csv")["score"].dropna().tolist()
        else:
            scores = hand_scores(f"{d}/feedback.md")
        if not scores:
            missing.append(i)
            continue
        steps.append(i)
        means.append(sum(scores) / len(scores))
    return steps, means, missing


fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
for split, ax, title in [("train", axes[0], "(a) Training"),
                         ("val", axes[1], "(b) Validation")]:
    agg, n_seeds = textgrad_agg(split)
    band = agg["std"].fillna(0.0)
    ax.plot(agg["step"], agg["mean"], color=TG_COLOR, linewidth=2.0,
            label=f"TextGrad (mean of {n_seeds} seeds)")
    ax.fill_between(agg["step"], agg["mean"] - band, agg["mean"] + band,
                    color=TG_COLOR, alpha=BAND_ALPHA, linewidth=0,
                    label="±1 SD across seeds")

    steps, means, missing = human_series(split)
    if steps:
        # Dashed wherever consecutive markers skip an unevaluated step, so an
        # interpolated segment is never read as a measurement.
        style = "-" if not missing else "--"
        ax.plot(steps, means, color=HUMAN_COLOR, linewidth=2.0, linestyle=style,
                marker="o", markersize=4, label="Human-in-the-loop")
    if missing:
        print(f"[{split}] no human evaluation at steps {missing}")

    ax.set_xlabel("Step")
    ax.set_title(title, y=-0.35)
    ax.set_xlim(0, XMAX)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Score")
axes[1].legend(loc="lower right", fontsize=7, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"wrote {OUT}")
