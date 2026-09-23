"""Train/validation score trajectories of the post-generation prompt optimization.

Orange: TextGrad, mean +/- 1 SD across the five seed runs (steps 0-8), from the
per-seed training_trajectory.csv written by utils.prompt_optimizer.

Blue: the human-in-the-loop run (steps 0-10), one run, so no band. Its two
panels come from different graders, which the caption has to say:

  (a) the ratings the reviewer hand-wrote into each iter_<N>/feedback.md for
      that iteration's 7-agent working batch (steps 0-7; 8-10 left blank)
  (b) the teacher, on the fixed 20-persona held-out set built by
      `prompt_optimizer --mode tweets-val-sweep` (jobs/humanopt_val20.job)

Step 1 is missing from both: that iteration's prompt.txt was never saved, so it
can be neither re-scored nor recovered. Segments spanning it are dashed.

    ./.venv_vllm/bin/python data/test_post/optimized_tweets/plot_train_val_panels.py
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

TG_DIR    = "data/test_post/optimized_tweets"
HUMAN_DIR = "data/prompt_optimization_h/qwen27_baseline"
MODEL     = "Qwen3.5-27B"
OUT       = f"{TG_DIR}/train_val_post_opt.png"

TG_COLOR    = "#d96907"
HUMAN_COLOR = "#2e7ebc"
BAND_ALPHA  = 0.22
XMAX        = 10


def textgrad_agg(split):
    """Mean and SD of the per-step score across the TextGrad seed runs."""
    files = sorted(glob.glob(f"{TG_DIR}/{MODEL}_seed*/training_trajectory.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["split"] == split]
    agg = df.groupby("step")["mean_score"].agg(["mean", "std"]).reset_index()
    return agg.sort_values("step"), df["seed"].nunique()


def human_series(split):
    """(steps, means, missing) for the human run."""
    if split == "val":
        df = pd.read_csv(f"{HUMAN_DIR}/val20/trajectory.csv").sort_values("iter")
        steps, means = df["iter"].tolist(), df["mean_score"].tolist()
    else:
        steps, means = [], []
        for i in range(11):
            # e.g. "scores: 7.2, 6.0, 7, ..." — one rating per batch agent.
            # Two files use the template's "test_score" label for the same list.
            for line in open(f"{HUMAN_DIR}/iter_{i}/feedback.md", encoding="utf-8"):
                key, _, rest = line.partition(":")
                if key.strip().lower() not in ("scores", "test_score"):
                    continue
                vals = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", rest)]
                if vals:
                    steps.append(i)
                    means.append(sum(vals) / len(vals))
                break
    return steps, means, [i for i in range(11) if i not in steps]


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
    # Dashed, because every line here spans the unrecoverable step 1.
    ax.plot(steps, means, color=HUMAN_COLOR, linewidth=2.0, linestyle="--",
            marker="o", markersize=4, label="Human-in-the-loop")
    print(f"[{split}] human steps {steps}, no score at {missing}")

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
