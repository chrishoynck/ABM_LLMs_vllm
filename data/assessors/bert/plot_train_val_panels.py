"""Three-panel train/val MAE trajectory of the MentalBERT+MLP regressor.

(a) baseline on the teacher corpus, (b) fine-tuned on the minimal-prompt corpus,
(c) fine-tuned on the human-optimized-prompt corpus. Train and validation are
overlaid in each panel, mean +/- 1 SD across the seed runs.

Panel (a) is the same run in both figures: every arm is fine-tuned from the
teacher regressor (see utils.assessors.ARMS).

    ./.venv_vllm/bin/python data/assessors/bert/plot_train_val_panels.py

Writes data/assessors/bert/plots/train_val_bert_{qwen,gemma}.png.
"""
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

ROOT = "data/assessors/bert"
OUT_DIR = f"{ROOT}/plots"
BASE_STEPS = (2, 20)                   # inclusive, exclusive
FINETUNE_STEPS = (2, 31)               # the fine-tuned arms get the full run
SPLITS = [("train", "Train", "#0072B2", "-"),
          ("val", "Validation", "#d96907", "--")]
BAND_ALPHA = 0.22

FIGURES = {
    "qwen": [("teacher", "Qwen3.5-27B", "(a) Baseline (teacher corpus)", BASE_STEPS),
             ("qwen27_minimal", "Qwen3.5-27B", "(b) FT, min. prompt", FINETUNE_STEPS),
             ("qwen27_optimized", "Qwen3.5-27B", "(c) FT, human-opt. prompt", FINETUNE_STEPS)],
    "gemma": [("teacher", "Qwen3.5-27B", "(a) Baseline (teacher corpus)", BASE_STEPS),
              ("gemma4_minimal", "gemma-4-31B-it", "(b) FT, min. prompt", FINETUNE_STEPS),
              ("gemma4_optimized", "gemma-4-31B-it", "(c) FT, human-opt. prompt", FINETUNE_STEPS)],
}


def load_arm(arm, model_short, steps):
    """Mean and SD of the per-step MAE across the arm's seed runs."""
    files = sorted(glob.glob(f"{ROOT}/{arm}/models/{model_short}_seed*/training_trajectory.csv"))
    if not files:
        raise FileNotFoundError(f"no seed runs under {ROOT}/{arm}/models/{model_short}_seed*")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[(df["step"] >= steps[0]) & (df["step"] < steps[1])]
    return df, len(files)


def draw(name, panels):
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.1))
    n_seeds = None
    for ax, (arm, model_short, panel_title, steps) in zip(axes, panels):
        df, n_seeds = load_arm(arm, model_short, steps)
        for split, label, color, style in SPLITS:
            agg = (df[df["split"] == split]
                   .groupby("step")["mean_score"]
                   .agg(["mean", "std"]).reset_index().sort_values("step"))
            band = agg["std"].fillna(0.0)
            ax.plot(agg["step"], agg["mean"], color=color, linewidth=1.6,
                    linestyle=style, label=label)
            ax.fill_between(agg["step"], agg["mean"] - band, agg["mean"] + band,
                            color=color, alpha=BAND_ALPHA, linewidth=0)
        ax.set_xlabel("Step", fontsize=10, labelpad=5)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(steps[0], steps[1] - 1)
        ax.tick_params(labelsize=9)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("MAE", fontsize=10)

    # (b) and (c) are the two fine-tuned arms, so put them on one scale; (a)
    # starts from a randomly initialised MLP and lives an order higher. Both
    # keep their tick labels, so the shared scale is visible rather than implied.
    lo = min(ax.get_ylim()[0] for ax in axes[1:])
    hi = max(ax.get_ylim()[1] for ax in axes[1:])
    for ax in axes[1:]:
        ax.set_ylim(lo, hi)

    axes[0].legend(loc="upper right", frameon=False, fontsize=8,
                   title=f"mean $\\pm$ 1 SD, {n_seeds} seeds", title_fontsize=7)
    fig.tight_layout()

    # Panel labels go under the finished axes rather than in a title with a
    # negative y: at this figure height a fixed offset lands on top of "Step".
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, panel in zip(axes, panels):
        bb = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        fig.text(bb.x0 + bb.width / 2, bb.y0 - 0.06, panel[2],
                 ha="center", va="top", fontsize=10)
    out = os.path.join(OUT_DIR, f"train_val_bert_{name}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, panels in FIGURES.items():
        draw(name, panels)
