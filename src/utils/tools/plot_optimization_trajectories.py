"""Train/validation trajectories of the optimization loops in the pipeline.

Running the module writes all three figures:

1. BERT regressor, train/val MAE per step (draw_bert, one figure per generator):
   (a) baseline on the teacher corpus, (b) fine-tuned on the minimal-prompt
   corpus, (c) fine-tuned on the human-optimized-prompt corpus. Train and
   validation are overlaid in each panel, mean +/- 1 SD across the seed runs.
   Panel (a) is the same run in both figures: every arm is fine-tuned from the
   teacher regressor (see utils.assessors.ARMS).

2. Post-generation prompt optimization, train/val score per step (draw_post_opt).
   Orange: TextGrad, mean ± SD across seed runs (steps 0-8), from the per-seed
   training_trajectory.csv written by utils.prompt_optimizer. A val point is the
   candidate proposed at that step, before the accept/revert test. A train point
   at step t is the prompt kept after step t (step 0 = initial prompt), scored
   on a fresh batch; train therefore ends at 7.

   Blue: the human-in-the-loop run, one run. Its two panels come from different
   graders, which the caption has to say:

     (a) the ratings the reviewer hand-wrote into each iter_<N>/feedback.md for
         that iteration's 7-agent working batch
     (b) the teacher, on the fixed 20-persona held-out set built by
         `prompt_optimizer --mode tweets-val-sweep` (jobs/humanopt_val20.job)

   iter_1 is left out (its prompt.txt was never saved) and iters 2-10 are
   plotted as steps 1-9.

3. PHQ-9 assessment prompt optimization on the human-optimized corpus, train/val
   MAE per step (draw_phq9_opt). TextGrad only, mean ± SD across the seed runs
   under data/test_post/optimized_phq9_human. Same step convention as (2): val =
   candidate before the accept/revert test, train shifted back one step.

Run: see src/README.md (Hand-run CLIs).
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# 1. BERT regressor training
# ---------------------------------------------------------------------------

BERT_ROOT = "data/assessors/bert"
BERT_OUT_DIR = f"{BERT_ROOT}/plots"
BASE_STEPS = (2, 20)                   # inclusive, exclusive
FINETUNE_STEPS = (2, 31)               # the fine-tuned arms get the full run
SPLITS = [("train", "Train", "#0072B2", "-"),
          ("val", "Validation", "#d96907", "--")]
BERT_BAND_ALPHA = 0.22

BERT_FIGURES = {
    "qwen": [("teacher", "Qwen3.5-27B", "(a) Baseline (teacher corpus)", BASE_STEPS),
             ("qwen27_minimal", "Qwen3.5-27B", "(b) FT, min. prompt", FINETUNE_STEPS),
             ("qwen27_optimized", "Qwen3.5-27B", "(c) FT, human-opt. prompt", FINETUNE_STEPS)],
    "gemma": [("teacher", "Qwen3.5-27B", "(a) Baseline (teacher corpus)", BASE_STEPS),
              ("gemma4_minimal", "gemma-4-31B-it", "(b) FT, min. prompt", FINETUNE_STEPS),
              ("gemma4_optimized", "gemma-4-31B-it", "(c) FT, human-opt. prompt", FINETUNE_STEPS)],
}


def load_arm(arm, model_short, steps):
    """Mean and SD of the per-step MAE across the arm's seed runs."""
    files = sorted(glob.glob(f"{BERT_ROOT}/{arm}/models/{model_short}_seed*/training_trajectory.csv"))
    if not files:
        raise FileNotFoundError(f"no seed runs under {BERT_ROOT}/{arm}/models/{model_short}_seed*")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[(df["step"] >= steps[0]) & (df["step"] < steps[1])]
    return df, len(files)


def draw_bert(name, panels):
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
                            color=color, alpha=BERT_BAND_ALPHA, linewidth=0)
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
    os.makedirs(BERT_OUT_DIR, exist_ok=True)
    out = os.path.join(BERT_OUT_DIR, f"train_val_bert_{name}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


# ---------------------------------------------------------------------------
# 2. Post-generation prompt optimization (TextGrad vs human)
# ---------------------------------------------------------------------------

TG_DIR    = "data/test_post/optimized_tweets"
HUMAN_DIR = "data/prompt_optimization_h/qwen27_baseline"
TG_MODEL  = "Qwen3.5-27B"
POST_OUT  = f"{TG_DIR}/train_val_post_opt.png"

TG_COLOR        = "#d96907"
HUMAN_COLOR     = "#2e7ebc"
POST_BAND_ALPHA = 0.2
XMAX            = 9


def textgrad_runs(split, run_dir=TG_DIR):
    """Per-seed scores as a step x seed table."""
    files = sorted(glob.glob(f"{run_dir}/{TG_MODEL}_seed*/training_trajectory.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["split"] == split].copy()
    if split == "train":
        # The optimizer logs a train row as step+1, but it scores the prompt in
        # use *before* that step's update, so logged step 1 is the initial prompt.
        df["step"] -= 1
    return df.pivot(index="step", columns="seed", values="mean_score").sort_index()


def human_series(split):
    """{step: mean score} for the human run, iter_1 dropped and later iters
    renumbered (iter_N -> step N-1)."""
    if split == "val":
        df = pd.read_csv(f"{HUMAN_DIR}/val20/trajectory.csv")
        by_iter = dict(zip(df["iter"], df["mean_score"]))
    else:
        by_iter = {}
        for i in range(11):
            # e.g. "scores: 7.2, 6.0, 7, ..." — one rating per batch agent.
            # iter_1/2 use the template's "test_score" label for the same list.
            for line in open(f"{HUMAN_DIR}/iter_{i}/feedback.md", encoding="utf-8"):
                key, _, rest = line.partition(":")
                if key.strip().lower() not in ("scores", "test_score"):
                    continue
                vals = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", rest)]
                if vals:
                    by_iter[i] = sum(vals) / len(vals)
                break
    return {(i - 1 if i > 1 else i): v for i, v in by_iter.items() if i != 1}


def draw_post_opt():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for split, ax, title in [("train", axes[0], "(a) Training"),
                             ("val", axes[1], "(b) Validation")]:
        runs = textgrad_runs(split)
        mean, sd = runs.mean(axis=1), runs.std(axis=1)
        ax.fill_between(runs.index, mean - sd, mean + sd, color=TG_COLOR,
                        alpha=POST_BAND_ALPHA, linewidth=0)
        ax.plot(runs.index, mean, color=TG_COLOR, linewidth=2.0,
                label="TextGrad (mean ± SD)")

        human = human_series(split)
        steps = sorted(human)
        ax.plot(steps, [human[s] for s in steps], color=HUMAN_COLOR, linewidth=2.0,
                marker="o", markersize=4, label="Human" if split == "train" else None)
        print(f"[{split}] human " + ", ".join(f"{s}: {v:.2f}" for s, v in sorted(human.items())))

        ax.set_xlabel("Step")
        ax.set_title(title, y=-0.35)
        ax.set_xlim(-0.3, XMAX + 0.3)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Score")
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(POST_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {POST_OUT}")


# ---------------------------------------------------------------------------
# 3. PHQ-9 assessment prompt optimization (TextGrad, human-optimized corpus)
# ---------------------------------------------------------------------------

PHQ9_DIR = "data/test_post/optimized_phq9_human"
PHQ9_OUT = f"{PHQ9_DIR}/train_val_phq9_opt.png"


def draw_phq9_opt():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for split, ax, title in [("train", axes[0], "(a) Training"),
                             ("val", axes[1], "(b) Validation")]:
        runs = textgrad_runs(split, PHQ9_DIR)
        mean, sd = runs.mean(axis=1), runs.std(axis=1)
        ax.fill_between(runs.index, mean - sd, mean + sd, color=TG_COLOR,
                        alpha=POST_BAND_ALPHA, linewidth=0)
        ax.plot(runs.index, mean, color=TG_COLOR, linewidth=2.0)
        print(f"[{split}] {runs.shape[1]} seeds, mean " +
              ", ".join(f"{s}: {v:.2f}" for s, v in mean.items()))

        ax.set_xlabel("Step")
        ax.set_title(title, y=-0.35)
        ax.set_xlim(runs.index.min(), runs.index.max())
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(PHQ9_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PHQ9_OUT}")


if __name__ == "__main__":
    for name, panels in BERT_FIGURES.items():
        draw_bert(name, panels)
    draw_post_opt()
    draw_phq9_opt()
