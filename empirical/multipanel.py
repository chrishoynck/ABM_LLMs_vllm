"""The paper's per-model figure (synthetic) with the real users' adjacent-band ladder added under (a).

`plot_model_with_empirical` is a copy of `visualization.plot_model_linearity_bias` (so the
co-author's module stays untouched) whose panel (a) is split into a synthetic and an
empirical row, the way (b) is split into LLM assessor and MentalBERT+MLP. Called from
`figures.ipynb` section 8 for Qwen.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.visualization import (MODEL_FIG_ASSESSORS, MODEL_FIG_LLM_ASSESSORS, MULTIMODEL_GENERATORS, MULTIMODEL_GRADINGS_300, MULTIMODEL_MINIMAL_ARMS, MULTIMODEL_PROMPT_COLOURS, MULTIMODEL_PROMPT_ITER_POSTS, MULTIMODEL_PROMPT_LABELS, MULTIMODEL_PROMPT_STYLES, MULTIMODEL_SA_PROMPTS, MULTIMODEL_SA_ROOTS_SEEDED, MULTIMODEL_TEST_POSTS, _MM_BANDS, _MM_BAND_SHORT, _band_bias_per_seed, _errorbar_annotated, _textgrad_best_seed)


def plot_model_with_empirical(label: str, out_path: str, empirical_ladder: pd.DataFrame,
                              exclude_prompt_iter_personas: bool = False,
                              emb_name: str = "embeddings_sbert.npz", sa_roots: dict = None,
                              gradings: dict = None):
    """`visualization.plot_model_linearity_bias` with the real users' ladder under panel (a).

    A copy of the per-model figure (so `visualization.py` stays untouched): (a) is split
    like (b) into a stack of two labelled rows sharing the step axis, each on its own
    y-scale: the synthetic adjacent-band cosine on top, exactly as in the original, and the
    empirical matched next-band cosine (`empirical/ladder.py`, bootstrap 95% CI) below.
    (b) (LLM assessor / MentalBERT+MLP) and (c) are unchanged; see the original's docstring.

    Args:
        label (str): generator, a `MULTIMODEL_GENERATORS` label.
        out_path (str): PNG to write; the synthetic (a) numbers go next to it as `<stem>.csv`.
        empirical_ladder (pd.DataFrame): `ladder_matched.csv` (cos, ci_lo, ci_hi per step).
        exclude_prompt_iter_personas, emb_name, sa_roots, gradings: as in the original.

    Returns:
        str | None: `out_path`, or None if this generator has no test posts / evals.
    """
    from utils.sensitivity.sa_analyze import phq9_adjacent_band_ladder
    from matplotlib.lines import Line2D
    # Seeded by default: the unseeded SA reps share a vLLM seed, so their round-0 posts
    # duplicate across reps and the error bars are understated.
    sa_roots = MULTIMODEL_SA_ROOTS_SEEDED if sa_roots is None else sa_roots
    gradings = MULTIMODEL_GRADINGS_300 if gradings is None else gradings
    short, posts_csv = MULTIMODEL_TEST_POSTS.get(label, (None, None))
    eval_dir = dict((g, d) for g, _, d in MULTIMODEL_GENERATORS).get(label, "")
    seeds = [f for f in sorted(glob.glob(os.path.join(eval_dir, "seed*.csv"))) if "summary" not in f]
    if not posts_csv or not os.path.isfile(posts_csv) or not seeds:
        print(f"[model-linearity] skip {label}: no test posts or eval CSVs")
        return None
    marker = "o"  # one model per figure, so the marker encodes nothing; keep both figures alike
    x = np.arange(len(_MM_BANDS))
    steps = [f"{a} → {b}" for a, b in zip(_MM_BAND_SHORT[:-1], _MM_BAND_SHORT[1:])]

    seen = set()  # personas the reviewer read during prompt editing
    if exclude_prompt_iter_personas:
        for f in glob.glob(MULTIMODEL_PROMPT_ITER_POSTS):
            seen |= set(pd.read_csv(f)["persona"].astype(str))

    fig = plt.figure(figsize=(7.5, 2.40))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2], wspace=0.45)
    inner_a = gs[0, 0].subgridspec(2, 1, hspace=0.10)
    ax_lin = fig.add_subplot(inner_a[0, 0])
    ax_emp = fig.add_subplot(inner_a[1, 0], sharex=ax_lin)
    ax_grade = fig.add_subplot(gs[0, 2])
    inner = gs[0, 1].subgridspec(len(MODEL_FIG_ASSESSORS), 1, hspace=0.10)
    bias_axes = {}
    for i, name in enumerate(MODEL_FIG_ASSESSORS):
        bias_axes[name] = fig.add_subplot(inner[i, 0],
                                          sharex=bias_axes[MODEL_FIG_ASSESSORS[0]] if i else None)
    rows, used_prompts = [], set()

    # (a) the SA PHQ-9-axis ladder, one line per generation prompt.
    for suffix, subdir in MULTIMODEL_SA_PROMPTS:
        ladder = phq9_adjacent_band_ladder(sa_roots.get(label, ""), emb_name=emb_name, subdir=subdir)
        if ladder is None or ladder.empty:
            continue
        _errorbar_annotated(ax_lin, ladder["cos"].to_numpy(), ladder["std"].to_numpy(),
                            MULTIMODEL_PROMPT_COLOURS[suffix], "-", marker,
                            MULTIMODEL_PROMPT_LABELS[suffix], annotate=False)
        used_prompts.add(suffix)
        rows += [{"generator": label, "prompt": subdir, "step": r.step, "cosine": r.cos, "std": r.std,
                  "n_reps": r.n_reps, "n_posts": r.n_anchors} for r in ladder.itertuples(index=False)]

    # (b) bottom row: the regressor fine-tuned on this generator's own posts, per arm.
    ax_bert = bias_axes["MentalBERT+MLP"]
    arms = [("human-opt.", posts_csv, seeds)]
    if label in MULTIMODEL_MINIMAL_ARMS:
        m_csv, m_dir = MULTIMODEL_MINIMAL_ARMS[label]
        m_seeds = [f for f in sorted(glob.glob(os.path.join(m_dir, "seed*.csv"))) if "summary" not in f]
        if os.path.isfile(m_csv) and m_seeds:
            arms.append(("minimal", m_csv, m_seeds))
    for prompt_key, arm_csv, arm_seeds in arms:
        colour = MULTIMODEL_PROMPT_COLOURS[prompt_key]
        used_prompts.add(prompt_key)
        persona_of = pd.read_csv(arm_csv).groupby("agent_id")["persona"].first().astype(str)
        keep_ids = set(persona_of.index[~persona_of.isin(seen)])
        bias = _band_bias_per_seed(arm_seeds, keep_ids)
        print(f"[model-linearity] {short} BERT ({prompt_key} posts): {len(keep_ids)} of "
              f"{len(persona_of)} blocks, {len(bias)} seeds")
        ax_bert.fill_between(x, bias.mean(0) - bias.std(0), bias.mean(0) + bias.std(0),
                             color=colour, alpha=0.2, linewidth=0)
        ax_bert.plot(x, bias.mean(0), "-", marker=marker, color=colour, linewidth=1.6,
                     markersize=5.5, markeredgecolor="black", markeredgewidth=0.5, zorder=3)

    # (b) top row: the prompted LLM on the same blocks.
    ax_llm = bias_axes["LLM assessor"]
    for gen, prompt_key, assess_key, pattern, llm_seeds, blocks_csv, select in MODEL_FIG_LLM_ASSESSORS:
        if gen != label or not os.path.isfile(blocks_csv):
            continue
        best = _textgrad_best_seed(pattern, llm_seeds) if select == "best" else None
        if select == "best" and best is None:
            continue
        use = [best] if select == "best" else llm_seeds
        blocks = pd.read_csv(blocks_csv).groupby("agent_id", sort=False)["persona"].first().astype(str)
        keep_ids = set(blocks.index[~blocks.isin(seen)])
        bias = _band_bias_per_seed([pattern.format(seed=s) for s in use], keep_ids,
                                   order_ids=blocks.index.to_numpy())
        if not len(bias):
            continue
        colour = MULTIMODEL_PROMPT_COLOURS[prompt_key]
        used_prompts.add(prompt_key)
        print(f"[model-linearity] {short} LLM ({prompt_key} posts): {len(keep_ids)} of "
              f"{len(blocks)} blocks, seeds {use}")
        if len(bias) > 1:  # a single run has no across-seed spread to show
            ax_llm.fill_between(x, bias.mean(0) - bias.std(0), bias.mean(0) + bias.std(0),
                                color=colour, alpha=0.2, linewidth=0)
        ax_llm.plot(x, bias.mean(0), "-", marker=marker, color=colour, linewidth=1.6,
                    markersize=5.5, markeredgecolor="black", markeredgewidth=0.5, zorder=3)

    # (c) teacher gradings of the same 300-block sets, one line per generation prompt.
    for prompt_key in MULTIMODEL_PROMPT_STYLES:
        path = gradings.get((label, prompt_key))
        if not path or not os.path.isfile(path):
            print(f"[model-linearity] no gradings: {short} ({prompt_key}) {path}")
            continue
        d = pd.read_csv(path)
        d = d[~d["persona"].astype(str).isin(seen)].dropna(subset=["score"])
        by_band = [d["score"][d["phq9"].between(lo, hi)] for lo, hi, _ in _MM_BANDS]
        mean = np.array([g.mean() for g in by_band])
        sem = np.array([g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0 for g in by_band])
        used_prompts.add(prompt_key)
        print(f"[model-linearity] {short} grades ({prompt_key}): {len(d)} blocks, mean {d['score'].mean():.2f}")
        _errorbar_annotated(ax_grade, mean, sem, MULTIMODEL_PROMPT_COLOURS[prompt_key], "-",
                            marker, MULTIMODEL_PROMPT_LABELS[prompt_key], annotate=False)

    e = empirical_ladder
    ax_emp.errorbar(np.arange(len(e)), e["cos"], yerr=[e["cos"] - e["ci_lo"], e["ci_hi"] - e["cos"]], fmt="s",
                    linestyle="-", color="#222222", capsize=2.5, linewidth=1.6, markersize=5,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=3)
    ax_emp.set_xticks(np.arange(len(steps)))
    ax_emp.set_xticklabels(steps, rotation=30, ha="right", fontsize=8)
    ax_lin.tick_params(labelbottom=False)
    ax_grade.set_ylabel("Teacher score (0–10)", fontsize=9.5)
    # Short tick labels: with a single arm plotted the range is narrow enough that
    # matplotlib picks two decimals, and those run left into (b).
    ax_grade.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    for ax in (*bias_axes.values(), ax_grade):
        ax.set_xticks(x)
        ax.set_xticklabels(_MM_BAND_SHORT, rotation=30, ha="right", fontsize=8)
    for ax in (ax_lin, ax_emp, ax_grade, *bias_axes.values()):
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.12, y=0.15)
    for ax, name in ((ax_lin, "Synthetic"), (ax_emp, "Twitter users")):  # (a) rows, styled like (b)'s
        ax.autoscale_view()
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.26 * (y1 - y0))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.text(0.03, 0.93, name, transform=ax.transAxes, ha="left", va="top",
                fontsize=7, fontweight="bold", color="0.15",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.6))
    for i, name in enumerate(MODEL_FIG_ASSESSORS):  # the label IS the row's key
        ax = bias_axes[name]
        ax.axhline(0, color="black", linewidth=0.8)
        # Each row keeps its OWN scale. The two assessors differ by an order of magnitude
        # on Gemma (LLM ~-6, regressor ~-2..+1), and one shared scale flattened the
        # regressor row; the row labels and the zero line keep the two readable apart.
        ax.autoscale_view()
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.26 * (y1 - y0))  # room above the curves for the row's label
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5]))
        ax.text(0.03, 0.93, name, transform=ax.transAxes, ha="left", va="top",
                fontsize=7, fontweight="bold", color="0.15",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.6))
        if i < len(MODEL_FIG_ASSESSORS) - 1:  # shared band axis: only the bottom row is labelled
            ax.tick_params(labelbottom=False)

    # One y-label for the (b) stack, centred on it and clear of the widest tick label.
    bias_pos = [ax.get_position() for ax in bias_axes.values()]
    fig.canvas.draw()
    tick_left = min(lb.get_window_extent(fig.canvas.get_renderer()).x0 for ax in bias_axes.values()
                    for lb in ax.get_yticklabels() if lb.get_text())
    fig.text(fig.transFigure.inverted().transform((tick_left - 6, 0))[0],
             (max(b.y1 for b in bias_pos) + min(b.y0 for b in bias_pos)) / 2,
             "Bias = mean(pred − true)", rotation=90, ha="right", va="center", fontsize=9.5)
    a_pos = [ax.get_position() for ax in (ax_lin, ax_emp)]
    tick_left_a = min(lb.get_window_extent(fig.canvas.get_renderer()).x0 for ax in (ax_lin, ax_emp)
                      for lb in ax.get_yticklabels() if lb.get_text())
    fig.text(fig.transFigure.inverted().transform((tick_left_a - 6, 0))[0],
             (max(p.y1 for p in a_pos) + min(p.y0 for p in a_pos)) / 2,
             "Cosine similarity", rotation=90, ha="right", va="center", fontsize=9.5)

    # One legend: within a figure the only thing a line encodes is the generation prompt.
    handles = [Line2D([], [], color=MULTIMODEL_PROMPT_COLOURS[k], marker=marker, linewidth=2.2,
                      markersize=5.5, markeredgecolor="black", markeredgewidth=0.5,
                      label=MULTIMODEL_PROMPT_LABELS[k])
               for k in MULTIMODEL_PROMPT_STYLES if k in used_prompts]
    leg = fig.legend(handles=handles, title="Generation prompt", ncol=len(handles), loc="lower left",
                     bbox_to_anchor=(0, 1.01), borderaxespad=0, fontsize=7.5, framealpha=0.9,
                     handlelength=2.0, handletextpad=0.5, borderpad=0.4, columnspacing=1.2)
    leg.get_title().set_fontsize(7.5)
    leg.get_title().set_fontweight("bold")
    fig.canvas.draw()
    inv = fig.transFigure.inverted()

    columns = [([ax_lin, ax_emp], "(a) Adjacent-band cosine"),
               (list(bias_axes.values()), "(b) Assessor bias per band"),
               ([ax_grade], "(c) Post gradings per band")]
    cap_y = min(a.get_tightbbox(fig.canvas.get_renderer()).transformed(inv).y0
                for axes, _ in columns for a in axes) - 0.012
    for axes, panel in columns:
        pos = [a.get_position() for a in axes]
        fig.text((min(b.x0 for b in pos) + max(b.x1 for b in pos)) / 2, cap_y, panel,
                 ha="center", va="top", fontsize=9.5)

    box = leg.get_window_extent().transformed(inv)  # centre the legend above the panels
    y = max(a.get_position().y1 for a in (ax_lin, ax_emp, ax_grade, *bias_axes.values())) + 0.04
    leg.set_bbox_to_anchor(((1 - box.width) / 2, y), transform=fig.transFigure)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.splitext(out_path)[0] + ".csv", index=False)
    print(f"Per-model linearity/bias/grading plot with empirical ladder → {out_path}")
    return out_path
