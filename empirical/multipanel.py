"""The paper's multi-model figure (synthetic) with the real users' adjacent-band ladder added under (a).

`plot_multimodel_with_empirical` is a copy of `visualization.plot_multimodel_linearity_bias`
(so the co-author's module stays untouched) whose panel (a) is split into a synthetic and
an empirical row, the way (b) is split by generator. Called from `figures.ipynb`.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.visualization import (MULTIMODEL_ASSESS_STYLES, MULTIMODEL_GENERATORS, MULTIMODEL_GRADINGS, MULTIMODEL_GRADING_PERSONAS, MULTIMODEL_LLM_ASSESSORS, MULTIMODEL_MINIMAL_ARMS, MULTIMODEL_MODEL_MARKERS, MULTIMODEL_MODEL_STYLES, MULTIMODEL_PROMPT_COLOURS, MULTIMODEL_PROMPT_ITER_POSTS, MULTIMODEL_PROMPT_LABELS, MULTIMODEL_PROMPT_STYLES, MULTIMODEL_SA_PROMPTS, MULTIMODEL_SA_ROOTS, MULTIMODEL_TEACHER_ASSESSORS, MULTIMODEL_TEST_POSTS, _MM_BANDS, _MM_BAND_SHORT, _band_bias_per_seed, _errorbar_annotated, _textgrad_best_seed)


def plot_multimodel_with_empirical(out_path: str, empirical_ladder: pd.DataFrame,
                                   exclude_prompt_iter_personas: bool = False,
                                   teacher_assessors: list = None,
                                   emb_name: str = "embeddings_sbert.npz",
                                   sa_roots: dict = None):
    """`visualization.plot_multimodel_linearity_bias` with the real users' ladder under panel (a).

    A copy of the paper's multi-model figure (so `visualization.py` stays untouched):
    (a) is split like (b) into a stack of two rows sharing the step axis, each on its own
    y-scale: the synthetic adjacent-band cosine on top, exactly as in the original, and
    the empirical matched next-band cosine (`empirical/ladder.py`, bootstrap 95% CI)
    below. (b) and (c) are unchanged. See the original's docstring for (a)-(c).

    Args:
        out_path (str): PNG to write; the synthetic (a) numbers go next to it as `<stem>.csv`.
        empirical_ladder (pd.DataFrame): `ladder_matched.csv` (step, cos, ci_lo, ci_hi).
        exclude_prompt_iter_personas, teacher_assessors, emb_name, sa_roots: as in the original.

    Returns:
        str | None: `out_path`, or None if no generator had data.
    """
    from utils.sensitivity.sa_analyze import phq9_adjacent_band_ladder
    from matplotlib.lines import Line2D
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    steps = [f"{a} → {b}" for a, b in zip(_MM_BAND_SHORT[:-1], _MM_BAND_SHORT[1:])]
    x = np.arange(len(_MM_BANDS))
    sa_roots = MULTIMODEL_SA_ROOTS if sa_roots is None else sa_roots
    # Which generators have data: each gets its own row of panel (b), so this is settled
    # before the figure is built.
    active = []
    for label, colour, eval_dir in MULTIMODEL_GENERATORS:
        short, posts_csv = MULTIMODEL_TEST_POSTS.get(label, (None, None))
        seeds = [f for f in sorted(glob.glob(os.path.join(eval_dir, "seed*.csv"))) if "summary" not in f]
        if not posts_csv or not os.path.isfile(posts_csv) or not seeds:
            print(f"[multimodel-linearity] skip {label}: no test posts or eval CSVs")
            continue
        active.append((label, colour, short, posts_csv, seeds))
    if not active:
        return None

    # (b) is a stack of sub-panels, one per generator, each holding that generator's
    # fine-tuned regressor and its prompted LLM assessor (the two sat far enough apart on
    # a shared y-axis to hide each other only while the optimized assessment prompt was
    # drawn too). They share the band axis (only the bottom row is labelled), one y-label
    # centred on the stack and one y-scale.
    llm_rows = [r for r in MULTIMODEL_LLM_ASSESSORS
                if r[0] in {label for label, *_ in active} and os.path.isfile(r[5])
                and any(os.path.isfile(r[3].format(seed=sd)) for sd in r[4])]
    bias_rows = [label for label, *_ in active]
    fig = plt.figure(figsize=(7.5, 1.15 * len(bias_rows) + 0.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2], wspace=0.45)
    inner_a = gs[0, 0].subgridspec(2, 1, hspace=0.10)
    ax_lin = fig.add_subplot(inner_a[0, 0])
    ax_emp = fig.add_subplot(inner_a[1, 0], sharex=ax_lin)
    ax_grade = fig.add_subplot(gs[0, 2])
    gp = ax_grade.get_position()  # the (a)-(b) gap holds (b)'s shared y-label, the (b)-(c) gap
    ax_grade.set_position([gp.x0 - 0.025, gp.y0, gp.width, gp.height])  # only (c)'s own labels
    inner = gs[0, 1].subgridspec(len(bias_rows), 1, hspace=0.10)
    bias_axes = {}
    for i, key in enumerate(bias_rows):
        bias_axes[key] = fig.add_subplot(inner[i, 0], sharex=bias_axes[bias_rows[0]] if i else None)
    rows = []
    used_gens, used_prompts, teacher_handles = [], set(), []  # what the shared legend has to cover
    seen = set()  # personas the reviewer read during prompt editing
    if exclude_prompt_iter_personas:
        for f in glob.glob(MULTIMODEL_PROMPT_ITER_POSTS):
            seen |= set(pd.read_csv(f)["persona"].astype(str))

    for label, _, short, posts_csv, seeds in active:
        marker = MULTIMODEL_MODEL_MARKERS.get(label, "o")  # the model is the marker everywhere
        mstyle = MULTIMODEL_MODEL_STYLES.get(label, "-")   # and the line outside panel (b)
        ax_bias = bias_axes[label]

        # (a) the SA PHQ-9-axis ladder, one line per prompt (skipped if that SA was not run/embedded).
        for suffix, subdir in MULTIMODEL_SA_PROMPTS:
            ladder = phq9_adjacent_band_ladder(sa_roots.get(label, ""),
                                               emb_name=emb_name, subdir=subdir)
            if ladder is None or ladder.empty:
                continue
            _errorbar_annotated(ax_lin, ladder["cos"].to_numpy(), ladder["std"].to_numpy(),
                                MULTIMODEL_PROMPT_COLOURS[suffix], mstyle, marker,
                                f"{short} ({suffix} prompt)", annotate=False)
            used_prompts.add(suffix)
            rows += [{"generator": label, "prompt": subdir, "step": r.step, "cosine": r.cos, "std": r.std,
                      "n_reps": r.n_reps, "n_anchors": r.n_anchors} for r in ladder.itertuples(index=False)]

        # (b) the regressor fine-tuned on this generator's posts, per prompt arm.
        arms = [("human-opt.", posts_csv, seeds)]
        if label in MULTIMODEL_MINIMAL_ARMS:
            m_csv, m_dir = MULTIMODEL_MINIMAL_ARMS[label]
            m_seeds = [f for f in sorted(glob.glob(os.path.join(m_dir, "seed*.csv"))) if "summary" not in f]
            if os.path.isfile(m_csv) and m_seeds:
                arms.append(("minimal", m_csv, m_seeds))
        for prompt_key, arm_csv, arm_seeds in arms:
            # In (b) the line is the assessor, so every regressor arm is solid and the row's
            # "Assessor" key holds for all of them; the prompt is in the colour.
            style, colour = MULTIMODEL_ASSESS_STYLES["bert"], MULTIMODEL_PROMPT_COLOURS[prompt_key]
            used_prompts.add(prompt_key)
            persona_of = pd.read_csv(arm_csv).groupby("agent_id")["persona"].first().astype(str)
            keep_ids = set(persona_of.index[~persona_of.isin(seen)])
            print(f"[multimodel-linearity] {short} ({prompt_key}): {len(keep_ids)} of {len(persona_of)} blocks kept")
            bias = _band_bias_per_seed(arm_seeds, keep_ids)
            ax_bias.fill_between(x, bias.mean(0) - bias.std(0), bias.mean(0) + bias.std(0),
                                 color=colour, alpha=0.2, linewidth=0)
            ax_bias.plot(x, bias.mean(0), linestyle=style, marker=marker, color=colour, linewidth=1.6,
                         markersize=5.5, markeredgecolor="black", markeredgewidth=0.5, zorder=3,
                         label=f"FT MentalBERT+MLP, {short} posts"
                               + (f" ({prompt_key} prompt)" if prompt_key != "human-opt." else ""))
        used_gens.append((short, marker, mstyle))  # "Qwen" / "Gemma", not the full model id

    # (b) the LLM assessor on the same blocks, in the same row as that generator's regressor.
    llm_axes = set()
    for gen, prompt_key, assess_key, pattern, seeds, blocks_csv, select in llm_rows:
        style, marker = MULTIMODEL_ASSESS_STYLES[assess_key], MULTIMODEL_MODEL_MARKERS.get(gen, "o")
        name = MULTIMODEL_PROMPT_LABELS[assess_key]
        best = _textgrad_best_seed(pattern, seeds) if select == "best" else None
        if select == "best" and best is None:
            continue
        use = [best] if select == "best" else seeds
        blocks = pd.read_csv(blocks_csv).groupby("agent_id", sort=False)["persona"].first().astype(str)
        keep_ids = set(blocks.index[~blocks.isin(seen)])
        bias = _band_bias_per_seed([pattern.format(seed=s) for s in use], keep_ids,
                                   order_ids=blocks.index.to_numpy())
        if not len(bias):
            continue
        short, colour = MULTIMODEL_TEST_POSTS.get(gen, (gen,))[0], MULTIMODEL_PROMPT_COLOURS[prompt_key]
        ax_bias = bias_axes[gen]
        llm_axes.add(gen)
        print(f"[multimodel-linearity] {short} LLM assessor, {name} ({prompt_key} posts): "
              f"{len(keep_ids)} of {len(blocks)} blocks kept, seeds {use}")
        if len(bias) > 1:  # a single run has no across-seed spread to show
            ax_bias.fill_between(x, bias.mean(0) - bias.std(0), bias.mean(0) + bias.std(0),
                                 color=colour, alpha=0.15, linewidth=0)
        ax_bias.plot(x, bias.mean(0), linestyle=style, marker=marker, color=colour, linewidth=1.4,
                     markersize=5, markeredgecolor="black", markeredgewidth=0.5, zorder=2,
                     label=f"LLM {name}, {short} posts")

    # The rows that hold both assessors say which line is which; a row with the regressor
    # alone needs no key, since colour already means the generator.
    for gen in llm_axes:
        handles = [Line2D([], [], color="0.35", linestyle=MULTIMODEL_ASSESS_STYLES["bert"],
                          linewidth=1.6, label="MentalBERT+MLP"),
                   Line2D([], [], color="0.35", linestyle=MULTIMODEL_ASSESS_STYLES["minimal"],
                          linewidth=1.6, label="LLM")]
        leg = bias_axes[gen].legend(handles=handles, title="Assessor", loc="lower left",
                                    fontsize=6, framealpha=0.9, handlelength=2.4,
                                    handletextpad=0.4, borderpad=0.3, labelspacing=0.25)
        leg.get_title().set_fontsize(6)
        leg.get_title().set_fontweight("bold")

    ax_bias = bias_axes[active[0][0]]  # teacher-set lines: not an LLM row, so the first regressor row
    for label, colour, style, marker, pattern, seeds in (
            MULTIMODEL_TEACHER_ASSESSORS if teacher_assessors is None else teacher_assessors):
        bias = _band_bias_per_seed([pattern.format(seed=s) for s in seeds])
        if not len(bias):
            continue
        ax_bias.fill_between(x, bias.mean(0) - bias.std(0), bias.mean(0) + bias.std(0),
                             color=colour, alpha=0.15, linewidth=0)
        ax_bias.plot(x, bias.mean(0), linestyle=style, marker=marker, color=colour, linewidth=1.4,
                     markersize=5, markeredgecolor="black", markeredgewidth=0.5, zorder=2, label=label)
        teacher_handles.append(Line2D([], [], color=colour, linestyle=style, marker=marker, linewidth=1.4,
                                      markersize=5, markeredgecolor="black", markeredgewidth=0.5, label=label))

    for gen, prompt_key, path in MULTIMODEL_GRADINGS:
        if not os.path.isfile(path):
            print(f"[multimodel-linearity] no gradings: {path}")
            continue
        short = MULTIMODEL_TEST_POSTS.get(gen, (gen,))[0]
        label, colour = f"{short} ({prompt_key} prompt)", MULTIMODEL_PROMPT_COLOURS[prompt_key]
        style = MULTIMODEL_MODEL_STYLES.get(gen, "-")
        marker = MULTIMODEL_MODEL_MARKERS.get(gen, "o")
        used_prompts.add(prompt_key)
        d = pd.read_csv(path)
        if "persona" not in d.columns:  # iter_0 raw scores: agent_id indexes the eval-1000 persona file
            d["persona"] = pd.read_csv(MULTIMODEL_GRADING_PERSONAS)["persona"].reindex(d["agent_id"]).to_numpy()
        d = d[~d["persona"].astype(str).isin(seen)].dropna(subset=["score"])
        by_band = [d["score"][d["phq9"].between(lo, hi)] for lo, hi, _ in _MM_BANDS]
        mean = np.array([g.mean() for g in by_band])
        sem = np.array([g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0 for g in by_band])
        _errorbar_annotated(ax_grade, mean, sem, colour, style, marker, label, annotate=False)  # four lines: no value labels

    e = empirical_ladder
    xs = np.arange(len(e))
    ax_emp.errorbar(xs, e["cos"], yerr=[e["cos"] - e["ci_lo"], e["ci_hi"] - e["cos"]], fmt="s",
                    linestyle="-", color="#222222", capsize=2.5, linewidth=1.6, markersize=5,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=3)
    ax_emp.set_xticks(np.arange(len(steps)))
    ax_emp.set_xticklabels(steps, rotation=30, ha="right", fontsize=8)
    ax_lin.tick_params(labelbottom=False)
    for ax, name in ((ax_lin, "Synthetic"), (ax_emp, "Twitter users")):  # each row on its own scale
        ax.text(0.03, 0.93, name, transform=ax.transAxes, ha="left", va="top", fontsize=7,
                fontweight="bold", color="0.15", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.6))
    ax_grade.set_ylabel("Teacher score (0–10)", fontsize=9.5)
    for ax in (*bias_axes.values(), ax_grade):
        ax.set_xticks(x)
        ax.set_xticklabels(_MM_BAND_SHORT, rotation=30, ha="right", fontsize=8)
    for i, key in enumerate(bias_rows):  # each row: zero line, and the band axis on the last one
        ax = bias_axes[key]
        ax.axhline(0, color="black", linewidth=0.8)
        if i < len(bias_rows) - 1:  # shared band axis: only the bottom row is labelled
            ax.tick_params(labelbottom=False)
    for ax in (ax_lin, ax_emp, ax_grade, *bias_axes.values()):
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.12, y=0.15)
    for ax in (ax_lin, ax_emp):
        ax.autoscale_view()
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.30 * (y1 - y0))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    lims = [ax.get_ylim() for ax in bias_axes.values()]  # every row of (b) on one scale
    lo, hi = min(lo for lo, _ in lims), max(hi for _, hi in lims)
    lo -= (0.15 if llm_axes else 0) * (hi - lo)  # room under the curves for the row's own legend
    for ax in bias_axes.values():
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5]))  # the wider
        # range would otherwise be left with ticks at 0 and -5 only

    # One y-label for the whole (b) stack, centred on it (the rows are shorter than the
    # other two panels, so this cannot ride on one row's ylabel).
    bias_pos = [ax.get_position() for ax in bias_axes.values()]
    fig.canvas.draw()  # the label has to clear whichever row has the widest tick labels
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

    # One legend for the whole figure, split into the two things a line encodes: marker =
    # generative model, colour = generation prompt. Each group is its own titled legend,
    # laid out left to right above the panels.
    groups = [("Generative model",
               [Line2D([], [], color="0.35", linestyle=style, marker=marker, linewidth=1.6, markersize=6,
                       markeredgecolor="black", markeredgewidth=0.5, label=short)
                for short, marker, style in used_gens]),
              ("Generation prompt",
               [Line2D([], [], color=MULTIMODEL_PROMPT_COLOURS[key], linewidth=2.6,
                       label=MULTIMODEL_PROMPT_LABELS[key])
                for key in MULTIMODEL_PROMPT_STYLES if key in used_prompts])]
    if teacher_handles:
        groups.append(("Teacher-set assessor", teacher_handles))
    legs = [fig.legend(handles=handles, title=title, ncol=len(handles), loc="lower left",
                       bbox_to_anchor=(0, 1.01), borderaxespad=0, fontsize=7.5, framealpha=0.9,
                       handlelength=2.0, handletextpad=0.5, borderpad=0.4, columnspacing=1.2)
            for title, handles in groups if handles]
    for leg in legs:
        leg.get_title().set_fontsize(7.5)
        leg.get_title().set_fontweight("bold")
    fig.canvas.draw()  # measure the legends, then centre them as a block just above the panels
    inv = fig.transFigure.inverted()

    # Panel captions, all on one baseline just under the deepest tick label of the three
    # columns (measured, not a fraction of panel height: the (b) rows are half-height).
    columns = [([ax_lin, ax_emp], "(a) Adjacent-band cosine"),
               (list(bias_axes.values()), "(b) Assessor bias per band"),
               ([ax_grade], "(c) Post gradings per band")]
    tight = {id(a): a.get_tightbbox(fig.canvas.get_renderer()).transformed(inv)
             for axes, _ in columns for a in axes}
    cap_y = min(tight[id(a)].y0 for axes, _ in columns for a in axes) - 0.012
    for axes, panel in columns:
        boxes_c = [tight[id(a)] for a in axes]
        fig.text((min(b.x0 for b in boxes_c) + max(b.x1 for b in boxes_c)) / 2, cap_y, panel,
                 ha="center", va="top", fontsize=10.5)

    boxes = [leg.get_window_extent().transformed(inv) for leg in legs]
    gap = 0.025
    y = max(a.get_position().y1 for a in (ax_lin, ax_emp, ax_grade, *bias_axes.values())) + 0.04
    total = sum(b.width for b in boxes) + gap * (len(legs) - 1)
    if total <= 1:  # they fit side by side on one row
        left = (1 - total) / 2
        for leg, b in zip(legs, boxes):
            leg.set_bbox_to_anchor((left, y), transform=fig.transFigure)
            left += b.width + gap
    else:  # too wide (the SI version with the teacher-set rows): one centred row per group
        for leg, b in zip(reversed(legs), reversed(boxes)):  # first group on top
            leg.set_bbox_to_anchor(((1 - b.width) / 2, y), transform=fig.transFigure)
            y += b.height + 0.02
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.splitext(out_path)[0] + ".csv", index=False)
    print(f"Multi-model linearity/bias/grading plot with empirical ladder → {out_path}")
    return out_path
