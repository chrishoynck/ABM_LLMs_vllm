"""Prompt-axis sensitivity: paired per-(agent, round) cosine across prompt variants.

This is the prompt counterpart to the sampling-axis sweep in ``sa_analyze.py``.
Here EVERYTHING except the post-generation instruction is held fixed at
generation time:

    * same personas        — ``--agent-seed`` fixed across variants
    * same neighbour posts  — ``--neighbor-seed`` fixed across variants
    * same per-slot PHQ-9   — falls out of the fixed agent-seed
    * LLM left UNSEEDED      — ``--nondeterministic`` (one free draw per variant)

So for a fixed ``(agent_id, round)`` anchor, the only systematic difference
between two variant CSVs is the prompt. The cosine between their post
embeddings at that anchor is the prompt's effect on the output (with one draw
of LLM noise mixed in).

We deliberately do NOT compute a within-variant LLM-noise baseline here — that
comes from the sampling-axis runs of the other models. And we keep EVERY
per-anchor, per-variant-pair cosine: the master CSV is one tidy row per
``(anchor, variant_a, variant_b)``.

Two figures are produced:

  * BOX PLOT  — each variant vs one or more REFERENCE prompts (``--box-against``,
                default ``minimal``; e.g. also the human-in-the-loop ``iter_10``).
                Full per-anchor distribution per (reference, variant) — no mean.
  * HEATMAP   — N×N pairwise output similarity across prompts. A cell must
                collapse the anchor distribution to one number; we use the
                MEDIAN per-anchor cosine. Drop prompts from it with
                ``--heatmap-exclude`` (e.g. exclude ``iter_10``).

Reads the ``SA_prompt`` layout produced by
``utils.create_data.generate_test_data`` (``--instruction-dir`` mode):

    <sa_dir>/<instr_id>_<model>.csv
        columns: agent_id, persona, age, step, phq9, tweet, interaction

To generate several variants with everything-but-the-prompt fixed::

    PYTHONPATH=src python -m utils.create_data.generate_test_data \\
        --instruction-dir <dir of prompt .txt files> --filename-pattern '*.txt' \\
        --persona-phq9-file data/personas_eval_1000_phq9.csv \\
        --model qwen27 --num_agents 120 --check_point 10 \\
        --agent-seed 42 --neighbor-seed 42 --num-neighbors 5 --nondeterministic

Then analyse (box vs minimal AND iter_10, heatmap without iter_10)::

    PYTHONPATH=src python -m utils.sensitivity.sa_prompt \\
        --sa-dir data/prompt_optimization_h/qwen27_baseline/SA_prompt \\
        --box-against minimal iter_10 --heatmap-exclude iter_10
"""

from __future__ import annotations

import argparse
import glob
import os
from itertools import combinations

import numpy as np
import pandas as pd

from utils.metrics import generate_sbert_model

from .sa_analyze import cosine_rows, phq9_to_band
from .sa_embed import encode_run

# Reference / category colour cycle for the box plot (one per reference prompt).
_REF_COLOURS = ["#3498db", "#d96907", "#9b59b6", "#2ecc71"]


def find_variants(sa_dir: str, exclude: list[str] | None = None) -> list[tuple[str, str]]:
    """Return sorted ``[(label, csv_path)]`` — one per prompt variant.

    Every ``*.csv`` directly under ``sa_dir`` is a variant, except the
    bookkeeping ``scores.csv``. The label is the filename stem. Any stem
    containing a token in ``exclude`` is skipped (e.g. a CSV shared with other
    pipelines that happens to sit in the same dir).
    """
    exclude = exclude or []
    out = []
    for p in sorted(glob.glob(os.path.join(sa_dir, "*.csv"))):
        base = os.path.basename(p)
        if base == "scores.csv":
            continue
        label = os.path.splitext(base)[0]
        if any(tok in label for tok in exclude):
            print(f"[prompt-sa] discovery: skipping {base} (matches --exclude)")
            continue
        out.append((label, p))
    return out


def embed_variant(model, csv_path: str, force: bool = False) -> dict:
    """Encode one variant CSV's posts, caching to ``<stem>_emb.npz`` alongside it."""
    cache = os.path.splitext(csv_path)[0] + "_emb.npz"
    if os.path.exists(cache) and not force:
        data = np.load(cache, allow_pickle=True)
        return {k: data[k] for k in data.files}
    data = encode_run(model, csv_path)          # reuses sa_embed (tweet/agent_id/step/phq9)
    # Keep the persona string per row too: it's the strongest cross-variant
    # seed-consistency check (slot i must be the same persona in every variant).
    data["personas"] = pd.read_csv(csv_path)["persona"].astype(str).values
    np.savez_compressed(cache, **data)
    return data


def _index_by_anchor(data: dict) -> dict:
    """Map ``(agent_id, round) -> row index``. First wins on duplicate anchors."""
    idx: dict[tuple[int, int], int] = {}
    for i, (a, r) in enumerate(zip(data["agent_ids"], data["rounds"])):
        key = (int(a), int(r))
        if key not in idx:                      # one post per anchor expected
            idx[key] = i
    return idx


def _resolve_labels(tokens: list[str], labels: list[str]) -> list[str]:
    """Resolve each token to a label: exact match, else unique substring.

    Missing tokens warn and are skipped (so defaults degrade gracefully before
    all prompts exist); ambiguous tokens are a hard error.
    """
    resolved = []
    for tok in tokens:
        if tok in labels:
            resolved.append(tok)
            continue
        matches = [l for l in labels if tok in l]
        if len(matches) == 0:
            print(f"[prompt-sa] WARNING: {tok!r} matched no variant — skipping. "
                  f"Available: {labels}")
        elif len(matches) > 1:
            raise SystemExit(f"[prompt-sa] {tok!r} is ambiguous: matched {matches}")
        else:
            resolved.append(matches[0])
    return resolved


def pairwise_anchor_cosines(variants: dict) -> pd.DataFrame:
    """One row per ``(anchor, variant_a, variant_b)`` cosine over ALL variant
    pairs. No averaging. Selection for the box plot / heatmap happens downstream.
    """
    labels = list(variants)
    indexed = {lab: (variants[lab], _index_by_anchor(variants[lab])) for lab in labels}

    # Anchors present in EVERY variant (same personas → should be all of them).
    common = None
    for _, idx in indexed.values():
        keys = set(idx.keys())
        common = keys if common is None else common & keys
    common = sorted(common or [])
    print(f"[prompt-sa] {len(labels)} variants, {len(common)} common (agent, round) anchors")

    # PHQ-9 must be identical per anchor across variants, else the prompt effect
    # is confounded with persona differences. Warn loudly.
    phq9_at, mismatch = {}, 0
    for (aid, rd) in common:
        vals = {int(indexed[lab][0]["phq9"][indexed[lab][1][(aid, rd)]]) for lab in labels}
        phq9_at[(aid, rd)] = next(iter(vals))
        if len(vals) > 1:
            mismatch += 1
    if mismatch:
        print(f"[prompt-sa] WARNING: PHQ-9 differs across variants at {mismatch}/"
              f"{len(common)} anchors — personas were NOT held fixed; the prompt "
              f"effect is confounded with persona differences there.")

    # Stronger seed-consistency check: the SAME persona must fill slot agent_id
    # in every variant. A mismatch means variants were generated with different
    # --agent-seed (slots not aligned) → the paired comparison is invalid.
    if all("personas" in indexed[lab][0] for lab in labels):
        persona_mismatch = sum(
            len({str(indexed[lab][0]["personas"][indexed[lab][1][(aid, rd)]])
                 for lab in labels}) > 1
            for (aid, rd) in common
        )
        if persona_mismatch:
            print(f"[prompt-sa] ERROR-LEVEL WARNING: persona differs across variants at "
                  f"{persona_mismatch}/{len(common)} anchors — variants were generated with "
                  f"different --agent-seed, so slot agent_id is NOT the same person across "
                  f"variants. Regenerate every variant (incl. iter_10) with the SAME "
                  f"--agent-seed/--neighbor-seed; the current comparison is invalid.")

    rows = []
    for (aid, rd) in common:
        phq9 = phq9_at[(aid, rd)]
        band = phq9_to_band(phq9)
        for la, lb in combinations(labels, 2):
            ea = indexed[la][0]["embeddings"][indexed[la][1][(aid, rd)]]
            eb = indexed[lb][0]["embeddings"][indexed[lb][1][(aid, rd)]]
            cs = float(cosine_rows(ea[None, :], eb[None, :])[0])
            rows.append({"agent_id": aid, "round": rd, "phq9": phq9, "band": band,
                         "variant_a": la, "variant_b": lb,
                         "pair": f"{la} vs {lb}", "cosine": cs})
    return pd.DataFrame(rows)


def box_rows_vs_references(df_all: pd.DataFrame, references: list[str]) -> pd.DataFrame:
    """Rows where exactly ONE side is a reference → tidy (reference, target, cosine).

    Reference↔reference pairs are dropped (both sides in the set), so each
    variant is compared only against the reference prompts.
    """
    ref_set = set(references)
    a_ref = df_all.variant_a.isin(ref_set) & ~df_all.variant_b.isin(ref_set)
    b_ref = df_all.variant_b.isin(ref_set) & ~df_all.variant_a.isin(ref_set)
    sub_a = df_all[a_ref].copy()
    sub_a["reference"], sub_a["target"] = sub_a["variant_a"], sub_a["variant_b"]
    sub_b = df_all[b_ref].copy()
    sub_b["reference"], sub_b["target"] = sub_b["variant_b"], sub_b["variant_a"]
    return pd.concat([sub_a, sub_b], ignore_index=True)


def build_heatmap_matrix(df_all: pd.DataFrame, labels: list[str]) -> np.ndarray:
    """N×N matrix of MEDIAN per-anchor cosine per prompt pair; diagonal = 1.0."""
    med = {frozenset((a, b)): float(g.cosine.median())
           for (a, b), g in df_all.groupby(["variant_a", "variant_b"])}
    n = len(labels)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            mat[i, j] = 1.0 if i == j else med.get(frozenset((a, b)), np.nan)
    return mat


def plot_box_vs_references(box_df: pd.DataFrame, references: list[str], out_path: str):
    """Grouped box plot: per variant, one box per reference prompt (coloured by
    reference). Full per-anchor distribution — no mean marker."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    refs = [r for r in references if r in set(box_df["reference"])]
    targets = sorted(box_df["target"].unique())
    if not refs or not targets:
        print("[prompt-sa] nothing to box-plot (no reference/target prompts).")
        return
    ref_colour = {r: _REF_COLOURS[i % len(_REF_COLOURS)] for i, r in enumerate(refs)}

    group_w = len(refs) + 1
    positions, data, colours, centres = [], [], [], []
    for t_i, t in enumerate(targets):
        for r_i, r in enumerate(refs):
            vals = box_df[(box_df.target == t) & (box_df.reference == r)].cosine.values
            if len(vals) == 0:
                continue
            positions.append(t_i * group_w + r_i)
            data.append(vals)
            colours.append(ref_colour[r])
        centres.append(t_i * group_w + (len(refs) - 1) / 2)

    fig_w = max(6.0, 0.9 * len(targets) * max(len(refs), 1) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    bp = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True,
                    showmeans=False,
                    medianprops=dict(color="black", linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.3,
                                    markeredgecolor="none", markerfacecolor="grey"))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.65); patch.set_edgecolor("black")

    ax.set_xticks(centres)
    ax.set_xticklabels(targets, rotation=20, ha="right")
    ax.set_ylabel("Per-anchor cosine similarity")
    ax.set_xlabel("Prompt variant")
    ax.set_title("Prompt sensitivity — each variant vs reference prompt(s)\n"
                 "(per-(agent, round) cosine; lower = output further from reference)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(handles=[Patch(facecolor=ref_colour[r], alpha=0.65, edgecolor="black",
                             label=f"vs {r}") for r in refs],
              loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_heatmap(mat: np.ndarray, labels: list[str], out_path: str):
    """Seaborn N×N heatmap of median per-anchor cosine across prompts."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = mat.shape[0]
    off = mat[~np.eye(n, dtype=bool)]
    vmin = float(np.nanmin(off)) if off.size and not np.all(np.isnan(off)) else 0.0
    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * n + 2), max(4.5, 0.9 * n + 1.5)))
    sns.heatmap(mat, ax=ax, xticklabels=labels, yticklabels=labels,
                vmin=vmin - 0.01, vmax=1.0, annot=True, fmt=".3f",
                annot_kws={"fontsize": 9}, cmap="Blues",
                linewidths=0.4, linecolor="white", square=True,
                cbar_kws={"label": "median per-anchor cosine"})
    ax.set_title("Prompt sensitivity — pairwise output similarity across prompts\n"
                 "(median per-(agent, round) cosine)")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sa-dir", required=True,
                        help="SA_prompt directory holding <instr_id>_<model>.csv variant files.")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write outputs (default: <sa-dir>/prompt_cosine).")
    parser.add_argument("--box-against", nargs="+", default=["minimal"],
                        help="Reference prompt label(s)/substring(s) the box plot compares every "
                             "other variant against (e.g. minimal iter_10). Default: minimal.")
    parser.add_argument("--heatmap-exclude", nargs="*", default=[],
                        help="Variant label(s)/substring(s) to drop from the heatmap "
                             "(e.g. iter_10). Default: none.")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Substring(s) of CSV stems to skip entirely during discovery "
                             "(e.g. a CSV shared with other pipelines sitting in --sa-dir).")
    parser.add_argument("--sbert", action="store_true",
                        help="Use SBERT all-MiniLM-L6-v2 instead of MentalBERT (default).")
    parser.add_argument("--force", action="store_true",
                        help="Re-encode variants whose <stem>_emb.npz cache exists.")
    args = parser.parse_args()

    variants = find_variants(args.sa_dir, exclude=args.exclude)
    if not variants:
        raise SystemExit(f"No variant CSVs under {args.sa_dir} (need <instr_id>_<model>.csv).")
    print(f"[prompt-sa] variants: {', '.join(lab for lab, _ in variants)}")

    out_dir = args.out_dir or os.path.join(args.sa_dir, "prompt_cosine")
    os.makedirs(out_dir, exist_ok=True)

    mentalbert = not args.sbert
    print(f"[prompt-sa] encoder = {'MentalBERT' if mentalbert else 'SBERT-MiniLM-L6-v2'}")
    model = generate_sbert_model(mentalbert=mentalbert)

    encoded = {}
    for lab, csv_path in variants:
        data = embed_variant(model, csv_path, force=args.force)
        encoded[lab] = data
        print(f"  [embed] {lab}: {data['embeddings'].shape[0]} posts, "
              f"dim {data['embeddings'].shape[1]}")

    if len(encoded) < 2:
        print("[prompt-sa] only 1 variant present — embeddings cached, but nothing to "
              "compare yet. Generate more prompt variants (see module docstring) and re-run.")
        return

    labels = list(encoded)
    df_all = pairwise_anchor_cosines(encoded)
    master_csv = os.path.join(out_dir, "prompt_variant_cosines.csv")
    df_all.to_csv(master_csv, index=False)
    print(f"[prompt-sa] {len(df_all)} per-anchor pair rows → {master_csv}")

    # ---- Box plot: each variant vs the reference prompt(s) ----
    references = _resolve_labels(args.box_against, labels)
    if references:
        box_df = box_rows_vs_references(df_all, references)
        if box_df.empty:
            print(f"[prompt-sa] no non-reference variants to compare against {references}.")
        else:
            box_df.to_csv(os.path.join(out_dir, "prompt_vs_reference_cosines.csv"), index=False)
            plot_box_vs_references(box_df, references,
                                   os.path.join(out_dir, "prompt_vs_reference_box.png"))
    else:
        print("[prompt-sa] no reference prompt resolved — skipping box plot.")

    # ---- Heatmap: pairwise across prompts, minus the excluded ones ----
    excluded = set(_resolve_labels(args.heatmap_exclude, labels)) if args.heatmap_exclude else set()
    keep = [l for l in labels if l not in excluded]
    if len(keep) >= 2:
        mat = build_heatmap_matrix(df_all, keep)
        plot_heatmap(mat, keep, os.path.join(out_dir, "prompt_variant_heatmap.png"))
        pd.DataFrame(mat, index=keep, columns=keep).to_csv(
            os.path.join(out_dir, "prompt_variant_heatmap_matrix.csv"))
        if excluded:
            print(f"[prompt-sa] heatmap excludes: {sorted(excluded)}")
    else:
        print(f"[prompt-sa] <2 prompts left after exclusion ({keep}) — skipping heatmap.")

    print(f"\n[done] outputs under {out_dir}/")


if __name__ == "__main__":
    main()
