"""Block-level PHQ-9 sensitivity analysis via the MentalBERT + MLP regressor.

Rationale
---------
Raw MentalBERT cosine is anisotropic (frequent-token dominated), so it is a poor
instrument for *content* sensitivity — that axis is better served by SBERT
(``sa_embed --sbert`` -> ``embeddings_sbert.npz`` -> ``sa_analyze --emb-name``).
For the PHQ-9 axis we instead push the embeddings through the supervised
regressor, whose MLP learns its own metric and is unaffected by the anisotropy
critique. The signal here is the **raw (non-rounded) predicted PHQ-9 score**.

What it does
------------
Reuses the per-post mean-pooled MentalBERT vectors already on disk
(``embeddings.npz`` written by ``sa_embed``): groups each run's posts by agent
(= one PHQ-9 "block"), builds the ``(mean ∥ max ∥ std)`` centroid the regressor
was trained on, and runs the best fine-tuned regressor (seed 35, test MAE 2.76)
to get one raw predicted PHQ-9 per block. No re-encoding — the centroid is built
straight from ``embeddings.npz``, filtering NO_POST/NO_TWEET/empty posts via the
stored ``texts`` exactly as ``eval_bert_on_csv``.

It then runs the same within- vs cross-setting comparison as ``sa_analyze`` but
on ``|Δ predicted PHQ-9|`` (points) instead of cosine:

    within-setting |Δ| = irreducible LLM stochasticity (3 reps, same seeds)
    cross-setting  |Δ| = effect of varying the axis, with LLM noise mixed in

Interpretation flips relative to cosine: here within ≈ small, and if cross > within
the axis moves the *predicted depression severity* more than LLM noise alone.

Outputs (default ``data/sensitivity/plots_phq9/``):
    <axis>_phq9_pred.csv       - one row per (setting, rep, agent): raw_pred, true_phq9, band
    <axis>_phq9_delta.csv      - one row per pair: within|cross, band, |Δ pred|
    <axis>_phq9_summary.csv    - mean ± std |Δ| per (band, within|cross)
    <axis>_phq9_bar_scatter.png
Plus a ``phq9_pred.csv`` next to each run's ``embeddings.npz``.

Usage::

    PYTHONPATH=src python -m utils.sensitivity.sa_phq9
    PYTHONPATH=src python -m utils.sensitivity.sa_phq9 --axes neighbor agent joint
    PYTHONPATH=src python -m utils.sensitivity.sa_phq9 --regressor <path/regressor.pt>
    PYTHONPATH=src python -m utils.sensitivity.sa_phq9 --agent-pairing slot  # match sa_analyze
"""

from __future__ import annotations

import argparse
import glob
import os
from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# neural_net_BERT must be importable so torch.load can unpickle the saved module.
try:
    from utils.prompt_optimizer import neural_net_BERT  # noqa: F401
except ImportError:
    from ..prompt_optimizer import neural_net_BERT  # noqa: F401

# The regressor was pickled while prompt_optimizer ran as __main__, so the class
# is referenced as __main__.neural_net_BERT. Register it there so torch.load can
# unpickle no matter how this module is invoked (-m, imported, or -c).
import __main__ as _main  # noqa: E402
if not hasattr(_main, "neural_net_BERT"):
    _main.neural_net_BERT = neural_net_BERT

from utils.sensitivity.sa_analyze import (
    BAND_LABELS,
    _parse_setting,
    comparison_combined,
    phq9_to_band,
)

# Best fine-tuned regressor (lowest test MAE = 2.76); also the default wired in
# llama_activate.py. Trained on MentalBERT (768-dim) -> 768*3 centroid input.
DEFAULT_REGRESSOR = "data/test_post/bert_regression_finetuned/Qwen3.5-27B_seed35/regressor.pt"
INVALID_POSTS = {"NO_POST", "NO_TWEET"}
MENTALBERT_DIM = 768


# =====================================================================
# Block-level prediction (reuses embeddings.npz; no re-encoding)
# =====================================================================

def load_regressor(path: str, device: torch.device):
    """Load the pickled neural_net_BERT module saved by train_BERT_model."""
    print(f"[phq9] loading regressor from {path}")
    reg = torch.load(path, map_location=device, weights_only=False)
    return reg.to(device).eval()


def _block_centroid(block_embs: np.ndarray, device: torch.device) -> torch.Tensor:
    """(n_posts, dim) post embeddings -> (3*dim,) mean ∥ max ∥ std centroid.

    Mirrors prompt_optimizer's training centroid and eval_bert_on_csv exactly,
    including torch's *unbiased* variance and the sqrt(var + 1e-8) std with the
    NaN guard (a 1-post block has unbiased var = NaN -> 0 -> std = 0).
    """
    emb = torch.as_tensor(block_embs, dtype=torch.float32, device=device)
    mean_v = emb.mean(dim=0)
    max_v = emb.max(dim=0)[0]
    var_emb = emb.var(dim=0)
    if torch.isnan(var_emb).any():
        var_emb = torch.zeros_like(var_emb)
    std_v = torch.sqrt(var_emb + 1e-8)
    return torch.cat([mean_v, max_v, std_v], dim=0)


def block_preds_from_npz(regressor, npz_path: str, device: torch.device) -> pd.DataFrame:
    """One raw predicted PHQ-9 per agent (block) for a single run's embeddings.npz."""
    data = np.load(npz_path, allow_pickle=True)
    embs = data["embeddings"]
    aids = data["agent_ids"]
    phq9 = data["phq9"]
    texts = data["texts"]

    if embs.shape[1] != MENTALBERT_DIM:
        raise ValueError(
            f"{npz_path} has dim {embs.shape[1]}, but the PHQ-9 regressor expects "
            f"MentalBERT ({MENTALBERT_DIM}). Use embeddings.npz, not embeddings_sbert.npz.")

    # Same validity filter as eval_bert_on_csv: drop empty / NO_POST / NO_TWEET.
    valid = np.array([bool(t) and str(t).upper() not in INVALID_POSTS for t in texts])

    centroids, meta = [], []
    for aid in np.unique(aids):
        mask = (aids == aid) & valid
        if not mask.any():
            continue
        centroids.append(_block_centroid(embs[mask], device))
        true = int(phq9[aids == aid][0])
        meta.append((int(aid), true, int(mask.sum())))

    if not centroids:
        return pd.DataFrame(columns=["agent_id", "true_phq9", "raw_pred", "band", "n_posts"])

    batch = torch.stack(centroids).to(device)
    with torch.no_grad():
        raw = regressor(batch).squeeze(-1).cpu().numpy().astype(float)

    return pd.DataFrame([
        {"agent_id": aid, "true_phq9": true, "raw_pred": float(raw[i]),
         "band": phq9_to_band(true), "n_posts": n}
        for i, (aid, true, n) in enumerate(meta)
    ])


def predict_axis(regressor, root: str, axis: str, device: torch.device,
                 emb_name: str = "embeddings.npz",
                 write_per_run: bool = True) -> dict:
    """Return {(setting, rep): per-block DataFrame} for one axis; writes phq9_pred.csv per run."""
    paths = sorted(glob.glob(os.path.join(root, axis, "setting_*", "rep_*", emb_name)))
    preds: dict = {}
    for p in paths:
        parts = p.split(os.sep)
        setting = _parse_setting(next(x for x in parts if x.startswith("setting_")))
        rep = int(next(x for x in parts if x.startswith("rep_")).split("_")[1])
        df = block_preds_from_npz(regressor, p, device)
        preds[(setting, rep)] = df
        if write_per_run and len(df):
            df.to_csv(os.path.join(os.path.dirname(p), "phq9_pred.csv"), index=False)
    return preds


def predict_phq9(regressor, root: str, device: torch.device,
                 emb_name: str = "embeddings.npz",
                 write_per_run: bool = True) -> dict:
    """Return {(band, rep): per-block DataFrame} for the PHQ-9 conditioning runs
    under ``root/phq9/<band>/rep_*/<emb_name>``. Band plays the role of setting,
    so phq9_within_cross (paired=True) gives within = same band / different rep
    (LLM-noise floor) and cross = same persona re-conditioned on a different band.
    Mirrors ``predict_axis`` but for the band/rep layout (with the legacy
    single-dir fallback)."""
    paths = sorted(glob.glob(os.path.join(root, "phq9", "*", "rep_*", emb_name)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "phq9", "*", emb_name)))
    preds: dict = {}
    for p in paths:
        parts = p.split(os.sep)
        band, rep = (parts[-3], int(parts[-2].split("_")[1])) \
            if parts[-2].startswith("rep_") else (parts[-2], 1)
        df = block_preds_from_npz(regressor, p, device)
        preds[(band, rep)] = df
        if write_per_run and len(df):
            df.to_csv(os.path.join(os.path.dirname(p), "phq9_pred.csv"), index=False)
    return preds


# =====================================================================
# Within- vs cross-setting on |Δ predicted PHQ-9|
# =====================================================================

def _assert_slot_phq9_consistent(preds: dict, axis: str) -> None:
    """Fail loudly if slot (agent_id) -> true_phq9 differs across settings.

    Per-slot pairing (``--agent-pairing slot``) on the agent axis is only valid
    when generation used ``--stratify-phq9`` so that slot i carries the same
    PHQ-9 in every setting (only the persona text differs). Mirrors the same
    guard in ``sa_analyze.agent_cosines`` so the two analyses stay comparable.
    """
    ref = None
    ref_key = None
    for key, df in preds.items():
        mapping = {int(a): int(p) for a, p in zip(df.agent_id, df.true_phq9)}
        if ref is None:
            ref, ref_key = mapping, key
            continue
        if mapping != ref:
            diffs = [k for k in mapping if mapping.get(k) != ref.get(k)]
            raise SystemExit(
                f"[{axis}] slot-level PHQ-9 differs between {ref_key} and {key} "
                f"(e.g. {diffs[:5]}). '--agent-pairing slot' requires --stratify-phq9 "
                "generation; use '--agent-pairing band' for non-stratified data.")


def phq9_within_cross(preds: dict, paired: bool) -> pd.DataFrame:
    """Pairwise |Δ raw_pred|, labelled within/cross and stratified by PHQ-9 band.

    paired=True  (neighbour / joint / decoding, or the agent axis under
        ``--agent-pairing slot``): the agent_id slot is shared across settings,
        so cross pairs the SAME slot across settings. On the agent axis this is
        only valid when generation used --stratify-phq9 (slot i keeps its PHQ-9,
        only the persona text differs); it matches sa_analyze.agent_cosines.
    paired=False (agent axis under the default ``--agent-pairing band``): the
        personas differ per setting, so cross pairs DIFFERENT agents in the same
        PHQ-9 band (mirrors the legacy agent_cosines_centroid).
    The within branch is identical either way: same agent, same setting, diff reps.
    """
    settings = sorted({s for (s, _) in preds})
    reps = sorted({r for (_, r) in preds})

    # P[(s, r)] = {agent_id: (raw_pred, band)}
    P: dict = {}
    for (s, r), df in preds.items():
        P[(s, r)] = {int(a): (float(v), b) for a, v, b in
                     zip(df.agent_id, df.raw_pred, df.band)}

    rows = []

    # WITHIN-SETTING: same agent across reps within one setting.
    for s in settings:
        agents = set().union(*[set(P[(s, r)]) for r in reps if (s, r) in P]) \
            if any((s, r) in P for r in reps) else set()
        for aid in agents:
            for ra, rb in combinations(reps, 2):
                if (s, ra) in P and (s, rb) in P \
                        and aid in P[(s, ra)] and aid in P[(s, rb)]:
                    va, band = P[(s, ra)][aid]
                    vb, _ = P[(s, rb)][aid]
                    rows.append({"pair_type": "within", "band": band,
                                 "setting_a": s, "setting_b": s,
                                 "rep_a": ra, "rep_b": rb,
                                 "agent_a": aid, "agent_b": aid,
                                 "delta": abs(va - vb)})

    # CROSS-SETTING.
    for sa, sb in combinations(settings, 2):
        for ra, rb in product(reps, reps):
            if (sa, ra) not in P or (sb, rb) not in P:
                continue
            if paired:
                # Same agent across the two settings.
                for aid in set(P[(sa, ra)]) & set(P[(sb, rb)]):
                    va, band = P[(sa, ra)][aid]
                    vb, _ = P[(sb, rb)][aid]
                    rows.append({"pair_type": "cross", "band": band,
                                 "setting_a": sa, "setting_b": sb,
                                 "rep_a": ra, "rep_b": rb,
                                 "agent_a": aid, "agent_b": aid,
                                 "delta": abs(va - vb)})
            else:
                # Band-matched different agents.
                band_a, band_b = {}, {}
                for aid, (v, band) in P[(sa, ra)].items():
                    band_a.setdefault(band, []).append((aid, v))
                for aid, (v, band) in P[(sb, rb)].items():
                    band_b.setdefault(band, []).append((aid, v))
                for band in band_a.keys() & band_b.keys():
                    for (aid_a, va), (aid_b, vb) in product(band_a[band], band_b[band]):
                        rows.append({"pair_type": "cross", "band": band,
                                     "setting_a": sa, "setting_b": sb,
                                     "rep_a": ra, "rep_b": rb,
                                     "agent_a": aid_a, "agent_b": aid_b,
                                     "delta": abs(va - vb)})

    return pd.DataFrame(rows)


def plot_phq9_bar_scatter(df: pd.DataFrame, axis_name: str, out_path: str,
                          max_scatter_per_band: int = 200) -> None:
    """Bars (mean |Δ pred|) + jittered scatter, two bars per PHQ-9 band."""
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(BAND_LABELS))
    bar_w = 0.36
    rng = np.random.default_rng(0)

    for ptype, colour, offset in [("within", "#cccccc", -bar_w / 2),
                                  ("cross",  "#e67e22", +bar_w / 2)]:
        means, errs = [], []
        for band in BAND_LABELS:
            sub = df[(df.band == band) & (df.pair_type == ptype)]
            means.append(sub.delta.mean() if len(sub) else np.nan)
            errs.append(sub.delta.std() if len(sub) > 1 else 0.0)
        ax.bar(x + offset, means, bar_w, yerr=errs, label=f"{ptype}-setting",
               color=colour, edgecolor="black", linewidth=0.5,
               error_kw={"elinewidth": 0.7, "capsize": 3})
        for j, band in enumerate(BAND_LABELS):
            sub = df[(df.band == band) & (df.pair_type == ptype)]
            if len(sub) == 0:
                continue
            if len(sub) > max_scatter_per_band:
                sub = sub.sample(max_scatter_per_band, random_state=int(j))
            jit = rng.uniform(-bar_w / 4, bar_w / 4, len(sub))
            ax.scatter(np.full(len(sub), x[j] + offset) + jit, sub.delta,
                       s=6, alpha=0.25, color="black", linewidths=0)

    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS, rotation=10)
    ax.set_ylabel("|Δ predicted PHQ-9|  (points)")
    ax.set_xlabel("PHQ-9 band")
    ax.set_title(f"{axis_name} sensitivity — within vs cross-setting predicted PHQ-9")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/sensitivity",
                        help="Directory containing axis subdirs.")
    parser.add_argument("--out-dir", default="data/sensitivity/plots_phq9",
                        help="Where to write combined CSVs + PNGs.")
    parser.add_argument("--regressor", default=DEFAULT_REGRESSOR,
                        help="regressor.pt to use (default: best fine-tuned seed 35).")
    parser.add_argument("--axes", nargs="+",
                        default=["neighbor", "agent", "joint", "phq9"],
                        help="Axes to analyse. Non-agent axes are always slot-paired; "
                             "the agent axis pairing is set by --agent-pairing. "
                             "'phq9' = the 5 severity-band conditioning runs.")
    parser.add_argument("--agent-pairing", choices=["band", "slot"], default="band",
                        help="Agent-axis cross pairing. 'band' (default): band-matched "
                             "different personas (legacy, mirrors agent_cosines_centroid). "
                             "'slot': per-slot paired across settings, matching "
                             "sa_analyze.agent_cosines (requires --stratify-phq9 generation).")
    parser.add_argument("--emb-name", default="embeddings.npz",
                        help="Encoder .npz to read (must be MentalBERT 768-dim).")
    parser.add_argument("--device", default=None, help="torch device (default: auto).")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)
    regressor = load_regressor(args.regressor, device)

    # Display names + per-axis slot-paired delta tables for the cross-axis figure.
    _DISPLAY = {"neighbor": "Neighbour", "agent": "Agent", "joint": "Joint",
                "phq9": "PHQ-9"}
    axis_deltas: dict = {}

    for axis in args.axes:
        full = os.path.join(args.root, axis)
        if not os.path.isdir(full):
            print(f"[skip] {full} not present")
            continue
        print(f"\n=== {axis} axis ===")
        if axis == "phq9":
            preds = predict_phq9(regressor, args.root, device, emb_name=args.emb_name)
        else:
            preds = predict_axis(regressor, args.root, axis, device, emb_name=args.emb_name)
        preds = {k: v for k, v in preds.items() if len(v)}
        if not preds:
            print(f"  no predictions for {axis}")
            continue
        print(f"  predicted {len(preds)} runs")

        # Combined per-(setting, rep, agent) predictions.
        combined = pd.concat(
            [df.assign(setting=s, rep=r) for (s, r), df in preds.items()],
            ignore_index=True)
        combined.to_csv(os.path.join(args.out_dir, f"{axis}_phq9_pred.csv"), index=False)

        # Within vs cross on |Δ predicted PHQ-9|. Non-agent axes share the
        # agent_id slot across settings, so they are always slot-paired. The
        # agent axis defaults to band-matching (personas differ per setting);
        # --agent-pairing slot opts into the per-slot comparison sa_analyze uses
        # (valid only for --stratify-phq9 data, so guard it).
        if axis == "agent":
            paired = args.agent_pairing == "slot"
            if paired:
                _assert_slot_phq9_consistent(preds, axis)
        else:
            paired = True
        print(f"  pairing: {'slot-paired' if paired else 'band-matched'}")
        delta = phq9_within_cross(preds, paired=paired)
        delta.to_csv(os.path.join(args.out_dir, f"{axis}_phq9_delta.csv"), index=False)

        summary = (delta.groupby(["band", "pair_type"]).delta
                   .agg(["mean", "std", "count"]).round(4))
        print("  |Δ predicted PHQ-9| per (band, pair_type):")
        print(summary)
        summary.to_csv(os.path.join(args.out_dir, f"{axis}_phq9_summary.csv"))

        pivot = (delta.groupby(["band", "pair_type"]).delta.mean().unstack("pair_type"))
        if {"within", "cross"} <= set(pivot.columns):
            pivot["delta(cross-within)"] = pivot["cross"] - pivot["within"]
            print("  cross − within per band (points):")
            print(pivot.round(4))

        plot_phq9_bar_scatter(delta, axis.capitalize(),
                              os.path.join(args.out_dir, f"{axis}_phq9_bar_scatter.png"))

        # For the cross-axis comparison figure every axis must be slot-paired so
        # the per-agent anchor (agent_a) is well defined — band-matched cross
        # pairs join DIFFERENT agents and share no anchor. Reuse `delta` when it
        # is already slot-paired, otherwise recompute it slot-paired.
        if paired:
            axis_deltas[_DISPLAY.get(axis, axis)] = delta
        else:
            _assert_slot_phq9_consistent(preds, axis)
            axis_deltas[_DISPLAY.get(axis, axis)] = phq9_within_cross(preds, paired=True)

    # Cross-axis comparison: the same box(distribution)+forest(mean ± CI) figure
    # as sa_analyze's axes_comparison, but on |Δ predicted PHQ-9| (cross − within)
    # instead of cosine drop. drop_sign=-1 flips within−cross → cross−within so a
    # tall box still means "factor moves output beyond LLM noise".
    if len(axis_deltas) >= 2:
        print("\n=== Cross-axis comparison (MentalBERT + MLP) ===")
        comparison_combined(
            axis_deltas, os.path.join(args.out_dir, "axes_comparison.png"),
            value_col="delta", ylabel="|Δ predicted PHQ-9|",
            anchor_cols=("agent_a",), drop_sign=-1.0)

    print(f"\n[done] PHQ-9 SA outputs under {args.out_dir}/")


if __name__ == "__main__":
    main()
