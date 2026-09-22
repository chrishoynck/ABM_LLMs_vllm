"""Is increasing the PHQ-9 band significant?

One row per PHQ-9 class. `within` is that class's own noise floor (same band,
different replicate). `adjacent` is the paired same-persona cosine between that
class and the NEXT class up, and delta is adjacent minus that class's own within,
i.e. "starting from this class, does moving up one band change the text more than
rerunning this class does?". Severe has no class above it, so it carries a within
value only. The next class's own floor is visible on the following row.

Unit of analysis is the persona (60), not the anchor (600): the 10 rounds of one
persona are correlated, so an anchor-level test overstates significance. Rounds
and replicates are averaged within a persona, then a paired t-test over personas.

Usage:
    PYTHONPATH=src ./.venv_vllm/bin/python checks/phq9_band_significance.py \
        --roots qwen=data/sensitivity/seeded/qwen gemma4=data/sensitivity/seeded/gemma4 \
        --tex data/sensitivity/seeded/phq9_band_significance.tex
"""
import argparse, glob, os
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from utils.sensitivity.sa_analyze import phq9_to_band, BAND_LABELS

DISPLAY = {"qwen": "Qwen3.5-27B", "gemma4": "Gemma-4-31B-it"}


def load_bands(root, subdir="phq9", emb="embeddings_sbert.npz"):
    """{band: {rep: (anchor->row index, embeddings)}} plus the common anchors."""
    by = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(root, subdir, "*", "rep_*", emb))):
        d = np.load(p, allow_pickle=True)
        rep = int(os.path.basename(os.path.dirname(p)).split("_")[1])
        idx = {(int(a), int(r)): i for i, (a, r) in
               enumerate(zip(d["agent_ids"], d["rounds"]))}
        band = pd.Series([phq9_to_band(int(s)) for s in d["phq9"]]).mode().iloc[0]
        by[band][rep] = (idx, d["embeddings"])
    common = None
    for reps in by.values():
        for idx, _ in reps.values():
            common = set(idx) if common is None else common & set(idx)
    return by, sorted(common)


def unit_rows(idx, emb, common):
    X = np.array([emb[idx[a]] for a in common], dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def table(root):
    by, common = load_bands(root)
    bands = [b for b in BAND_LABELS if b in by]
    reps = sorted(by[bands[0]])
    persona = np.array([a for a, _ in common])

    within = {b: np.mean([(unit_rows(*by[b][ra], common) *
                           unit_rows(*by[b][rb], common)).sum(axis=1)
                          for ra, rb in combinations(reps, 2)], axis=0)
              for b in bands}

    rows = []
    for i, b0 in enumerate(bands):
        row = {"band": b0, "within": within[b0].mean(), "n": len(set(persona))}
        if i + 1 < len(bands):
            b1 = bands[i + 1]
            adjacent = np.mean([(unit_rows(*by[b0][r], common) *
                                 unit_rows(*by[b1][r], common)).sum(axis=1)
                                for r in reps], axis=0)
            g = (pd.DataFrame({"persona": persona, "adj": adjacent,
                               "base": within[b0]}).groupby("persona").mean())
            t, p = stats.ttest_rel(g.adj, g.base)
            d = g.adj - g.base
            row.update({"to_band": b1, "adjacent": g.adj.mean(), "delta": d.mean(),
                        "t": t, "p": p, "dz": d.mean() / d.std(ddof=1)})
        else:
            row.update({"to_band": "", "adjacent": np.nan, "delta": np.nan,
                        "t": np.nan, "p": np.nan, "dz": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_p(p):
    if p >= 0.01:
        return f"{p:.2f}"
    e = int(np.floor(np.log10(p)))
    return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="tag=path pairs")
    ap.add_argument("--tex", default="")
    args = ap.parse_args()

    blocks = {}
    for spec in args.roots:
        tag, path = spec.split("=", 1)
        blocks[DISPLAY.get(tag, tag)] = table(path)
        print(f"\n=== {DISPLAY.get(tag, tag)} ===")
        print(blocks[DISPLAY.get(tag, tag)].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))

    if not args.tex:
        return
    L = [r"\begin{table}[ht]", r"\centering",
         r"\caption{Is increasing the PHQ-9 band significant? One row per severity "
         r"class. \emph{Within} is that class's own noise floor: the same-persona "
         r"cosine between two replicates of the same band, i.e. how much the text "
         r"moves when nothing is changed. \emph{Adjacent} pairs the same persona "
         r"across that class and the next class up, and $\Delta$ is adjacent minus "
         r"within, so a negative $\Delta$ means the severity change moved the text "
         r"further than resampling alone. Severe has no class above it. Paired "
         r"$t$-test over $n=60$ personas (rounds and replicates averaged within "
         r"persona). S-BERT embeddings, per-agent-seeded runs.}",
         r"\label{tab:phq9_band_significance}",
         r"\begin{tabular}{llrrrrr}", r"\hline",
         r"Model & Class & Within & Adjacent & $\Delta$ & $t(59)$ & $p$ \\", r"\hline"]
    for name, df in blocks.items():
        for i, r in df.iterrows():
            model = name if i == 0 else ""
            if pd.isna(r["p"]):
                L.append(f"{model} & {r['band']} & {r['within']:.3f} & "
                         r"-- & -- & -- & -- \\")
            else:
                L.append(f"{model} & {r['band']} & {r['within']:.3f} & "
                         f"{r['adjacent']:.3f} & {r['delta']:+.3f} & "
                         f"{r['t']:.2f} & {fmt_p(r['p'])} \\\\")
        L.append(r"\hline")
    L += [r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(args.tex) or ".", exist_ok=True)
    with open(args.tex, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[tex] {args.tex}")


if __name__ == "__main__":
    main()
