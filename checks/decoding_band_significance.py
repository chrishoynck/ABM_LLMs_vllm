"""Is increasing the PHQ-9 band significant in the decoding SA?

Decoding runs use each persona's own PHQ-9, so every band holds different personas and
a persona can't be compared with itself one band up. `within` is therefore the cosine
between a persona and the OTHER personas of its own band, and `adjacent` the cosine
between that persona and the personas of the next band up. Posts are paired by round
inside the same run (rep), averaged over rounds, partners and reps. Delta is adjacent
minus within; a paired t-test over the personas of the lower band. One decoding setting
only (default: setting_baseline = Qwen (0.7, 0.9), Gemma (1.0, 0.975)).

Usage:
    PYTHONPATH=src ./.venv_vllm/bin/python checks/decoding_band_significance.py \
        --roots qwen=data/sensitivity/seeded/qwen gemma4=data/sensitivity/seeded/gemma4 \
        --tex data/sensitivity/seeded/decoding_band_significance.tex
"""
import argparse, glob, json, os

import numpy as np
import pandas as pd
from scipy import stats

from utils.sensitivity.sa_analyze import phq9_to_band, BAND_LABELS

DISPLAY = {"qwen": "Qwen3.5-27B", "gemma4": "Gemma-4-31B-it"}


def load_setting(root, setting):
    """Per-persona cosine matrix (same rep, same round, averaged) and each persona's band."""
    mats = []
    for p in sorted(glob.glob(os.path.join(root, "decoding", f"setting_{setting}", "rep_*",
                                           "embeddings_sbert.npz"))):
        d = np.load(p, allow_pickle=True)
        X = d["embeddings"] / np.linalg.norm(d["embeddings"], axis=1, keepdims=True)
        a, r = d["agent_ids"].astype(int), d["rounds"].astype(int)
        ids, rounds = np.unique(a), np.unique(r)
        T = np.zeros((len(ids), len(rounds), X.shape[1]))          # persona x round x dim
        T[np.searchsorted(ids, a), np.searchsorted(rounds, r)] = X
        mats.append(np.einsum("itd,jtd->ij", T, T) / len(rounds))
        band = np.array([phq9_to_band(int(d["phq9"][a == i][0])) for i in ids])
    meta = os.path.join(os.path.dirname(p), "posts.csv.meta.json")
    params = json.load(open(meta)) if os.path.exists(meta) else {}
    return np.mean(mats, axis=0), band, (params.get("temp"), params.get("top_p"))


def table(root, setting):
    S, band, params = load_setting(root, setting)
    rows = []
    for i, b0 in enumerate(BAND_LABELS):
        I = np.where(band == b0)[0]
        within = np.array([S[k, I[I != k]].mean() for k in I])
        row = {"band": b0, "n": len(I), "within": within.mean()}
        if i + 1 < len(BAND_LABELS):
            J = np.where(band == BAND_LABELS[i + 1])[0]
            adjacent = np.array([S[k, J].mean() for k in I])
            t, p = stats.ttest_rel(adjacent, within)
            row.update({"adjacent": adjacent.mean(), "delta": (adjacent - within).mean(),
                        "t": t, "p": p})
        rows.append(row)
    return pd.DataFrame(rows), params


def fmt_p(p):
    if p >= 0.01:
        return f"{p:.2f}"
    e = int(np.floor(np.log10(p)))
    return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="tag=path pairs")
    ap.add_argument("--setting", default="baseline")
    ap.add_argument("--tex", default="")
    args = ap.parse_args()

    blocks = {}
    for spec in args.roots:
        tag, path = spec.split("=", 1)
        df, params = table(path, args.setting)
        name = f"{DISPLAY.get(tag, tag)} {params}"
        blocks[name] = df
        print(f"\n=== {name} ===")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not args.tex:
        return
    L = [r"\begin{table}[ht]", r"\centering",
         r"\caption{Is increasing the PHQ-9 band significant in the decoding runs? The "
         r"decoding runs keep each persona's own PHQ-9 score, so every class holds "
         r"different personas. \emph{Within} is the cosine between a persona and the "
         r"other personas of its own class; \emph{Adjacent} is the cosine between that "
         r"persona and the personas of the next class up. Posts are paired by round "
         r"within a run. A negative $\Delta$ means the next class is further away than "
         r"another persona of the same class, a positive $\Delta$ that it is closer. "
         r"Paired $t$-test over the $n$ personas of "
         r"the class. S-BERT embeddings, per-agent-seeded runs, one decoding setting "
         r"(temperature, top-$p$) per model.}",
         r"\label{tab:decoding_band_significance}",
         r"\begin{tabular}{llrrrrr}", r"\hline",
         r"Model & Class & $n$ & Within & Adjacent & $\Delta$ & $p$ \\", r"\hline"]
    for name, df in blocks.items():
        for i, r in df.iterrows():
            model = name if i == 0 else ""
            if pd.isna(r.get("p", np.nan)):
                L.append(f"{model} & {r['band']} & {r['n']} & {r['within']:.3f} & "
                         r"-- & -- & -- \\")
            else:
                L.append(f"{model} & {r['band']} & {r['n']} & {r['within']:.3f} & "
                         f"{r['adjacent']:.3f} & {r['delta']:+.3f} & {fmt_p(r['p'])} \\\\")
        L.append(r"\hline")
    L += [r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(args.tex) or ".", exist_ok=True)
    with open(args.tex, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[tex] {args.tex}")


if __name__ == "__main__":
    main()
