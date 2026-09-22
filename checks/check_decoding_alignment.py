"""Do Gemma-4's decoding settings (1.0 / 0.975) behave like Qwen's (0.7 / 0.9)?

Reads the per-agent-seeded decoding grids of both generators (jobs/sa_seeded.job +
sa_seeded_analyze.job) and puts the two grids on one table, per setting:

  temp, top_p      the values actually run (posts.csv.meta.json)
  diversity        sa_analyze within-setting diversity (1 - cosine across the 3 reps,
                   S-BERT, decoding_settings_summary.csv) with its 95% CI
  bal_acc          adjacent-band separability, balanced accuracy of the S-BERT band
                   classifier (decoding_phq9_separability.csv)
  distinct-2       unique bigrams / total bigrams over the setting's 3 reps
  cross-agent J    mean word-set Jaccard between different agents' posts of a round
  chars            mean post length

"Aligned" means Gemma's operating point sits where Qwen's does on these axes; the rest
of Gemma's grid shows which of its settings would match Qwen's baseline better. Also
prints, per model, the Qwen-baseline diversity and the Gemma setting closest to it.
    PYTHONPATH=src .venv_vllm/bin/python checks/check_decoding_alignment.py
"""
import glob
import json
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd

ROOTS = {"Qwen3.5-27B": "data/sensitivity/seeded/qwen", "Gemma-4-31B-it": "data/sensitivity/seeded/gemma4"}
_WORD = re.compile(r"[a-z0-9']+")


def lexical(setting_dir):
    posts, dup_j, big = [], [], []
    for rep in sorted(glob.glob(os.path.join(setting_dir, "rep_*", "posts.csv"))):
        d = pd.read_csv(rep)
        d["tweet"] = d["tweet"].fillna("").astype(str)
        posts += d["tweet"].tolist()
        for _, g in d.groupby("step"):
            ws = [set(_WORD.findall(p.lower())) for p in g["tweet"]]
            dup_j += [len(a & b) / len(a | b) if (a | b) else 0.0 for a, b in combinations(ws, 2)]
    for p in posts:
        w = _WORD.findall(p.lower())
        big += list(zip(w[:-1], w[1:]))
    return {"distinct-2": len(set(big)) / max(len(big), 1), "cross-agent J": float(np.mean(dup_j)),
            "chars": float(np.mean([len(p) for p in posts]))}


def main():
    rows = []
    for model, root in ROOTS.items():
        summ = os.path.join(root, "plots_sbert", "decoding_settings_summary.csv")
        sep = os.path.join(root, "plots_sbert", "decoding_phq9_separability.csv")
        if not (os.path.isfile(summ) and os.path.isfile(sep)):
            print(f"[skip] {model}: run jobs/sa_seeded_analyze.job first ({summ})")
            continue
        div = pd.read_csv(summ).set_index("setting")
        acc = pd.read_csv(sep).set_index("setting")
        for sdir in sorted(glob.glob(os.path.join(root, "decoding", "setting_*"))):
            setting = os.path.basename(sdir)[len("setting_"):]
            meta = os.path.join(sdir, "rep_1", "posts.csv.meta.json")
            m = json.load(open(meta)) if os.path.isfile(meta) else {}
            row = {"model": model, "setting": setting, "temp": m.get("temp"), "top_p": m.get("top_p"),
                   "diversity": div.loc[setting, "mean_diversity"] if setting in div.index else np.nan,
                   "div_ci_lo": div.loc[setting, "ci_lo"] if setting in div.index else np.nan,
                   "div_ci_hi": div.loc[setting, "ci_hi"] if setting in div.index else np.nan,
                   "bal_acc": acc.loc[setting, "mean_bal_acc"] if setting in acc.index else np.nan}
            row.update(lexical(sdir))
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return
    out = out.sort_values(["model", "temp", "top_p"])
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    q = out[(out.model == "Qwen3.5-27B") & (out.setting == "baseline")]
    g = out[out.model == "Gemma-4-31B-it"]
    if len(q) and len(g):
        qd = float(q["diversity"].iloc[0])
        near = g.iloc[(g["diversity"] - qd).abs().argsort()[:2]]
        gb = g[g.setting == "baseline"].iloc[0]
        print(f"\nQwen baseline diversity {qd:.3f}; Gemma baseline {gb['diversity']:.3f} "
              f"[{gb['div_ci_lo']:.3f}, {gb['div_ci_hi']:.3f}]")
        print("Gemma settings closest to Qwen's baseline diversity:")
        print(near[["setting", "temp", "top_p", "diversity", "bal_acc", "distinct-2"]].round(3).to_string(index=False))
    os.makedirs("data/sensitivity/seeded", exist_ok=True)
    out.to_csv("data/sensitivity/seeded/decoding_alignment.csv", index=False)
    print("\n-> data/sensitivity/seeded/decoding_alignment.csv")


if __name__ == "__main__":
    main()
