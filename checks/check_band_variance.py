"""Within-band vs between-band variance of the 300-block test corpora (all four arms).

Quantifies how much of the textual variation in a generated corpus is severity and how
much is persona/topic, using the SAME encoder as the figure's panel (a)
(SBERT all-MiniLM-L6-v2, `metrics.generate_sbert_model`).

Unit of analysis is the BLOCK (one persona = the mean of its 10 post vectors,
re-normalised), so the number reported is variation ACROSS PERSONAS, not the run-to-run
stochasticity of regenerating one persona. Two baselines bracket it:

  within-persona   the 10 posts of one block, against each other. The floor: what the
                   model produces when persona AND PHQ-9 are both held fixed.
  overall          all block pairs regardless of band. The ceiling: total corpus diversity.

With L2-normalised vectors the variance and the mean pairwise cosine are the SAME statistic:
var = 1 - ||xbar||^2, and mean pairwise cosine over distinct pairs = (n(1-var) - 1)/(n-1),
which reproduces cos_overall to 6 decimals for every arm here. REPORT THE COSINE, not the
variance: it is the unit panel (a) is already in, and "variance 0.51" of unit vectors has no
intuitive scale. eta^2 = between/total from the variance columns is BIASED UPWARD: shuffling the band labels
still gives eta^2 = (k-1)/(n-1) = 0.0134 here (k=5 bands, n=300), confirmed by a 300-draw
permutation null. Report the bias-corrected form instead, which is exactly what the pairwise
cosines give: 1-(1-cos_within)/(1-cos_overall) reproduces (eta^2 - bias)/(1 - bias) to 4
decimals for every arm. It also makes samples of different n comparable (the Grok reference
has n=490, so its raw-eta^2 null floor is 0.0082, not 0.0134). Variance splits exactly:
total = within-band + between-band, and eta^2 = between / total is the share of corpus
variation attributable to the severity band.

Run on CPU (~2 min for 12,000 posts):
    PYTHONPATH=src .venv_vllm/bin/python checks/check_band_variance.py
"""
import argparse
import os

import numpy as np
import pandas as pd

from utils.metrics import generate_sbert_model
from utils.visualization import _MM_BANDS

ARMS = [  # label, posts CSV
    ("Qwen  human-opt", "data/finetune/qwen/test_posts_qwen.csv"),
    ("Qwen  minimal  ", "data/finetune/qwen_minimal/test_posts_qwen_minimal.csv"),
    ("Gemma human-opt", "data/finetune/gemma4/test_posts_gemma4.csv"),
    ("Gemma minimal  ", "data/finetune/gemma4_minimal/test_posts_gemma4_minimal.csv"),
]
# Loose external reference for the diversity numbers only: a different model family on a
# different prompt and different personas (only 51 of its 483 overlap the eval-1000 pool),
# so it is NOT paired with the four arms and its eta^2 is not comparable to theirs. It
# answers "is this much within-band spread normal for LLM social posts" and nothing else.
REFERENCE = [("Grok (unpaired ref)", "data/grok_posts/posts_with_phq9.csv")]
CACHE = "data/finetune/band_variance_emb"


def unit(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def mean_pairwise_cos(x):
    """Mean cosine over all distinct pairs of the rows of `x` (unit vectors)."""
    n = len(x)
    if n < 2:
        return np.nan
    g = x @ x.T
    return float((g.sum() - np.trace(g)) / (n * (n - 1)))


def embed_arm(model, label, path):
    """Per-post unit vectors for one corpus, cached next to the data."""
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, label.strip().replace(" ", "_") + ".npz")
    if os.path.isfile(key):
        d = np.load(key, allow_pickle=True)
        return d["emb"], d["agent_id"], d["phq9"]
    d = pd.read_csv(path)
    d = d[d["tweet"].astype(str).ne("NO_POST")].dropna(subset=["tweet"])
    emb = unit(np.asarray(model.encode(d["tweet"].astype(str).tolist(),
                                       batch_size=256, show_progress_bar=False), dtype=np.float64))
    np.savez_compressed(key, emb=emb, agent_id=d["agent_id"].to_numpy(), phq9=d["phq9"].to_numpy())
    return emb, d["agent_id"].to_numpy(), d["phq9"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/test_post/method_comparison/multimodel/band_variance.csv")
    args = ap.parse_args()

    model = generate_sbert_model()
    band_of = lambda v: next(i for i, (lo, hi, _) in enumerate(_MM_BANDS) if lo <= v <= hi)
    rows = []

    for label, path in ARMS + REFERENCE:
        if not os.path.isfile(path):
            print(f"[skip] {label}: {path} missing")
            continue
        emb, aid, phq9 = embed_arm(model, label, path)

        # --- baseline floor: the 10 posts of one block, persona AND PHQ-9 fixed ---
        within_persona = np.nanmean([mean_pairwise_cos(emb[aid == a]) for a in np.unique(aid)])

        # --- block level: one unit vector per persona ---
        blocks = pd.DataFrame({"aid": aid, "phq9": phq9})
        order = blocks.groupby("aid").first().index.to_numpy()
        B = unit(np.vstack([emb[aid == a].mean(0) for a in order]))
        bq = blocks.groupby("aid").phq9.first().reindex(order).to_numpy()
        bb = np.array([band_of(v) for v in bq])

        # variance decomposition (unit vectors: var = mean squared distance to centroid)
        gm = B.mean(0)
        total = float(((B - gm) ** 2).sum(1).mean())
        within = float(np.average([((B[bb == k] - B[bb == k].mean(0)) ** 2).sum(1).mean()
                                   for k in np.unique(bb)],
                                  weights=[np.sum(bb == k) for k in np.unique(bb)]))
        between = total - within

        # cosine view of the same thing
        cos_within = float(np.average([mean_pairwise_cos(B[bb == k]) for k in np.unique(bb)],
                                      weights=[np.sum(bb == k) for k in np.unique(bb)]))
        g = B @ B.T
        diff = bb[:, None] != bb[None, :]
        cos_between = float(g[diff].mean())
        cos_overall = mean_pairwise_cos(B)

        # --- post level, directly comparable to panel (a) ---------------------
        # Panel (a) is a post-level cosine (same persona, adjacent bands, paired by
        # round), so block-mean vectors cannot be read against it: averaging 10 posts
        # smooths them and inflates every cosine. These are the post-level companions.
        bpost = np.array([band_of(v) for v in phq9])
        G = emb @ emb.T
        np.fill_diagonal(G, np.nan)
        same_p = aid[:, None] == aid[None, :]
        same_b = bpost[:, None] == bpost[None, :]
        m = lambda mask: float(np.nanmean(G[mask]))
        post = dict(
            p_same_persona_same_band=m(same_p & same_b),        # floor: model stochasticity
            p_diff_persona_same_band=m(~same_p & same_b),       # within band, across personas
            p_diff_persona_diff_band=m(~same_p & ~same_b),      # across bands, across personas
            p_diff_persona_all=m(~same_p),
        )
        post["band_contrast"] = post["p_diff_persona_same_band"] - post["p_diff_persona_diff_band"]
        # Cross-persona diversity (1 - mean cosine) per band, for the table that mirrors the
        # figure's x-axis. Same-persona pairs are excluded so this is persona-to-persona
        # spread inside one severity band, not round-to-round variation.
        for k, (_, _, name) in enumerate(_MM_BANDS):
            sel = bpost == k
            if sel.sum() < 2:
                post[f"div_{name}"] = np.nan
                continue
            sub, sub_a = emb[sel], aid[sel]
            g = sub @ sub.T
            post[f"div_{name}"] = 1.0 - float(g[sub_a[:, None] != sub_a[None, :]].mean())
        del G

        rows.append(dict(**post, arm=label.strip(), n_blocks=len(B), total_var=total,
                         within_band_var=within, between_band_var=between,
                         eta2=between / total, cos_within_band=cos_within,
                         cos_between_band=cos_between, cos_overall=cos_overall,
                         cos_within_persona=float(within_persona)))
        print(f"[done] {label}  {len(B)} blocks")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    print("\n=== block-level variance across personas (SBERT MiniLM, unit vectors) ===")
    print(df[["arm", "n_blocks", "total_var", "within_band_var", "between_band_var", "eta2"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== same numbers as cosine, with both baselines ===")
    print(df[["arm", "cos_within_persona", "cos_within_band", "cos_between_band", "cos_overall"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== POST level: the scale panel (a)'s adjacent-band cosine should be read against ===")
    print("  same_persona_same_band = floor (model stochasticity, persona+PHQ-9 fixed)")
    print("  diff_persona_same_band = within a band, across personas  <- the reference scale")
    print("  band_contrast          = same_band - diff_band, how much severity structures text")
    print(df[["arm", "p_same_persona_same_band", "p_diff_persona_same_band",
              "p_diff_persona_diff_band", "band_contrast"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    band_names = [n for _, _, n in _MM_BANDS]
    df["diversity_cross_persona"] = 1 - df["p_diff_persona_all"]
    df["diversity_within_band"] = 1 - df["p_diff_persona_same_band"]
    print("\n=== cross-persona diversity (1 - mean cosine), post level ===")
    print(df[["arm", "diversity_cross_persona", "diversity_within_band"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== within-band cross-persona diversity, per band ===")
    print(df[["arm"] + [f"div_{n}" for n in band_names]]
          .rename(columns={f"div_{n}": n for n in band_names})
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
