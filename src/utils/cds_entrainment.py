"""Local entrainment of Cognitive Distortion Schema (CDS) over directed topologies.

Question: does network topology (clustering / degree) drive the *local spread* of
cognitive distortions?  For each agent we ask whether the cosine similarity of its
CDS TF-IDF vector to its in-neighbours is higher than to a random (global) baseline
(local entrainment, following the sent formula):

    s_local(i,w)  = cos( v_i^w , mean_{j in N^-(i)}        v_j^w )
    s_random(i,w) = cos( v_i^w , mean_{j in random, |.|=|N^-(i)|} v_j^w )   (avg over R draws)
    Delta = s_local - s_random          (entrainment gap; >0 => neighbours more alike)

The random baseline is SIZE-MATCHED: for each focal it averages over the same number
of randomly chosen active agents as the agent has in-neighbours, repeated R times.
This removes a group-size confound in the originally-sent global baseline (averaging
over all ~N-1 agents collapses toward the shared centroid and inflates cosine, which
produced a spurious negative Delta).

Design choices (see plan):
  * v_i^w is built over a sliding window of W rounds (default 10, step 1): each
    agent's *real* tweets in [w, w+W) are pooled into one K=241 CDS-marker TF-IDF
    vector.  Single-tweet vectors are too sparse; whole-run pooling measures static
    homophily rather than time-aligned entrainment.
  * K = 241 CDS markers; spelling variants collapse onto their base-marker dimension.
  * IDF is fit once over *all* individual real tweets pooled across every config and
    seed, so all vectors live in one comparable weighted space.
  * Neighbour vectors are L2-normalised before averaging (each neighbour contributes
    equally; cosine already normalises the focal vector).  Toggle with ``normalize``.

Edge direction (verified empirically, not assumed):  in this ABM a stored edge
``[i, j]`` means agent ``i`` is EXPOSED TO agent ``j``'s content, so the in-neighbours
whose content ``i`` sees (N^-(i)) are ``i``'s SUCCESSORS.  This was confirmed against
``neighbor_history`` (the agents whose tweets ``i`` actually saw each round) — see
``verify_edge_direction`` and the runner's printed check.

Standalone: reads ``net.json`` directly, so it does not pull the heavy
``metrics``/``reading_in`` import chain (torch / umap / sentence_transformers).

Run:  PYTHONPATH=src python src/utils/cds_entrainment.py
"""

import os
import re
import csv
import json
import glob
import argparse

import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
# src/utils/cds_entrainment.py -> repo root is two dirs up.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NGRAMS_TSV = os.path.join(REPO, "data", "distorted_language_ngrams.tsv")
NETWORKS_DIR = os.path.join(REPO, "data", "networks_post")
OUT_DIR = os.path.join(REPO, "data", "analysis", "cds_entrainment")

# Silent-round sentinels (FC.NO_CONTENT is "NO_POST"/"NO_TWEET").
NO_CONTENT = {"no_post", "no_tweet"}
ROUNDS_DIR = "rounds300_N100"

# The 6 directed configs on disk (debiased). (label, path under networks_post).
# Labels follow the canonical naming table; the legend strips any " (...)" suffix
# (see label.split(" (")[0] in build_timeseries), so "SDA low degree (low C)"
# shows as "SDA low degree".
CONFIGS = [
    ("SDA calibrated",         "basis/sda/directed/debiased/2_1655_d4_5_dim5"),
    ("SDA high degree",        "basis/sda/directed/debiased/2_1655_d6_dim5"),
    ("SDA low degree (low C)", "basis/sda/directed/debiased/1_1655_d3_dim5"),
    ("SDC calibrated",         "basis/sdc/directed/debiased/4_9429_d8_2539_dim3"),
    ("SDC high degree",        "basis/sdc/directed/debiased/4_9429_d10_dim3"),
    (r"SDC high PHQ-9$_\rho$", "basis/sdc/directed/debiased/8_0_d8_2539_dim2"),
]

# Qualitative palette reused from sa_analyze.py (_COLOUR_BY_NAME + SA-axis / PHQ-9
# band hexes) so these figures match the rest of the thesis.
SA_PALETTE = ["#2e7ebc", "#d96907", "#2e8b57", "#8d2c03", "#6a3d9a", "#e74c3c"]


# ── CDS marker vocabulary ────────────────────────────────────────────────────
def load_markers(path=NGRAMS_TSV):
    """Return ordered list of (category, base_marker, [surface_forms]) — K markers.

    Mirrors ``metrics.load_ngrams_tsv`` parsing but keeps the marker grouping so
    base + variants map to ONE dimension.
    """
    markers = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            category = row[0].strip()
            base = row[1].strip().lower()
            if not base:
                continue
            forms = {base}
            if len(row) > 2 and row[2].strip():
                try:
                    variants = json.loads(row[2])
                    if isinstance(variants, list):
                        forms.update(v.strip().lower() for v in variants if v.strip())
                except json.JSONDecodeError:
                    forms.add(row[2].strip().lower())
            markers.append((category, base, sorted(forms)))
    return markers


def compile_patterns(markers):
    """One word-boundary regex per marker (alternation of its surface forms), plus a
    union pre-filter regex to cheaply skip tweets with no markers at all."""
    pats = [
        re.compile(r"\b(?:" + "|".join(re.escape(f) for f in forms) + r")\b")
        for _, _, forms in markers
    ]
    all_forms = sorted({f for _, _, forms in markers for f in forms}, key=len, reverse=True)
    any_pat = re.compile(r"\b(?:" + "|".join(re.escape(f) for f in all_forms) + r")\b")
    return pats, any_pat


def tweet_tf(text, pats, any_pat, K):
    """Raw CDS-marker term-frequency vector (length K) for one tweet."""
    low = text.lower()
    vec = np.zeros(K, dtype=np.float32)
    if not any_pat.search(low):  # fast path: most tweets have no markers
        return vec
    for k, p in enumerate(pats):
        c = len(p.findall(low))
        if c:
            vec[k] = c
    return vec


# ── net.json loading (direct, no heavy imports) ──────────────────────────────
def load_net(path):
    """Return (history, phq9, A) for one seed.

    history : list[N] of list[T] tweet strings
    phq9    : (N, T) float array of per-round PHQ-9 sumscores
    A       : (N, N) float32; A[i, j] = 1 iff i is exposed to j's content
              (stored edge i->j; focal i, content-source neighbour j).
    """
    with open(path) as f:
        d = json.load(f)
    N = d["Number of Agents"]
    by_id = {a["id"]: a for a in d["Agents"]}
    history = [by_id[i]["history"] for i in range(N)]
    phq9 = np.array([by_id[i]["phq9"] for i in range(N)], dtype=float)
    A = np.zeros((N, N), dtype=np.float32)
    for a, b in d["Connections"]:
        A[a, b] = 1.0
    return history, phq9, A


def net_tf_sparse(history, pats, any_pat, K):
    """Scan every real tweet once. Return (idx, val, real) where idx/val are the
    sparse non-zero entries of the (N, T, K) per-round TF tensor and ``real`` is the
    (N, T) bool mask of rounds with an actual post."""
    N = len(history)
    T = len(history[0])
    real = np.zeros((N, T), dtype=bool)
    idx_rows = []  # (a, t, k)
    vals = []
    for a, hist in enumerate(history):
        for t, tw in enumerate(hist):
            if not isinstance(tw, str):
                continue
            s = tw.strip()
            if not s or s.lower() in NO_CONTENT:
                continue
            real[a, t] = True
            v = tweet_tf(s, pats, any_pat, K)
            nz = np.nonzero(v)[0]
            for k in nz:
                idx_rows.append((a, t, k))
                vals.append(v[k])
    idx = np.array(idx_rows, dtype=np.int32).reshape(-1, 3)
    val = np.array(vals, dtype=np.float32)
    return idx, val, real


def dense_tf(idx, val, N, T, K):
    tf = np.zeros((N, T, K), dtype=np.float32)
    if len(val):
        tf[idx[:, 0], idx[:, 1], idx[:, 2]] = val
    return tf


# ── entrainment metric ───────────────────────────────────────────────────────
def entrainment_for_net(tf, real, A, idf, W=10, step=1, normalize=True, R=10, seed=0):
    """Mean local / size-matched-random cosine similarity over valid (agent, window).

    For each window the focal window vector v_i^w is compared to (a) the mean of its
    non-zero in-neighbours and (b) the mean of an equally-sized random set of other
    non-zero agents, averaged over R draws (size-matched baseline).

    A pair is *valid* when the focal vector is non-zero, has >=1 non-zero in-neighbour,
    and >=1 other non-zero agent exists (so both terms are defined; kept paired so
    Delta is a clean per-(i,w) difference).
    coverage = #valid pairs / #agent-windows in which the agent actually posted.
    """
    N, T, K = tf.shape
    tfidf = tf * idf  # broadcast over markers
    rng = np.random.default_rng(seed)

    # prefix sums for O(1) window pooling
    cs = np.concatenate([np.zeros((N, 1, K), dtype=tfidf.dtype),
                         np.cumsum(tfidf, axis=1)], axis=1)         # (N, T+1, K)
    rcs = np.concatenate([np.zeros((N, 1)), np.cumsum(real, axis=1)], axis=1)  # (N, T+1)

    s_local, s_random = [], []
    n_active = 0
    for w in range(0, T - W + 1, step):
        M = cs[:, w + W, :] - cs[:, w, :]               # (N, K) window TF-IDF
        n_active += int(((rcs[:, w + W] - rcs[:, w]) > 0).sum())  # posted >=1 tweet

        norms = np.linalg.norm(M, axis=1)
        nz = norms > 0
        n_nz = int(nz.sum())
        if n_nz < 3:                                      # focal + neighbour + 1 other
            continue

        Mn = np.zeros_like(M)
        Mn[nz] = M[nz] / norms[nz, None]                  # unit focal directions
        base = Mn if normalize else M
        base_masked = base * nz[:, None]                  # zero out empty agents

        # local mean over non-zero in-neighbours (successors), vectorised over agents
        cnt_local = A @ nz.astype(np.float32)             # (N,)
        sum_local = A @ base_masked                       # (N, K)
        with np.errstate(divide="ignore", invalid="ignore"):
            local_mean = sum_local / cnt_local[:, None]
            ln = np.linalg.norm(local_mean, axis=1)
            sloc_all = (Mn * local_mean).sum(1) / ln

        nz_idx = np.where(nz)[0]
        for i in nz_idx:
            d = int(cnt_local[i])
            if d < 1 or not (ln[i] > 0):
                continue
            pool = nz_idx[nz_idx != i]
            dd = min(d, len(pool))
            if dd < 1:
                continue
            # R size-matched random sets of dd DISTINCT agents (vectorised over R)
            order = np.argpartition(rng.random((R, len(pool))), dd - 1, axis=1)[:, :dd]
            samp = pool[order]                            # (R, dd)
            means = base_masked[samp].mean(1)             # (R, K)
            rn = np.linalg.norm(means, axis=1)
            ok = rn > 0
            if not ok.any():
                continue
            coss = (means[ok] @ Mn[i]) / rn[ok]
            s_local.append(float(sloc_all[i]))
            s_random.append(float(coss.mean()))

    n_valid = len(s_local)
    if n_valid == 0:
        return dict(s_local=np.nan, s_random=np.nan, delta=np.nan,
                    coverage=0.0, n_valid=0)
    sl = float(np.mean(s_local))
    sr = float(np.mean(s_random))
    return dict(s_local=sl, s_random=sr, delta=sl - sr,
                coverage=n_valid / n_active if n_active else 0.0,
                n_valid=n_valid)


def verify_edge_direction(path):
    """Confirm that the agents recorded in ``neighbor_history`` (whom an agent saw)
    are its SUCCESSORS, not predecessors. Returns (ok, detail)."""
    with open(path) as f:
        d = json.load(f)
    conns = d["Connections"]
    succ, pred = {}, {}
    for a, b in conns:
        succ.setdefault(a, set()).add(b)
        pred.setdefault(b, set()).add(a)
    succ_ok = pred_ok = checked = 0
    for ag in d["Agents"]:
        seen = {n["id"] for rec in ag["neighbor_history"] for n in rec.get("neighbors", [])}
        if not seen:
            continue
        checked += 1
        if seen <= succ.get(ag["id"], set()):
            succ_ok += 1
        if seen <= pred.get(ag["id"], set()):
            pred_ok += 1
    return succ_ok == checked and succ_ok > pred_ok, dict(
        checked=checked, succ_ok=succ_ok, pred_ok=pred_ok)


# ── PHQ-9 columns ────────────────────────────────────────────────────────────
def avg_dw_phq9(phq9, A):
    """Mean over time of the out-degree-weighted PHQ-9 (matches
    ``metrics.degree_weighted_mean``: weight by agent_connections = out-degree)."""
    deg = A.sum(1)                     # (N,) out-degree
    total = A.sum()
    if total == 0:
        return float("nan")
    dw_t = (deg[:, None] * phq9).sum(0) / total   # (T,)
    return float(dw_t.mean())


def end_mean_phq9(phq9):
    return float(phq9[:, -1].mean())


# ── table builder ────────────────────────────────────────────────────────────
def build_table(configs=CONFIGS, W=10, step=1, normalize=True, R=10, seed=0, verbose=True):
    markers = load_markers()
    K = len(markers)
    assert K == 241, f"expected 241 CDS markers, got {K}"
    pats, any_pat = compile_patterns(markers)

    # ── pass 1: scan tweets, fit shared IDF, cache sparse TF per seed ─────────
    cache = []        # (label, seed, idx, val, real, A, phq9)
    df = np.zeros(K)
    n_docs = 0
    checked_dir = False
    for label, rel in configs:
        seed_files = sorted(glob.glob(os.path.join(NETWORKS_DIR, rel, ROUNDS_DIR,
                                                   "seed_*", "net.json")))
        if verbose:
            print(f"[load] {label}: {len(seed_files)} seeds")
        for sf in seed_files:
            if not checked_dir:
                ok, detail = verify_edge_direction(sf)
                print(f"[verify] edge direction successors==content-source: "
                      f"{'PASS' if ok else 'FAIL'} {detail}")
                assert ok, "edge direction check failed; flip A to predecessors"
                checked_dir = True
            history, phq9, A = load_net(sf)
            N, T = len(history), len(history[0])
            idx, val, real = net_tf_sparse(history, pats, any_pat, K)
            # tweet-level document frequency (one real tweet = one document)
            tf = dense_tf(idx, val, N, T, K)
            present = (tf > 0)[real]                 # (n_real, K)
            n_docs += int(real.sum())
            df += present.sum(0)
            seed_id = os.path.basename(os.path.dirname(sf)).replace("seed_", "")
            cache.append((label, seed_id, idx, val, real, A, phq9))

    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
    if verbose:
        print(f"[idf] docs={n_docs}  markers seen in >=1 doc: {int((df > 0).sum())}/{K}")

    # ── pass 2: per-seed metrics ─────────────────────────────────────────────
    rows = []
    for label, seed_id, idx, val, real, A, phq9 in cache:
        N, T = real.shape
        tf = dense_tf(idx, val, N, T, K)
        ent = entrainment_for_net(tf, real, A, idf, W=W, step=step,
                                  normalize=normalize, R=R, seed=seed)
        rows.append(dict(
            config=label, seed=seed_id,
            s_local=ent["s_local"], s_random=ent["s_random"], delta=ent["delta"],
            coverage=ent["coverage"], n_valid=ent["n_valid"],
            avg_dw_phq9=avg_dw_phq9(phq9, A), end_mean_phq9=end_mean_phq9(phq9),
        ))
    per_seed = pd.DataFrame(rows)

    # ── aggregate across seeds (mean +/- std), preserve config order ─────────
    order = [c[0] for c in configs]
    metrics_cols = ["s_local", "s_random", "delta", "coverage",
                    "avg_dw_phq9", "end_mean_phq9"]
    g = per_seed.groupby("config", sort=False)
    agg = g[metrics_cols].agg(["mean", "std"])
    n_seeds = g.size().rename("n_seeds")
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.join(n_seeds).reindex(order).reset_index()
    return per_seed, agg


def format_table(agg):
    """Pretty 'mean +/- std' display table."""
    def cell(row, m, nd):
        return f"{row[m + '_mean']:.{nd}f} ± {row[m + '_std']:.{nd}f}"
    disp = pd.DataFrame({
        "config": agg["config"],
        "s_local": agg.apply(lambda r: cell(r, "s_local", 3), axis=1),
        "s_random": agg.apply(lambda r: cell(r, "s_random", 3), axis=1),
        "delta": agg.apply(lambda r: cell(r, "delta", 3), axis=1),
        "coverage": agg.apply(lambda r: cell(r, "coverage", 2), axis=1),
        "avg_dw_phq9": agg.apply(lambda r: cell(r, "avg_dw_phq9", 2), axis=1),
        "end_mean_phq9": agg.apply(lambda r: cell(r, "end_mean_phq9", 2), axis=1),
        "n_seeds": agg["n_seeds"],
    })
    return disp


# ── time-resolved entrainment Delta(t) ───────────────────────────────────────
def entrainment_timeseries_for_net(tf, real, A, idf, W=10, step=1,
                                   normalize=True, R=10, seed=0):
    """Per-window mean entrainment Delta over valid focal agents.

    Same metric as ``entrainment_for_net`` but kept resolved by window instead of
    pooled. Returns (centers, delta_w, n_w): window-centre round, mean Delta in that
    window (nan if no valid focal), and the number of valid focals per window.
    """
    N, T, K = tf.shape
    tfidf = tf * idf
    rng = np.random.default_rng(seed)
    cs = np.concatenate([np.zeros((N, 1, K), dtype=tfidf.dtype),
                         np.cumsum(tfidf, axis=1)], axis=1)

    starts = list(range(0, T - W + 1, step))
    delta_w = np.full(len(starts), np.nan)
    n_w = np.zeros(len(starts), dtype=int)
    for wi, w in enumerate(starts):
        M = cs[:, w + W, :] - cs[:, w, :]
        norms = np.linalg.norm(M, axis=1)
        nz = norms > 0
        if int(nz.sum()) < 3:
            continue
        Mn = np.zeros_like(M)
        Mn[nz] = M[nz] / norms[nz, None]
        base = Mn if normalize else M
        base_masked = base * nz[:, None]
        cnt_local = A @ nz.astype(np.float32)
        sum_local = A @ base_masked
        with np.errstate(divide="ignore", invalid="ignore"):
            local_mean = sum_local / cnt_local[:, None]
            ln = np.linalg.norm(local_mean, axis=1)
            sloc_all = (Mn * local_mean).sum(1) / ln

        deltas = []
        for i in np.where(nz)[0]:
            d = int(cnt_local[i])
            if d < 1 or not (ln[i] > 0):
                continue
            pool = np.where(nz)[0]
            pool = pool[pool != i]
            dd = min(d, len(pool))
            if dd < 1:
                continue
            order = np.argpartition(rng.random((R, len(pool))), dd - 1, axis=1)[:, :dd]
            means = base_masked[pool[order]].mean(1)
            rn = np.linalg.norm(means, axis=1)
            ok = rn > 0
            if not ok.any():
                continue
            srand = float(((means[ok] @ Mn[i]) / rn[ok]).mean())
            deltas.append(float(sloc_all[i]) - srand)
        if deltas:
            delta_w[wi] = float(np.mean(deltas))
            n_w[wi] = len(deltas)
    centers = np.array(starts) + W / 2.0
    return centers, delta_w, n_w


def build_timeseries(configs=CONFIGS, W=10, step=1, normalize=True, R=10, seed=0,
                     verbose=True):
    """Per-config Delta(t): mean and SEM across seeds, window by window."""
    markers = load_markers()
    K = len(markers)
    assert K == 241, f"expected 241 CDS markers, got {K}"
    pats, any_pat = compile_patterns(markers)

    cache, df, n_docs = [], np.zeros(K), 0
    for label, rel in configs:
        seed_files = sorted(glob.glob(os.path.join(NETWORKS_DIR, rel, ROUNDS_DIR,
                                                   "seed_*", "net.json")))
        if verbose:
            print(f"[load] {label}: {len(seed_files)} seeds")
        for sf in seed_files:
            history, _, A = load_net(sf)
            N, T = len(history), len(history[0])
            idx, val, real = net_tf_sparse(history, pats, any_pat, K)
            tf = dense_tf(idx, val, N, T, K)
            df += (tf > 0)[real].sum(0)
            n_docs += int(real.sum())
            cache.append((label, idx, val, real, A))
    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    by_label, centers = {c[0]: [] for c in configs}, None
    for label, idx, val, real, A in cache:
        N, T = real.shape
        tf = dense_tf(idx, val, N, T, K)
        centers, dw, _ = entrainment_timeseries_for_net(
            tf, real, A, idf, W=W, step=step, normalize=normalize, R=R, seed=seed)
        by_label[label].append(dw)

    out = {}
    for label in (c[0] for c in configs):
        mat = np.vstack(by_label[label])                      # (n_seeds, n_windows)
        cnt = np.sum(~np.isnan(mat), axis=0)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(cnt, 1))
        out[label] = (mean, sem, mat)
    return centers, out


def plot_entrainment_timeseries(centers, out, out_png, W, step, smooth=11, band=True):
    """One Delta(t) line per topology (seed-mean), optional +/-SEM band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def roll(x, k):
        if k and k > 1:
            return pd.Series(x).rolling(int(k), center=True, min_periods=1).mean().to_numpy()
        return x

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for k, (label, (mean, sem, _)) in enumerate(out.items()):
        c = SA_PALETTE[k % len(SA_PALETTE)]
        m = roll(mean, smooth)
        ax.plot(centers, m, lw=1.2, color=c, label=label.split(" (")[0])
        if band:
            s = roll(sem, smooth)
            ax.fill_between(centers, m - s, m + s, alpha=0.15, color=c, linewidth=0)
    ax.axhline(0.0, color="0.5", lw=0.7, ls="--")
    ax.set_xlim(0, centers[-1] + centers[0])     # full round range (0-300)
    ax.set_xlabel("round", fontsize=8)
    ax.set_ylabel(r"entrainment $\Delta$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper left",
              handlelength=1.2, columnspacing=1.0, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=int, default=10, help="window length W (rounds)")
    ap.add_argument("--step", type=int, default=1, help="window step")
    ap.add_argument("--repeats", type=int, default=10,
                    help="R: random draws for the size-matched baseline")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the baseline")
    ap.add_argument("--no-normalize", action="store_true",
                    help="do NOT L2-normalise neighbour vectors before averaging")
    ap.add_argument("--out", default=OUT_DIR, help="output directory for CSVs")
    ap.add_argument("--plot", action="store_true",
                    help="compute time-resolved Delta(t) and save the line plot")
    ap.add_argument("--smooth", type=int, default=11,
                    help="rolling-mean window (in windows) for the plot; 1 = none")
    args = ap.parse_args()

    if args.plot:
        centers, ts = build_timeseries(W=args.window, step=args.step,
                                       normalize=not args.no_normalize,
                                       R=args.repeats, seed=args.seed)
        os.makedirs(args.out, exist_ok=True)
        png = os.path.join(args.out, "entrainment_timeseries.png")
        plot_entrainment_timeseries(centers, ts, png, W=args.window, step=args.step,
                                    smooth=args.smooth)
        # per-window seed-mean Delta(t), one column per config
        df_ts = pd.DataFrame({"round_center": centers})
        for label, (mean, sem, _) in ts.items():
            df_ts[label] = mean
            df_ts[label + "__sem"] = sem
        csv = os.path.join(args.out, "entrainment_timeseries.csv")
        df_ts.to_csv(csv, index=False)
        print(f"[saved] {png}")
        print(f"[saved] {csv}")
        return

    per_seed, agg = build_table(W=args.window, step=args.step,
                                normalize=not args.no_normalize,
                                R=args.repeats, seed=args.seed)

    print("\n=== CDS local entrainment — directed topologies "
          f"(W={args.window}, step={args.step}, R={args.repeats}, "
          f"{'normalised' if not args.no_normalize else 'raw'} neighbours) ===")
    print("s_random = size-matched baseline (same #agents as in-neighbours).")
    print("Delta = s_local - s_random  (>0 => in-neighbours more alike than baseline)")
    print("Note: no higher-clustering directed SDA exists; the a2.17 vs a1.17 pair is")
    print("      the clustering contrast, confounded with degree.\n")
    disp = format_table(agg)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(disp.to_string(index=False))

    cov = agg["coverage_mean"].min()
    if cov < 0.2:
        print(f"\n[warn] min mean coverage {cov:.2f} < 0.20 — consider --window 20.")

    os.makedirs(args.out, exist_ok=True)
    per_seed.to_csv(os.path.join(args.out, "entrainment_per_seed.csv"), index=False)
    agg.to_csv(os.path.join(args.out, "entrainment_table.csv"), index=False)
    disp.to_csv(os.path.join(args.out, "entrainment_table_display.csv"), index=False)
    print(f"\n[saved] {os.path.join(args.out, 'entrainment_table.csv')}")
    print(f"[saved] {os.path.join(args.out, 'entrainment_per_seed.csv')}")


if __name__ == "__main__":
    main()
