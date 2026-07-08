#!/usr/bin/env python3
"""
Velocity-of-contagion table (time-to-PHQ-9>=10) for undirected + debiased + main config.

For each cell (setting x model) we build per-round trajectories of:
  - degree-weighted mean PHQ-9  (matches metrics.degree_weighted_mean)
  - plain mean PHQ-9
  - fraction of agents with PHQ-9 >= 10
averaged over the 5 seeds, then report the first round the *mean trajectory* crosses
each threshold (or NR = not reached within the run). Per-seed diagnostics printed too.

Run:  PYTHONPATH=src python src/utils/tools/velocity_table.py
"""
import json
import os
import numpy as np

# Repo-relative data root: <repo>/data/networks_post  (file lives at src/utils/tools/).
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ROOT = os.path.join(REPO, "data", "networks_post")
SEEDS = [14, 15, 16, 17, 18]
ROUNDS_DIR = "rounds300_N100"
PHQ_THRESH = 10.0
FRAC_THRESH = 0.5

# (setting_dir, setting_label, model, config_dir, config_label, leaf_subdir)
# leaf_subdir is the extra path level under rounds300_N100 ("init_0" for the
# init_0 runs, "" otherwise).
CELLS = [
    ("basis", "init_0", "sda", "2_1655_d4_5_dim5",    "SDA -- calibrated (main)",  "init_0"),
    ("basis", "normal", "sda", "2_1655_d4_5_dim5",    "SDA -- calibrated (main)",  ""),
    ("happy", "happy",  "sda", "2_1655_d4_5_dim5",    "SDA -- calibrated (main)",  ""),
    # high PHQ-9_rho variant (high initial PHQ-9 assortativity) -- only "normal" was run.
    ("basis", "normal", "sda", "2_1655_d4_5_dim3",    r"SDA -- high PHQ-9$_\rho$", ""),
    ("basis", "init_0", "sdc", "4_9429_d8_2539_dim3", "SDC -- calibrated (main)",  "init_0"),
    ("basis", "normal", "sdc", "4_9429_d8_2539_dim3", "SDC -- calibrated (main)",  ""),
    ("happy", "happy",  "sdc", "4_9429_d8_2539_dim3", "SDC -- calibrated (main)",  ""),
    ("basis", "normal", "sdc", "8_0_d8_2539_dim2",    r"SDC -- high PHQ-9$_\rho$", ""),
]


def load_seed(path):
    """Return (phq9 matrix [N x T], degree vector [N]) from one net.json."""
    with open(path) as f:
        d = json.load(f)
    phq9 = np.array([a["phq9"] for a in d["Agents"]], dtype=float)  # N x T
    n = d["Number of Agents"]
    deg = np.zeros(n)
    for u, v in d["Connections"]:
        deg[u] += 1
        deg[v] += 1
    return phq9, deg


def series_for_seed(phq9, deg):
    """Per-round dw-mean, plain mean, frac>=10 for one seed."""
    total_deg = deg.sum()
    dw = (deg[:, None] * phq9).sum(axis=0) / total_deg if total_deg > 0 else phq9.mean(axis=0)
    mean = phq9.mean(axis=0)
    frac = (phq9 >= PHQ_THRESH).mean(axis=0)
    return dw, mean, frac


def first_crossing(series, thresh):
    """First round index where series >= thresh, else None."""
    idx = int(np.argmax(series >= thresh))
    if series[idx] >= thresh:
        return idx
    return None


def fmt(x):
    return "NR" if x is None else str(x)


def main():
    rows = []
    for set_dir, set_lab, model, cfg, cfg_lab, leaf in CELLS:
        base = os.path.join(ROOT, set_dir, model, "undirected", "debiased", cfg, ROUNDS_DIR, leaf)
        dw_seeds, mean_seeds, frac_seeds, end_means = [], [], [], []
        per_seed = {"dw": [], "frac": [], "mean": []}
        for s in SEEDS:
            p = os.path.join(base, f"seed_{s}", "net.json")
            if not os.path.exists(p):
                print(f"  [warn] missing: {p}")
                continue
            phq9, deg = load_seed(p)
            dw, mean, frac = series_for_seed(phq9, deg)
            dw_seeds.append(dw); mean_seeds.append(mean); frac_seeds.append(frac)
            end_means.append(mean[-1])
            per_seed["dw"].append(first_crossing(dw, PHQ_THRESH))
            per_seed["mean"].append(first_crossing(mean, PHQ_THRESH))
            per_seed["frac"].append(first_crossing(frac, FRAC_THRESH))

        if not dw_seeds:
            print(f"  [warn] no seeds for {set_lab}/{model} — skipping cell")
            continue

        # mean trajectory across seeds
        dw_m = np.mean(dw_seeds, axis=0)
        mean_m = np.mean(mean_seeds, axis=0)
        frac_m = np.mean(frac_seeds, axis=0)

        r_dw = first_crossing(dw_m, PHQ_THRESH)
        r_frac = first_crossing(frac_m, FRAC_THRESH)
        r_mean = first_crossing(mean_m, PHQ_THRESH)
        end_mean = float(np.mean(end_means))
        # Sample SD (ddof=1) of the end-of-run mean PHQ-9 across the seeds.
        end_sd = float(np.std(end_means, ddof=1)) if len(end_means) > 1 else 0.0

        n_seeds = len(dw_seeds)
        def reached(lst):
            return sum(x is not None for x in lst)

        # Per-seed crossing statistics: (n_reached, n_total, mean, SD) over the seeds
        # that reached the threshold. SD is None when fewer than 2 seeds reached.
        def stat(lst):
            vals = [x for x in lst if x is not None]
            n = len(vals)
            m = float(np.mean(vals)) if n else None
            s = float(np.std(vals, ddof=1)) if n > 1 else None
            return (n, n_seeds, m, s)
        dw_stat, frac_stat, mean_stat = stat(per_seed['dw']), stat(per_seed['frac']), stat(per_seed['mean'])

        rows.append((cfg_lab, set_lab, dw_stat, frac_stat, mean_stat, end_mean, end_sd))

        # diagnostics
        def cross_stats(lst):
            """mean +/- SD of the per-seed crossing rounds, over seeds that reached."""
            vals = [x for x in lst if x is not None]
            if not vals:
                return "(none reached)"
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            return f"per-seed mean {m:.0f} +/- {s:.0f}"
        print(f"\n### {cfg_lab}  |  setting={set_lab}  ({set_dir}/{model})")
        print(f"  per-seed DW-mean>=10   crossings: {[fmt(x) for x in per_seed['dw']]}   reached {reached(per_seed['dw'])}/{n_seeds}   {cross_stats(per_seed['dw'])}")
        print(f"  per-seed 50%pop>=10    crossings: {[fmt(x) for x in per_seed['frac']]}   reached {reached(per_seed['frac'])}/{n_seeds}   {cross_stats(per_seed['frac'])}")
        print(f"  per-seed mean>=10      crossings: {[fmt(x) for x in per_seed['mean']]}   reached {reached(per_seed['mean'])}/{n_seeds}   {cross_stats(per_seed['mean'])}")
        print(f"  end mean PHQ-9 per seed: {[round(x,2) for x in end_means]}  -> {end_mean:.2f} +/- {end_sd:.2f} (SD)")
        print(f"  MEAN-TRAJECTORY crossings: DW={fmt(r_dw)}  50%pop={fmt(r_frac)}  mean={fmt(r_mean)}")

    # ---- console table ----
    def cell_txt(st):
        n, tot, m, s = st
        if n == 0:
            return f"NR (0/{tot})"
        if s is None:
            return f"{m:.0f} ({n}/{tot})"
        return f"{m:.0f}+/-{s:.0f} ({n}/{tot})"

    print("\n\n========= VELOCITY-OF-CONTAGION TABLE (per-seed crossing mean +/- SD) =========")
    hdr = ["Config", "Setting", "R:DW>=10", "R:50%pop>=10", "R:mean>=10", "End mean"]
    print(f"{hdr[0]:<26}{hdr[1]:<9}{hdr[2]:>16}{hdr[3]:>16}{hdr[4]:>16}{hdr[5]:>14}")
    for cfg_lab, set_lab, dw_stat, frac_stat, mean_stat, end_mean, end_sd in rows:
        clean = cfg_lab.replace("--", "-")
        end_cell = f"{end_mean:.2f}+/-{end_sd:.2f}"
        print(f"{clean:<26}{set_lab:<9}{cell_txt(dw_stat):>16}{cell_txt(frac_stat):>16}{cell_txt(mean_stat):>16}{end_cell:>14}")

    # ---- LaTeX ----
    tex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "velocity_table.tex")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Velocity of contagion for the undirected, debiased calibrated (main) networks:"
        r" the round at which the population reaches a moderate-depression level (PHQ-9 $\geq 10$)."
        r" For each metric --- degree-weighted mean PHQ-9 ($R_{\mathrm{DW}}$), fraction of agents with"
        r" PHQ-9 $\geq 10$ (half the population, $R_{50\%}$), and plain population mean PHQ-9"
        r" ($R_{\bar{\mu}}$) --- we report the per-seed crossing round as mean~$\pm$~SD over the seeds"
        r" that reached the threshold, with the number of seeds reaching it (out of 5) in parentheses."
        r" The SD is omitted when only one seed reaches the threshold; NR~(0/5) means no seed reached it"
        r" within 300 rounds. The end-of-run (round 300) mean PHQ-9 is mean~$\pm$~SD across all 5 seeds"
        r" (sample SD throughout).}",
        r"\label{tab:velocity_contagion}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"\textbf{Config} & \textbf{Setting} & \textbf{$R_{\mathrm{DW}\geq10}$} & "
        r"\textbf{$R_{50\%\geq10}$} & \textbf{$R_{\bar{\mu}\geq10}$} & \textbf{End mean PHQ-9} \\",
        r"\midrule",
    ]

    def cell_tex(st):
        n, tot, m, s = st
        if n == 0:
            return f"NR (0/{tot})"
        if s is None:
            return f"{m:.0f} ({n}/{tot})"
        return f"{m:.0f} $\\pm$ {s:.0f} ({n}/{tot})"

    for cfg_lab, set_lab, dw_stat, frac_stat, mean_stat, end_mean, end_sd in rows:
        set_tex = set_lab.replace("_", r"\_")   # escape init_0 for LaTeX
        lines.append(f"{cfg_lab} & {set_tex} & {cell_tex(dw_stat)} & {cell_tex(frac_stat)} & "
                      f"{cell_tex(mean_stat)} & {end_mean:.2f} $\\pm$ {end_sd:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nLaTeX table written to: {tex_path}")


if __name__ == "__main__":
    main()
