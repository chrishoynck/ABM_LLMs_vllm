# GABM (CLS) thesis — chapter/figure → code map

Thesis: the Computational Science ("CLS") GABM thesis, repo
`~/thesis/Computational_Science_GABM/` (UvA MSc Computational Science). Code
repo (this one): `https://github.com/chrishoynck/ABM_LLMs_vllm.git`. Paths
relative to the code-repo root. Doc index: [../README.md](../README.md).

| Thesis part | Repo component | Generator | Data inputs |
|---|---|---|---|
| Methods §Simulation; App. A (simulation-step Alg.) | `src/classes/{network,agent}.py`, `src/llama_activate.py` | `sbatch jobs/run_simulation_sda.job` / `run_simulation_sdc.job` (2×A100, 9h) | bias table, personas; outputs `data/networks_post/…` (`net.json` checkpoints) |
| Methods §Network (SDA eq., SDC stub matching) | `src/classes/network.py` (social space, bisection on `b`, power-law fit, stub matcher) | — (part of simulation) | — |
| App. E §Sobol Sensitivity + §Parameter Search | `src/utils/sensitivity/sa_network.py` | `bash scripts/sensitivity/run_sa_network.sh` (CPU) | `data/confidential/phq9.sav`; outputs `data/sensitivity/network_100_3/` |
| Exp. §Network Statistics (`network_target_ranges{,_sdc}.png`) | `src/utils/sensitivity/plot_network_targets{,_sdc}.py` | hand-run — see `../src/README.md` ("Hand-run CLIs") | saved simulation runs |
| Metrics §Semantic Drift + Results §Lexical Entrainment (global) | `src/utils/analyses/lexical_entrainment/global/plot_lexical_entrainment.py` | `sbatch jobs/run_lexical_entrainment.job` | `data/networks_post/` tweets; outputs `plots/lexical_entrainment/` |
| App. C §Local Language Entrainment + §CDS detection | `src/utils/analyses/lexical_entrainment/local/cds_entrainment.py`, `src/utils/metrics.py` | hand-run — see `../src/README.md` | `data/distorted_language_ngrams.tsv`; outputs `plots/lexical_entrainment/local/` |
| Results §Phase Space (`phase_plot_*`, `compare_calibrated_*`) | `experiment.ipynb` + `src/utils/visualization.py` | run the notebook | `data/networks_post/` |
| App. C §PHQ-9 and Assortativity Trajectories; CDS over time | `src/utils/network_evolution.py` via `src/utils/tools/plot_network_evolution.py` | `bash scripts/plotting/run_plot_evolution.sh` (CPU, idempotent); `sbatch jobs/run_plots.job` | `data/networks_post/`; per-run `plots/` |
| Results §Mobility (+ non-debiased variant, App. C) | `src/utils/tools/plot_phq9_mobility.py` | hand-run — see `../src/README.md` | saved runs; outputs `data/networks_post/basis/plots/` |
| Metrics §Velocity of Contagion | `src/utils/tools/velocity_table.py` | hand-run — see `../src/README.md` | saved runs; output `plots/velocity_table.tex` |
| App. B §Bias Correction; Exp. §Debiasing | `src/utils/tools/phq9_bias.py` (used inside `network.py`) | `sbatch jobs/run_bias_calibration.job` | unseen persona pool; output `phq9_bias_table.csv` (sims load the notebook-exported `_fullfit` variant) |
| App. B shared pipeline material (data generation, prompt opt, MentalBERT+MLP) | same components as the CS thesis | — | see [THESIS_MAP_CS.md](THESIS_MAP_CS.md) |
| Exp./App. C §Happy Hub | happy-hub configs in `scripts/simulation/run_simulation_{sda,sdc}.sh` | same simulation jobs | `data/happy_persona.csv`; outputs `data/networks_post/happy/` |

Build instructions + known LaTeX quirks live in the thesis repo's own `README.md`.
