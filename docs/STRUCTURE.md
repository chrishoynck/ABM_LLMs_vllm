# Repo structure — what is used where

GABM depression simulation + PHQ-9 assessment pipeline. Grew across two theses
(CS = SA + generative/assessment model performance; GABM = network simulation)
and the paper `LLM_agent_Depression__PNAS_Nexus`. Doc index: the repo-root
`README.md`. Companions in this folder: `SCRIPTS.md` (every driver script +
SLURM job), `PAPER_MAP_PNAS.md` (paper figure/table provenance),
`THESIS_MAP_CS.md` / `THESIS_MAP_GABM.md` (thesis chapter→code maps),
`prompt_optimizer.md` (TextGrad deep-dive). Per-folder `NOTES.md` files hold
data details. All paths in this file are relative to the repo root.

| Folder / file | Used by | Status |
|---|---|---|
| `README.md` + `docs/` | all | project documentation (this file + the five companion docs; `README.md` is the front door) |
| `src/` | all | live code (classes, utils, sensitivity, analyses) — module map in `src/README.md` |
| `scripts/{simulation,sensitivity,assessment,plotting,data_generation}/` + `jobs/` | all | live drivers + versioned SLURM wrappers — see `SCRIPTS.md` for the full map |
| `data/prompts_optimal.json` | all | LIVE prompt file (`FC.PROMPTS_FILE`) — has dead keys, see `data/NOTES.md` |
| `data/prompts_post.json` | provenance | HISTORICAL — generated the high-fidelity set (see `data/test_post/Qwen_Qwen3.5-27B/NOTES.md`) |
| `data/prompts_post_minimal.json` | CS/paper | live (minimal prompt baseline) |
| `data/networks_post/` | GABM | live simulation runs |
| `data/test_post/` | CS + paper | live: high-fidelity set, regressors, prompt-opt evals |
| `data/test/Qwen/` | CS | live embeddings cache; rest of `data/test/` binned |
| `data/sensitivity/` | CS + GABM + paper | live SA trees (axes, decoding, phq9, network Sobol) |
| `data/prompt_optimization_h/` | CS | human-in-the-loop prompt opt; prompt-variant SA (`prompt_sa*`, `SA_prompt`) kept but **in no manuscript** |
| `data/finetune/`, `data/confidential/`, `data/personas_*.csv` | all | live inputs |
| `data/grok_posts/` | none | generator output, no manuscript consumer |
| `data/methodology_paper/` | **different paper** | frozen — do not touch (incl. its own `prompts.json`) |
| `plots/` | CS/GABM/paper | figure outputs (`cds_validation.png`, notebook PNGs, `lexical_entrainment/`) |
| `logs/` | — | run logs only; nothing reads them back |
| `bin/` | — | recoverable discard (cleanup 2026-08); restore = `mv bin/X X`; see `bin/NOTES.md` |
| `experiment.ipynb` | CS/paper | live entry point for several figures |

## Known quirks (flagged, deliberately not fixed here)
- CS thesis broken includes (fix in the thesis repo, not here): `Chapters/Methods.tex:111`
  (`textgrad_diagram.png` vs on-disk `Textgrad_diagram.png`) and
  `Chapters/ExperimentsandResults.tex:36` (`test_post_opt.png` vs `Test_post_opt.png`).
- PNAS manuscript: empty captions at `manuscript.tex:96,108`; 16 unused files in `Fig/`.
- (fixed 2026-08) the SLURM jobs moved from `$HOME` to `jobs/` and now all use the
  standard `PYTHONPATH=src … utils.…` import form; `SCRIPTS.md` has the rename map.
- Naming leaks: tweet-vs-post vocabulary (`optimized_tweets/` holds post prompts;
  `tweets_with_phq9.csv` in post mode), `_h` suffix only on `prompt_optimization_h`,
  `SA_prompt` vs `prompt_sa` vs `prompt_sa_reps` (four near-identical names, one dir).
- `data/prompts_optimal.json` dead/broken keys — see `data/NOTES.md`; removing them
  needs edits in `src/classes/agent.py:246,282` (later code-cleanup stage).
- (audit 2026-08-26) Dead functions, documented here instead of deleted:
  `visualization.py` — `plot_tf_idf_PCA:347`, `plot_bias:1030`, `plot_phq9_error:1063`,
  `plot_combined_bias_error:1087`, `plot_model_comparison_by_settings:1227`;
  `metrics.py` — `analyze_distorted_language:136`, `all_agent__tweet_cd:1124` (name typo),
  `compute_prompt_robustness:1152` (live copy: `sa_analyze.py:1840`);
  `sa_analyze.py` — `draw_prompt_sim_heatmap:1997` (live twin: `_draw_prompt_sim_heatmap:1894`).
- `generate_synthetic_dataset._build_network:51` claims to duplicate
  `llama_activate.build_network:121` but has diverged — omits 8 kwargs (`n_clusters`,
  `latent_weight`, `age_weight`, `gamma`, `phq9_mode`, `bert_regressor_path`,
  `bias_table_path`, `bert_mentalbert`), so it silently builds a differently
  parameterized network.
- PHQ-9 severity banding (0-4/5-9/10-14/15-19/20-27) is implemented 10×:
  `agent.phq9_severity_category`, `prompt_optimizer:1354`, `test_phq9_llms:15`,
  `generate_posts_grok:51`, `validate_cds:127`, `sa_analyze:59`,
  `network_evolution:1078`, `plot_confusion_depression:84`,
  `plot_sbert_cosine_conditioning:100`, `visualization:2052`. All live — a
  divergence silently corrupts figures.
- 3 independent CDS detectors: `metrics` (flat substring — weakest, but wired into
  the live simulation), `validate_cds` (category-aware word-boundary regex — the
  validated one, used by `network_evolution`), `cds_entrainment` (241-term TF-IDF
  vocab). `llama_activate.call_visualizations:472-499` still emits the CDS panels
  that `network_evolution.py`'s docstring documents as wrong.
- Package `src/utils/analyses/lexical_entrainment/global/` is named after a reserved
  keyword — `import …global.…` is a SyntaxError; only reachable via `python -m`
  (as `scripts/plotting/run_lexical_entrainment.sh` does). Never import it from code.
- `experiment.ipynb` imports the same modules under two names (`utils.*` via
  `sys.path.append("src")` and `src.utils.*` via cwd) → two module objects, a
  footgun with `%autoreload 2`; cell 27 also re-implements
  `visualization.plot_semantic_entrainment` inline.
- Exec bit missing on `scripts/assessment/run_finetune.sh`,
  `scripts/plotting/run_lexical_entrainment.sh`,
  `scripts/sensitivity/sa_prompt_baseline_run.sh` — invoke via `bash scripts/…`
  (as the jobs do) or `chmod +x`.
