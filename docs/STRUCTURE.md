# Repo structure — what is used where

GABM depression simulation + PHQ-9 assessment pipeline. Grew across two theses
(CS = SA + generative/assessment model performance; GABM = network simulation)
and the paper `LLM_agent_Depression__PNAS_Nexus`. Doc index: the repo-root
`README.md`. Companions in this folder: `ROADMAP.md` (pipeline walk-through,
stages tagged per manuscript), `SCRIPTS.md` (every driver script + SLURM job),
`PAPER_MAP_PNAS.md` (paper figure/table provenance), `THESIS_MAP_CS.md` /
`THESIS_MAP_GABM.md` (thesis chapter→code maps), `prompt_optimizer.md`
(TextGrad deep-dive). Per-folder `NOTES.md` files hold
data details. All paths in this file are relative to the repo root.

| Folder / file | Used by | Status |
|---|---|---|
| `README.md` + `docs/` | all | project documentation (this file + the six companion docs; `README.md` is the front door) |
| `src/` | all | live code (classes, utils, sensitivity, analyses) — module map in `src/README.md` |
| `scripts/{simulation,sensitivity,assessment,plotting,data_generation}/` + `jobs/` | all | live drivers + versioned SLURM wrappers — see `SCRIPTS.md` for the full map |
| `checks/` | CS + paper | small quality checks with their own jobs (surface cues, multi-model smoke run, CDS vs PHQ-9); see `checks/README.md` |
| `data/prompts_optimal.json` | all | LIVE prompt file (`FC.PROMPTS_FILE`) — has dead keys, see `data/NOTES.md` |
| `data/prompts_post.json` | provenance | HISTORICAL — the generation prompt (`tweet_gen.*`) behind the high-fidelity set; obsolete `phq9.*` keys removed 2026-08-27 (see `data/test_post/Qwen_Qwen3.5-27B/NOTES.md`) |
| `data/prompts_post_minimal.json` | CS/paper | live (minimal prompt baseline) |
| `data/networks_post/` | GABM | live simulation runs |
| `data/test_post/` | CS + paper | live: high-fidelity set, regressors, prompt-opt evals |
| `data/test/Qwen/` | CS | live embeddings cache; rest of `data/test/` binned |
| `data/sensitivity/` | CS + GABM + paper | live SA trees (axes, decoding, phq9, network Sobol) |
| `data/prompt_optimization_h/` | CS | human-in-the-loop prompt opt; prompt-variant SA (`prompt_sa*`, `SA_prompt`) kept but **in no manuscript** |
| `data/finetune/`, `data/confidential/`, `data/personas_*.csv` | all | live inputs |
| `data/grok_posts/` | none | generator output, no manuscript consumer; the generator script was binned 2026-09 |
| `data/methodology_paper/` | **different paper** | frozen — do not touch (incl. its own `prompts.json`) |
| `plots/` | CS/GABM/paper | figure outputs (`cds_validation.png`, notebook PNGs, `lexical_entrainment/`) |
| `logs/` | — | run logs only; nothing reads them back |
| `bin/` | — | recoverable discard (cleanups 2026-08 and 2026-09); restore = `mv bin/X X`; see `bin/NOTES.md` |
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
- (cleanup 2026-09-14) The dead functions that used to be listed here were deleted, and
  `generate_synthetic_dataset.py` / `generate_posts_grok.py` were binned; `bin/NOTES.md`
  has the full list and the recovery commands.
- PHQ-9 severity banding (0-4/5-9/10-14/15-19/20-27) is implemented 10 times:
  `agent.phq9_severity_category`, `prompt_optimizer._phq9_severity`,
  `test_phq9_llms._phq9_severity`, `sa_analyze.PHQ9_BANDS`,
  `network_evolution.cds_validation_summary`, `plot_assessment_diagnostics.PHQ9_BANDS`,
  `visualization._phq9_severity_color`, `visualization._MM_BANDS`,
  `visualization.plot_multimodel_band_bias` and `checks/check_cds_tracks_phq9.SEVERITY_BANDS`.
  All live; a divergence silently corrupts figures.
- 3 independent CDS detectors: `metrics` (flat substring, the weakest, but wired into
  the live simulation), `tools/cds.py` (category-aware word-boundary regex, the
  validated one, used by `network_evolution` and `checks/check_cds_tracks_phq9.py`),
  `cds_entrainment` (241-term TF-IDF vocab). `llama_activate.call_visualizations`
  still emits the CDS panels that `network_evolution.py` supersedes.
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
