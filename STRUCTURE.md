# Repo structure — what is used where

GABM depression simulation + PHQ-9 assessment pipeline. Grew across two theses
(CS = SA + generative/assessment model performance; GABM = network simulation)
and the paper `LLM_agent_Depression__PNAS_Nexus`. See `PAPER_MAP_PNAS.md` for
the paper's figure/table provenance, `SCRIPTS.md` for every driver script and
SLURM job, and per-folder `NOTES.md` files for details.

| Folder / file | Used by | Status |
|---|---|---|
| `src/` | all | live code (classes, utils, sensitivity, analyses) |
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
