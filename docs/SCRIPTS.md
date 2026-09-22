# SCRIPTS.md — driver scripts + SLURM jobs (layout 2026-08)

All paths below are relative to the repo root. Doc index: [../README.md](../README.md).

All drivers live in `scripts/<stage>/` and self-locate — run them from anywhere.
Jobs live in `jobs/`; submit from anywhere: `sbatch jobs/<name>.job`.
SLURM output goes to `<repo-root>/slurm_output_<jobid>.out` (gitignored).

## simulation
| script | purpose | job (GPU/time) | inputs | outputs |
|---|---|---|---|---|
| `run_simulation_sda.sh` | full ABM, calibrated SDA, happy hub | `run_simulation_sda.job` (2×A100, 9h) | bias table, personas | `data/networks_post/happy/sda/...` |
| `run_simulation_sdc.sh` | SDC counterpart | `run_simulation_sdc.job` (2×A100, 9h) | same | `data/networks_post/happy/sdc/...` |

## sensitivity
| script | purpose | job | inputs | outputs |
|---|---|---|---|---|
| `run_sa_network.sh` | network-topology Sobol SA (CPU) | — | `data/confidential/phq9.sav` | `data/sensitivity/network_n100/` |
| `sa_run.sh` | neighbour/agent/joint + PHQ-9 axes | — (GPU needed) | `data/sensitivity/inputs/` | `data/sensitivity/<axis>/` |
| `sa_decoding_run.sh` | temp/top_p axis | — (GPU needed) | same | `data/sensitivity/decoding/` |
| `sa_phq9_minimal_run.sh` | PHQ-9 bands, minimal prompt | `sa_phq9_minimal_run.job` (2×A100, 2h) | same | `data/sensitivity/phq9_minimal_prompt/` |
| same, env overrides `SA_MODEL SA_PROMPT NUM_PHQ9_REPS SA_ROOT OUT_SUBDIR PYTHON` | PHQ-9 bands for another generator (only the PHQ-9 axis, no stochastic axes) | `sa_phq9_gemma4.job` (array 0-1: iter_10 × 3 reps, iter_0 × 1 rep; 2×A100, 5h each), then `sa_phq9_gemma4_analyze.job` (`--dependency=afterok`; S-BERT embed + `sa_analyze --root data/sensitivity/gemma4`) | same | `data/sensitivity/gemma4/{phq9,phq9_minimal_prompt}/`, `.../gemma4/plots_sbert/` |
| `sa_phq9_minimal_run.sh` / `sa_decoding_run.sh` with `SA_SEED_MODE=agent` | per-agent-seeded rerun of the PHQ-9 axis (iter_10 × 3, iter_0 × 1) and the decoding grid for Qwen and Gemma (Gemma grid via `SA_GRID`, `SA_BASE_SEED` 1000 / 2000) | `sa_seeded.job` (array 0-3; 2×A100, 5h), then `sa_seeded_analyze.job` (`--dependency=afterok`; MentalBERT + S-BERT embed, `sa_analyze`, `sa_phq9` per root with each generator's regressor) | same | `data/sensitivity/seeded/{qwen,gemma4}/{phq9,phq9_minimal_prompt,decoding}/`, `.../plots_sbert/`, `.../plots_phq9/`; `checks/check_decoding_alignment.py` compares the two grids |
| `sa_prompt_run.sh` | prompt axis, single draw | — (GPU needed) | `qwen27_baseline/inputs/` | `.../prompt_sa/` |
| `sa_prompt_baseline_run.sh` | prompt axis w/ replicates | `sa_prompt_baseline_run.job` (2×A100, 3.5h; job adds embed + analyze steps) | same | `.../prompt_sa_reps/` |

## assessment
| script | purpose | job | inputs | outputs |
|---|---|---|---|---|
| `run_finetune.sh` | generate posts + finetune BERT regressor + eval; env-var config, `GEN_TAG=<tag> GEN_MODEL=<alias>` for other generators (unset == `qwen`) | `run_finetune_gemma4.job` (2×A100, 6h) | `data/finetune/` posts, `personas_test_300.csv` | `data/assessors/bert/<arm>/{models,eval/on_<arm>}/`, `assessors/bert/teacher/eval/on_<arm>/` |
| `run_llm_assessor_on_heldout.sh <tag>` | Qwen assessor (minimal + TextGrad prompts) on another generator's 300-block set, then the multimodel summary | `run_llm_assessor_gemma4.job` (2×A100, 2h; submit with `--dependency=afterok`) | `data/finetune/<tag>/test_posts_<tag>.csv` | `optimized_phq9/*/{minimal,eval_on}_<tag>300/` |
| `python -m utils.visualization multimodel` (CPU) | generator x estimator MAE/bias table + linearity/bias figures | — | the eval CSVs above | `method_comparison/multimodel/` |
| `run_llm_assessor_on_heldout.sh` variants for panel (c) | four graded arms + the per-generator figures | `grade_posts_panelc.job` (2×A100, 2h) | the 300-block sets | `data/test_post/teacher_grades/`, `method_comparison/multimodel/linearity_bias_{qwen,gemma}.*` |
| `run_finetune.sh` with the minimal start prompt | minimal-prompt fine-tune arm, both generators | `run_finetune_minimal.job` (2×A100, 5h, array 0-1) | `data/finetune/{qwen,gemma4}_minimal/` | `data/assessors/bert/{qwen27,gemma4}_minimal/` |
| `python -m utils.create_data.generate_test_data` (Gemma venv) | Gemma-4 bias-calibration posts | `calibration_gemma4.job` (2×A100, 2h) | persona pool | `data/finetune/gemma4/calibration_posts_gemma4.csv` |
| `python -m utils.prompt_optimizer --mode grade-posts --posts-file <csv> --out-csv <csv>` | Qwen teacher grades every block of an existing posts CSV (0–10, the post-optimizer's rating prompt) | `grade_posts_multimodel.job` (2×A100, 2h; grades both paired 300-block sets, then the multimodel summary) | `data/finetune/qwen/test_posts_qwen.csv`, `data/finetune/gemma4/test_posts_gemma4.csv` | `data/test_post/teacher_grades/grades_{qwen,gemma4}300.csv` |
| `run_bias_calibration.sh` | 28-level PHQ-9 bias table | `run_bias_calibration.job` (2×A100, 6h) | unseen persona pool | `phq9_bias_table.csv` (note: sims load the notebook-exported `_fullfit` variant) |
| `run_phq9_on_bert_testset.sh` | prompts scored on BERT holdout | — (GPU needed) | embeddings cache (`data/test/Qwen/`) | `optimized_phq9/*/eval_on_*` |
| `run_minimal_shift.sh` | minimal vs optimized prompt under shift | — (GPU needed) | `data/finetune/qwen/test_posts_qwen.csv` | `minimal_*/` subdirs + the reprinted comparison table |
| `run_eval_comparison.sh` | estimator-comparison table (CPU) | — | eval CSVs on disk | stdout only — the PNAS Tables 1–2 source |

## plotting
| script | purpose | job | inputs | outputs |
|---|---|---|---|---|
| `run_plot_evolution.sh` | regenerate network-evolution figs (CPU, idempotent) | — | `data/networks_post/` | per-run `plots/` |
| `run_lexical_entrainment.sh` | MentalBERT entrainment grids | `run_lexical_entrainment.job` (1×A100, 1h) | networks_post tweets | `plots/lexical_entrainment/` |

## data_generation
`create_data_menu.sh` — menu of 5 generation pipelines, uncomment ONE block (GPU for most).

## inline jobs (no .sh behind them)
`run_plots.job` (replot saved SDA nets, 1×A100 1h) · `run_teacher_eval_iter0.job`
(teacher-eval of the iter_0 post-gen prompt — ex `run_test_phq9.job`) ·
`run_bert_optimizer.job` (train regressor, MIG 1h) · `run_prompt_optimizer.job` (H100 5h) ·
`run_prompt_optimizer_phq9.job` (H100 1h) · `run_prompt_optimizer_phq9_human.job` (array of 5 seeds,
H100 5h each, ~6 h needed so expect one `RESUME=1` resubmit: TextGrad assessment prompt on the human-optimized corpus, minimal start prompt,
tested on the shared 300 blocks → `data/test_post/optimized_phq9_human/`).
The prompt-optimizer jobs are documented
in depth in `prompt_optimizer.md`.

## checks
Small "how good is X" scripts with their own jobs, outside `src/`; see [../checks/README.md](../checks/README.md).
| job | purpose | GPU/time |
|---|---|---|
| `checks/check_surface_cues.job` | regressor re-scored with emoji / punctuation stripped | gpu_mig, 30 min |

## notes
- Multi-model arm (2026-09): Gemma-4-31B-it and Mistral-Small-3.2-24B run in `.venv_vllm_g4`
  (`requirements_vllm_g4.txt`, vLLM 0.19.1 + transformers 5.5.4); their weights live in
  `/gpfs/work5/0/prjs1820/hf_cache` (`HF_HUB_CACHE`, set in the jobs). Aliases + per-model
  decoding in `src/utils/create_data/loaders.py` (`MODEL_ALIASES`, `STUDENT_DECODING`).
  A Kimi-Linear-48B-A3B arm was run
  and dropped on 2026-09-11 (posts did not follow the PHQ-9 conditioning); behaviour and archive
  location in `data/README.md` section 4a.
- No SLURM wrapper yet (GPU needed, run in an interactive GPU session):
  `run_minimal_shift`, `run_phq9_on_bert_testset`, `sa_run`, `sa_decoding_run`, `sa_prompt_run`.
- Renames (2026-08): `run_simulation.sh`→`run_simulation_sda.sh`, `run_simulation2.sh`→`run_simulation_sdc.sh`;
  jobs `run_data_simulation{,2}.job`→`run_simulation_{sda,sdc}.job`, `run_bias_data.job`→`run_bias_calibration.job`,
  `run_phq9_sa.job`→`sa_phq9_minimal_run.job`, `run_sa_prompt.job`→`sa_prompt_baseline_run.job`,
  `run_test_phq9.job`→`run_teacher_eval_iter0.job`. **Old `sbatch ~/run_X.job` paths are gone.**
- Archived: `bin/src/utils/scripts.sh` held one command — tweets-rerun-test of
  `optimized_tweets/Qwen3.5-27B_seed53/optimized_instruction_tweet.txt` (seed 42, 100 agents).
- Still in `$HOME`: `delete_out.sh` (cleans `*.out` in cwd — slurm logs now land at the repo
  root, so run it from there) and `useful_commands.sh` (snippet NOTES, **not executable** —
  contains `scancel -u $USER` and cache purges).
- The hand-run CLIs have no `.sh`/`.job` wrapper at all (assessment-diagnostics/mobility/velocity/
  network-target/entrainment plotting + `build_personas eval` + `sa_phq9`) — inventoried with
  run commands in [../src/README.md](../src/README.md) ("Hand-run CLIs").
