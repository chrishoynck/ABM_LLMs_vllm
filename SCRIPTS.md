# SCRIPTS.md — driver scripts + SLURM jobs (layout 2026-08)

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
| `run_sa_network.sh` | network-topology Sobol SA (CPU) | — | `data/confidential/phq9.sav` | `data/sensitivity/network_100_3/` |
| `sa_run.sh` | neighbour/agent/joint + PHQ-9 axes | — (GPU needed) | `data/sensitivity/inputs/` | `data/sensitivity/<axis>/` |
| `sa_decoding_run.sh` | temp/top_p axis | — (GPU needed) | same | `data/sensitivity/decoding/` |
| `sa_phq9_minimal_run.sh` | PHQ-9 bands, minimal prompt | `sa_phq9_minimal_run.job` (2×A100, 2h) | same | `data/sensitivity/phq9_minimal_prompt/` |
| `sa_prompt_run.sh` | prompt axis, single draw | — (GPU needed) | `qwen27_baseline/inputs/` | `.../prompt_sa/` |
| `sa_prompt_baseline_run.sh` | prompt axis w/ replicates | `sa_prompt_baseline_run.job` (2×A100, 3.5h; job adds embed + analyze steps) | same | `.../prompt_sa_reps/` |

## assessment
| script | purpose | job | inputs | outputs |
|---|---|---|---|---|
| `run_finetune.sh` | finetune BERT regressor + eval | — (GPU needed) | `data/finetune/` posts | `data/test_post/bert_regression_finetuned/` |
| `run_bias_calibration.sh` | 28-level PHQ-9 bias table | `run_bias_calibration.job` (2×A100, 6h) | unseen persona pool | `phq9_bias_table.csv` (note: sims load the notebook-exported `_fullfit` variant) |
| `run_phq9_on_bert_testset.sh` | prompts scored on BERT holdout | — (GPU needed) | embeddings cache (`data/test/Qwen/`) | `optimized_phq9/*/eval_on_*` |
| `run_minimal_shift.sh` | minimal vs optimized prompt under shift | — (GPU needed) | `data/finetune/test_posts.csv` | `minimal_*/` subdirs + fig2 |
| `run_eval_comparison.sh` | estimator-comparison figures (CPU) | — | eval CSVs on disk | `method_comparison/fig{1,2}` — the PNAS Tables 1–2 source |

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
`run_prompt_optimizer_phq9.job` (H100 1h).

## notes
- No SLURM wrapper yet (GPU needed, run in an interactive GPU session): `run_finetune`,
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
