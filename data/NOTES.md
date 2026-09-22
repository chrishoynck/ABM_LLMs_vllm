# data/ — notes (cleanup 2026-08)

> Looking for **where a dataset comes from / what it is used for**? That lives
> in [README.md](README.md) (the provenance map). This file holds
> status/cleanup details.

## Prompt JSONs
| File | Status |
|---|---|
| `prompts_optimal.json` | LIVE (`FC.PROMPTS_FILE`, format_config.py:41). Used keys: `phq9.{system_instruction,System_format,user_template_persona,system_persona}`, `tweet_gen.{system_forced,user_template_forced}`. |
| `prompts_post.json` | HISTORICAL — generated the high-fidelity teacher set (see `test_post/Qwen_Qwen3.5-27B/NOTES.md`). Edited after generation (May 11); March bytes unrecoverable (data/ untracked). Also read by `experiment.ipynb`. |
| `prompts_post_minimal.json` | LIVE — tweet-optimizer seed (prompt_optimizer.py:3428), `run_minimal_shift.sh`, `run_phq9_on_bert_testset.sh`. |

Dead-key cleanup (2026-08-31): `tweet_gen.{system_standard,user_template}` and
`phq9.{system_user,user_template_user}` were removed together with their dead
code paths — `agent.py` now raises on non-forced tweet generation (legacy
TestLLMs interaction mode) and builds the PHQ-9 prompt persona-only.
`phq9.user_template_forced` stays (read by `prompt_optimizer._build_user_message`
when no persona is passed) but its dangling `{persona}` placeholder — a
guaranteed `KeyError` — was removed.

## Multi-model arm (2026-09)
- `finetune/<tag>/`, `assessors/bert/<arm>/`,
  `assessors/bert/teacher/eval/on_<arm>/`,
  `assessors/bert/qwen27_optimized/eval/on_<arm>/`,
  `test_post/optimized_phq9/*/{minimal,eval_on}_<tag>300/`,
  `test_post/method_comparison/multimodel/`: LIVE, tags `gemma4` and `mistral`
  (the `kimi` tag was dropped 2026-09-11; outputs archived, see README.md section 4a).
  Layout + what is held fixed: [README.md §4a](README.md). Qwen's own paths
  (`finetune/qwen/`, `assessors/bert/qwen27_optimized/`) are untouched by these runs.
- `finetune/personas_test_300.csv` — LIVE, the shared 300 test personas
  (identical to the 120+180 behind `finetune/qwen/test_posts_qwen.csv`).

## finetune/ layout (2026-09-22)
Qwen's human-optimized corpus moved out of the `finetune/` root into its own
arm folder, so every arm is now `finetune/<tag>/<file>_<tag>.csv`:

| Was | Is |
|---|---|
| `finetune/train_posts.csv` | `finetune/qwen/train_posts_qwen.csv` |
| `finetune/test_posts.csv` | `finetune/qwen/test_posts_qwen.csv` |
| `finetune/test_posts_extra.csv` | `finetune/qwen/test_posts_extra_qwen.csv` |
| `finetune/calibration_posts.csv` | `finetune/qwen/calibration_posts_qwen.csv` |

Contents are byte-identical (`git mv`), so the `run_meta.txt` / `eval_meta.txt`
provenance files under `test_post/optimized_phq9*/` still name the old paths.
`run_finetune.sh` takes `GEN_TAG=qwen` as a synonym for the unset default.
Deleted in the same pass: `finetune/smoke/` with `checks/check_multimodel_smoke.{py,job}`,
`checks/check_minimal_arm_smoke.py` and `jobs/smoke_finetune_minimal.job`.

## Other
- `methodology_paper/` — belongs to a DIFFERENT paper. Frozen; do not touch
  (its internal `prompts.json` included).
- `grok_posts/` — `posts_with_phq9.csv` = generator default (generate_posts_grok.py,
  binned 2026-09), read by `checks/check_band_variance.py`. The `.txt` sibling was
  REMOVED 2026-09-22 (unread duplicate). `eval_bert_on_csv.py:14` docstring names a
  never-created `posts_eval_grok_aligned.csv`.
- Personas: `personas_eval_1000_phq9.csv` = SA/eval anchor set (hottest file);
  `personas_finetune_phq9.csv` = finetune; `personas_short_10k.csv` = pool.
  `personas_10k.csv` REMOVED 2026-09-22 (unread by any code; its builder
  `parse_persons` was already binned, so restore from git if the long-form schema
  is needed again). `personas_eval_1000.csv` is referenced by some defaults but
  MISSING on disk (known dangling default).
- `confidential/` = HELIUS (phq9.sav + derived phq9_filtered.csv, written by
  load_personas.py:265).

## Cleanup 2026-09-22 (plots, orphan data, dead jobs)
Figures the manuscripts never used, plus the data and code that only fed them.
No `\includegraphics` in either thesis or the paper points into `data/`, so
nothing here can break a LaTeX build. Per-figure detail: `sensitivity/NOTES.md`.
- Removed plots: `test_post/method_comparison/{fig1_bert_finetune,
  fig2_bert_vs_prompt_robustness,sbert_cosine_conditioning_seed35}.png`,
  `.../multimodel/{multimodel_mae_bias,mae_bias_per_band_finetuned}.png`, and the
  `plots_sbert` set listed in `sensitivity/NOTES.md`.
- Their producers were dropped from `utils/visualization.py`,
  `utils/sensitivity/sa_analyze.py` and `utils/tools/plot_assessment_diagnostics.py`.
  Three of them kept their computation because a CSV is still consumed:
  `decoding_settings_summary.csv` + `decoding_phq9_separability.csv`
  (`checks/check_decoding_alignment.py`) and `sbert_cosine_conditioning_seed35.csv`
  (panel (c) of `confusion_depression_classes.png`, and the PNAS `figure_scripts/`).
  `run_eval_comparison.sh` kept too: it still prints the Tables 1-2 numbers.
- Removed orphan data: `personas_10k.csv`, `grok_posts/posts_with_phq9.txt`,
  `sensitivity/plots_sbert/{axes_comparison_fixed.png,phq9_within_cross_by_band.csv}`,
  `prompt_optimization_h/qwen27_baseline/SA_prompt/scores.csv`,
  `test_post/optimized_tweets/SA_prompt/` (whole dir).
- Removed jobs/checks: `jobs/run_{finetune,llm_assessor}_mistral.job` (that arm was
  never generated - see `assessors/NOTES.md`), `checks/check_phq9_optimizer_smoke.job`
  (its target job has finished all 5 seeds), `checks/check_seeding.{py,job}` (verdict
  preserved in `checks/README.md`, "Retired checks").
- Also cleared: 31 root `slurm_output_*.out` (123 MB) and `logs/` (83 MB), both
  gitignored run logs.
