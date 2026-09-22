# checks/ — small tests of how good something is

Each file here answers one question about a model, a prompt or a dataset. They are
not unit tests and there is no test runner. Run them from the repo root with
`PYTHONPATH=src`; the ones that need a GPU have their own SLURM job next to them
(`sbatch checks/<name>.job`).

| File | Question it answers | Run | GPU | Writes |
|---|---|---|---|---|
| `check_surface_cues.py` | Does the PHQ-9 regressor read the words, or just emoji and punctuation? Re-scores a post CSV with emoji / punctuation stripped and compares MAE, bias and correlation. | `sbatch checks/check_surface_cues.job` or `PYTHONPATH=src ./.venv_vllm/bin/python checks/check_surface_cues.py --regressor <regressor.pt> --csv <posts.csv> --out-dir <dir>` | 1 (or `--device cpu`) | `<out-dir>/summary.csv` + per-variant CSVs |
| `check_decoding_alignment.py` | Does Gemma-4's operating point (1.0 / 0.975) behave like Qwen's (0.7 / 0.9)? Tabulates both per-agent-seeded decoding grids (within-setting S-BERT diversity with CI, band separability, distinct-bigrams, cross-agent overlap, length) and names the Gemma setting closest to Qwen's baseline. | `PYTHONPATH=src ./.venv_vllm/bin/python checks/check_decoding_alignment.py` (after `jobs/sa_seeded_analyze.job`) | CPU | `data/sensitivity/seeded/decoding_alignment.csv` |
| `check_grading_consistency.{py,job}` | Does `prompt_optimizer --mode grade-posts` reproduce the original teacher grades (iter_10, 100 agents x 3 posts, 7.41), how noisy is the grader, do 3-post and 10-post block grades agree, and how do Qwen and Gemma compare on the paper's 100-persona x 3-post protocol? `prepare` writes the 3-post truncations and the 100 personas, the job regrades (3 seeds), grades the truncations, generates + grades fresh 3-post blocks from both generators, `report` compares; `prompts` (CPU) tests human-optimized vs TextGrad vs minimal prompt scores on the shared 100 agents. | `sbatch checks/check_grading_consistency.job` (or `PYTHONPATH=src ./.venv_vllm/bin/python checks/check_grading_consistency.py {prepare,report}` around the grading calls) | 2×A100, ~1.5 h | `data/test_post/teacher_grades/{regrade_iter10_test100*,grades_<tag>300_first3,posts100x3_<tag>,grades_100x3_<tag>}.csv` |
| `check_cds_tracks_phq9.py` | Do cognitive-distortion n-grams get more common as PHQ-9 rises in the generated posts? Overall and per category, Qwen vs Gemma on the paired 3,000-block training corpora (`finetune/qwen/train_posts_qwen.csv`, `finetune/gemma4/train_posts_gemma4.csv`). | `PYTHONPATH=src ./.venv_vllm/bin/python checks/check_cds_tracks_phq9.py [--out data/finetune/cds_by_phq9.csv]` | no | `plots/cds_validation.png` (used in the paper), optional CSVs |
| `check_band_variance.py` | How much of the S-BERT variance is between PHQ-9 bands rather than between personas within a band? Block-level, bias-corrected eta² over the four 300-block test corpora (Qwen/Gemma human-optimized + minimal, Grok reference). | `PYTHONPATH=src ./.venv_vllm/bin/python checks/check_band_variance.py` | no | `data/test_post/method_comparison/multimodel/band_variance.csv`; caches under `data/finetune/band_variance_emb/` |
| `phq9_band_significance.py` | Per PHQ-9 class, does moving up one band change the text more than rerunning the same band does? Paired t-test on the per-agent-seeded SA embeddings. | `PYTHONPATH=src ./.venv_vllm/bin/python checks/phq9_band_significance.py [--tex <out.tex>]` | no | stdout, optional LaTeX table |

The regressor scorer these use (`src/utils/eval_bert_on_csv.py`) and the CDS detector
(`src/utils/tools/cds.py`) stay in `src/` because pipeline code imports them too.

## Retired checks

- `check_phq9_optimizer_smoke.job` — REMOVED 2026-09-22. Smoke test for
  `jobs/run_prompt_optimizer_phq9_human.job`, which has since finished all 5 seeds.
- `check_seeding.{py,job}` — REMOVED 2026-09-22, but **its verdict is the reason the
  SA reruns are seeded the way they are**, so it is recorded here:
  > Result 2026-09-17. Qwen (vLLM 0.17.1) is not reproducible under any seed
  > (kernel nondeterminism; ~60% identical at round 0), and seeding changes nothing
  > else — leave it unseeded. Gemma (vLLM 0.19.1) unseeded shares 42% of posts
  > between reps; a shared seed (`--seed S`, one sampling seed per round for all
  > agents) adds duplicates; a per-agent seed (`--seed S --per-agent-seed`,
  > `SeedSequence[S, agent id, round]` per request) is 100% reproducible,
  > independent across seeds, and as diverse as unseeded. Hence
  > `--seed <base> --per-agent-seed` in `jobs/sa_seeded.job`.
