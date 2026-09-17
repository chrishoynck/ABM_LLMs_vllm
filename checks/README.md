# checks/ — small tests of how good something is

Each file here answers one question about a model, a prompt or a dataset. They are
not unit tests and there is no test runner. Run them from the repo root with
`PYTHONPATH=src`; the two that need a GPU have their own SLURM job next to them
(`sbatch checks/<name>.job`).

| File | Question it answers | Run | GPU | Writes |
|---|---|---|---|---|
| `check_surface_cues.py` | Does the PHQ-9 regressor read the words, or just emoji and punctuation? Re-scores a post CSV with emoji / punctuation stripped and compares MAE, bias and correlation. | `sbatch checks/check_surface_cues.job` or `PYTHONPATH=src ./.venv_vllm/bin/python checks/check_surface_cues.py --regressor <regressor.pt> --csv <posts.csv> --out-dir <dir>` | 1 (or `--device cpu`) | `<out-dir>/summary.csv` + per-variant CSVs |
| `check_multimodel_smoke.job` | Does a new generator model load, follow the prompt and parse cleanly? Generates 3 blocks per model, then runs the checker below. | `sbatch checks/check_multimodel_smoke.job` | 2×A100, ~15 min | `data/finetune/smoke/smoke_<model>.csv` |
| `check_multimodel_smoke.py` | Asserts on those smoke CSVs: row and agent counts, no leaked chat tokens (Qwen, Gemma 4 and Mistral control tokens), NO_POST rate, post length, Qwen and Mistral decoding unchanged. | `PYTHONPATH=src ./.venv_vllm/bin/python checks/check_multimodel_smoke.py` | no | prints PASS/FAIL, exit 1 on FAIL |
| `check_grading_consistency.{py,job}` | Does `prompt_optimizer --mode grade-posts` reproduce the original teacher grades (iter_10, 100 agents x 3 posts, 7.41), how noisy is the grader, do 3-post and 10-post block grades agree, and how do Qwen and Gemma compare on the paper's 100-persona x 3-post protocol? `prepare` writes the 3-post truncations and the 100 personas, the job regrades (3 seeds), grades the truncations, generates + grades fresh 3-post blocks from both generators, `report` compares; `prompts` (CPU) tests human-optimized vs TextGrad vs minimal prompt scores on the shared 100 agents. | `sbatch checks/check_grading_consistency.job` (or `PYTHONPATH=src ./.venv_vllm/bin/python checks/check_grading_consistency.py {prepare,report}` around the grading calls) | 2×A100, ~1.5 h | `data/test_post/teacher_grades/{regrade_iter10_test100*,grades_<tag>300_first3,posts100x3_<tag>,grades_100x3_<tag>}.csv` |
| `check_phq9_optimizer_smoke.job` | Does the human-corpus TextGrad job (`jobs/run_prompt_optimizer_phq9_human.job`) load the model, read `data/finetune/{train,test}_posts.csv`, start from the minimal prompt and complete one step + test write? Same command with 1 step / 2 blocks. | `sbatch checks/check_phq9_optimizer_smoke.job` | 1×H100, ~10 min | `data/test_post/optimized_phq9_human_smoke/` (delete afterwards) |
| `check_cds_tracks_phq9.py` | Do cognitive-distortion n-grams get more common as PHQ-9 rises in the generated posts? Overall and per category. | `PYTHONPATH=src ./.venv_vllm/bin/python checks/check_cds_tracks_phq9.py [--out data/finetune/cds_by_phq9.csv]` | no | `plots/cds_validation.png` (used in the paper), optional CSVs |

The regressor scorer these use (`src/utils/eval_bert_on_csv.py`) and the CDS detector
(`src/utils/tools/cds.py`) stay in `src/` because pipeline code imports them too.
