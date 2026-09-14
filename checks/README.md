# checks/ — small tests of how good something is

Each file here answers one question about a model, a prompt or a dataset. They are
not unit tests and there is no test runner. Run them from the repo root with
`PYTHONPATH=src`; the two that need a GPU have their own SLURM job next to them
(`sbatch checks/<name>.job`).

| File | Question it answers | Run | GPU | Writes |
|---|---|---|---|---|
| `check_surface_cues.py` | Does the PHQ-9 regressor read the words, or just emoji and punctuation? Re-scores a post CSV with emoji / punctuation stripped and compares MAE, bias and correlation. | `sbatch checks/check_surface_cues.job` or `PYTHONPATH=src ./.venv_vllm/bin/python checks/check_surface_cues.py --regressor <regressor.pt> --csv <posts.csv> --out-dir <dir>` | 1 (or `--device cpu`) | `<out-dir>/summary.csv` + per-variant CSVs |
| `check_multimodel_smoke.job` | Does a new generator model load, follow the prompt and parse cleanly? Generates 3 blocks per model, then runs the checker below. | `sbatch checks/check_multimodel_smoke.job` | 2×A100, ~15 min | `data/finetune/smoke/smoke_<model>.csv` |
| `check_multimodel_smoke.py` | Asserts on those smoke CSVs: row and agent counts, no leaked chat tokens, NO_POST rate, post length, Qwen decoding unchanged. | `PYTHONPATH=src ./.venv_vllm/bin/python checks/check_multimodel_smoke.py` | no | prints PASS/FAIL, exit 1 on FAIL |
| `check_cds_tracks_phq9.py` | Do cognitive-distortion n-grams get more common as PHQ-9 rises in the generated posts? Overall and per category. | `PYTHONPATH=src ./.venv_vllm/bin/python checks/check_cds_tracks_phq9.py [--out data/finetune/cds_by_phq9.csv]` | no | `plots/cds_validation.png` (used in the paper), optional CSVs |

The regressor scorer these use (`src/utils/eval_bert_on_csv.py`) and the CDS detector
(`src/utils/tools/cds.py`) stay in `src/` because pipeline code imports them too.
