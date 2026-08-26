# data/sensitivity/inputs/ — local SA generation inputs (2026-08-26)

Copies (`cp -p`, mtimes preserved) so the axis SAs are self-contained.
Originals unchanged in their homes (they have non-SA consumers).

| File | Copied from | Consumers |
|---|---|---|
| `personas_eval_1000_phq9.csv` | `data/personas_eval_1000_phq9.csv` | sa_run.sh, sa_decoding_run.sh, sa_phq9_minimal_run.sh |
| `prompt_iter_10.txt` | `data/prompt_optimization_h/qwen27_baseline/iter_10/prompt.txt` | sa_run.sh, sa_decoding_run.sh |
| `prompt_iter_0.txt` | `.../iter_0/prompt.txt` (minimal) | sa_phq9_minimal_run.sh |

NOT duplicated (documented deps): the implicit neighbour pool
(`DEFAULT_NEIGHBOR_ROOTS` → `data/test_post/Qwen_Qwen3.5-27B/...`) and the
`sa_phq9.py` regressor (`data/test_post/bert_regression_finetuned/
Qwen3.5-27B_seed35/regressor.pt`).
