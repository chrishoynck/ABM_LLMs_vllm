# data/test_post/ — notes (cleanup 2026-08)

Dual role (accidental name collision): (a) `TestPathManager` output base
(`data/test` + `FC.DIR_SUFFIX`), (b) home of the evaluation/model trees below.
~40 hardcoded literals point here — do not rename this tree.

| Subdir | What | Used by |
|---|---|---|
| `Qwen_Qwen3.5-27B/` | high-fidelity teacher dataset + implicit SA neighbour pool | CS + paper; see its own NOTES.md |
| `bert_regression/` | non-finetuned MentalBERT+MLP regressors (seeds 34–38), eval_baseline, test blocks | CS figs, paper Tables 1–2 |
| `bert_regression_finetuned/` | fine-tuned regressors; seed35 = `DEFAULT_REGRESSOR` (llama_activate.py:213, sa_phq9.py:80) + bias tables for the simulation | simulation, GABM appendix |
| `method_comparison/` | comparison figures + aggregate outputs (`run_eval_comparison.sh`) | paper Tables 1–2, CS appendix |
| `optimized_phq9/` | TextGrad PHQ-9 prompt runs (seeds 23,24,25,32,33) + eval_on_* + `Qwen3.5-27B_sensitivity/` | CS figs 4.7/4.8, paper tables |
| `optimized_tweets/` | TextGrad post-generation prompt runs (name is a tweet-era leak — these are POST prompts) | CS appendix; prompts copied to `prompt_optimization_h/qwen27_baseline/inputs/` |

Binned (→ `bin/data/test_post/`): `bert_regression_scratch/` (abandoned
from-scratch ablation, zero refs), `bert_regression_finetuned/method_comparison/`
(stale duplicate of the top-level `method_comparison/`).
