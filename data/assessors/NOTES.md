# data/assessors/ — the PHQ-9 assessors, one directory per training corpus

Created 2026-09-21 out of the five `data/test_post/bert_regression*/` trees. The
old names flattened two independent axes into one string, so `..._gemma4` and
`..._gemma4_minimal` looked like two models when they are one model (Gemma) under
two generation prompts. Here the axes are separate and the directory carries a
`meta.json` saying what made it. Resolver + vocabulary:
[src/utils/assessors.py](../../src/utils/assessors.py).

## An arm

An **arm** is one training corpus, named `<generator>_<generation prompt>`. Every
arm is the MentalBERT+MLP regressor at 5 seeds (34-38).

| Arm | Generator | Generation prompt | Fine-tuned from | Role |
|---|---|---|---|---|
| `teacher` | Qwen3.5-27B | historical `prompts_post.json` | (none, trained from scratch) | The baseline. 12k-block teacher corpus. Paper Tables 1-2. |
| `qwen27_optimized` | Qwen3.5-27B | iter_10 (human-optimized) | `teacher` | **Deployed in the simulation**: seed 35 is `llama_activate.py`'s default regressor, and its bias tables live in that seed dir. |
| `gemma4_optimized` | Gemma-4-31B-it | iter_10 | `teacher` | Multi-model arm. |
| `qwen27_minimal` | Qwen3.5-27B | iter_0 (minimal) | `teacher` | Minimal-prompt arm, panel (b). Same generator as `qwen27_optimized`. |
| `gemma4_minimal` | Gemma-4-31B-it | iter_0 (minimal) | `teacher` | Minimal-prompt arm, panel (b). Same generator as `gemma4_optimized`. |
| `mistral_optimized` | Mistral-Small-3.2-24B | iter_10 | `teacher` | **Never generated.** Listed in `ARMS` so the figures skip it cleanly. |

`minimal` and `optimized` are PROMPTS, not models. That is the distinction the
old `_minimal` suffix hid.

## Layout

```
data/assessors/bert/plots/        figures that span arms (scripts/plotting/)
data/assessors/bert/<arm>/
    meta.json                  generator, prompt, train corpus, seeds, lr, epochs
    models/<model>_seed<NN>/   regressor.pt, performance.json, phq9_bias_table*.csv
    eval/on_<corpus>/          seed<NN>.csv + seed<NN>_summary.csv + aggregate.csv
    plots/                     figures about this arm alone
```

`teacher/` additionally holds `test_blocks_seed35.csv`, the materialized seed-35
test split (11,250 rows = 1,125 blocks) that the LLM assessor is re-scored on so
both estimators are judged on identical data.

## The estimator x corpus matrix

Eval directories reuse the arm vocabulary, so the table behind the paper's
multi-model result is one glob rather than three naming schemes:

```bash
# every assessor, scored on Gemma's held-out human-optimized posts
ls data/assessors/bert/*/eval/on_gemma4_optimized/aggregate.csv
```

| Assessor (row) | on `qwen27_optimized` | on `gemma4_optimized` | on `qwen27_minimal` | on `gemma4_minimal` |
|---|---|---|---|---|
| `teacher` (baseline) | x | x | x | x |
| `qwen27_optimized` (transfer) | own | x | x | x |
| `gemma4_optimized` | | own | | |
| `qwen27_minimal` | | | own | |
| `gemma4_minimal` | | | | own |

"own" is the diagonal: the arm scored on the corpus it was fine-tuned on, which
is why arm names and corpus names are the same vocabulary. `on_calibration/`
under `qwen27_optimized` is the extra column feeding the simulation's bias table.

## Old path -> new path

| Was | Is now |
|---|---|
| `test_post/bert_regression/` | `assessors/bert/teacher/` |
| `test_post/bert_regression_finetuned/` | `assessors/bert/qwen27_optimized/` |
| `test_post/bert_regression_finetuned_gemma4/` | `assessors/bert/gemma4_optimized/` |
| `test_post/bert_regression_finetuned_qwen_minimal/` | `assessors/bert/qwen27_minimal/` |
| `test_post/bert_regression_finetuned_gemma4_minimal/` | `assessors/bert/gemma4_minimal/` |
| `<tree>/<model>_seed<NN>/` | `<arm>/models/<model>_seed<NN>/` |
| `bert_regression/eval_baseline[_<tag>]/` | `teacher/eval/on_<arm>/` |
| `bert_regression_finetuned/eval_finetuned/` | `qwen27_optimized/eval/on_qwen27_optimized/` |
| `bert_regression_finetuned/eval_<tag>300/` | `qwen27_optimized/eval/on_<arm>/` |
| `bert_regression_finetuned/eval_calibration/` | `qwen27_optimized/eval/on_calibration/` |
| `bert_regression_finetuned_<tag>/eval_finetuned/` | `<arm>/eval/on_<arm>/` |

Data was moved, not copied: all 311 files verified byte-identical afterwards. No
compatibility symlinks were left, so a stale path fails loudly instead of
silently resolving.

## Adding an arm

`GEN_TAG` still takes the old tag spelling and resolves through
`utils.assessors.resolve` (`gemma4` -> `gemma4_optimized`, `qwen_minimal` ->
`qwen27_minimal`), so existing job scripts are unchanged:

```bash
GEN_TAG=gemma4 GEN_MODEL=gemma4-31b PYTHON_GEN=.venv_vllm_g4/bin/python \
    bash scripts/assessment/run_finetune.sh
```

For a genuinely new arm, add an entry to `ARMS` (and its held-out CSV to
`CORPORA`) in `src/utils/assessors.py`. `run_finetune.sh` then derives every
path from it and writes `meta.json` at the end of the run.

## Caveat

`meta.json` for the five existing arms was **backfilled** on 2026-09-21 from the
run logs and the post `.meta.json` sidecars (the field `backfilled` records
this). Only arms trained from now on get it written by the run itself.
