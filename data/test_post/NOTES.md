# data/test_post/ — notes (cleanup 2026-08)

Dual role (accidental name collision): (a) `TestPathManager` output base
(`data/test` + `FC.DIR_SUFFIX`), (b) home of the evaluation/model trees below.
~40 hardcoded literals point here — do not rename this tree.

> **Moved 2026-09-21.** The five `bert_regression*/` trees are no longer here.
> They now live under [`data/assessors/bert/<arm>/`](../assessors/NOTES.md), one
> directory per training corpus, with `models/`, `eval/on_<corpus>/` and a
> `meta.json`. See that file for the old-path -> new-path table.

| Subdir | What | Used by |
|---|---|---|
| `Qwen_Qwen3.5-27B/` | high-fidelity teacher dataset + implicit SA neighbour pool | CS + paper; see its own NOTES.md |
| `method_comparison/` | comparison figures + aggregate outputs (`run_eval_comparison.sh`) | paper Tables 1–2, CS appendix |
| `optimized_phq9/` | TextGrad PHQ-9 prompt runs (seeds 23,24,25,32,33) + eval_on_* + `Qwen3.5-27B_sensitivity/` | CS figs 4.7/4.8, paper tables |
| `optimized_phq9_human/` | TextGrad PHQ-9 prompt runs re-optimized on the human-optimized corpus (train `finetune/qwen/train_posts_qwen.csv`, test `finetune/qwen/test_posts_qwen.csv`, minimal start prompt; seeds 23,24,25,32,33; `run_meta.txt` per run) — the LLM counterpart of `assessors/bert/qwen27_optimized/` | paper assessment tables (planned replacement of the base-set rows); the LLM-assessor line of panel (b) of `method_comparison/multimodel/linearity_bias.png` (validation-best seed, currently 23) |
| `optimized_tweets/` | TextGrad post-generation prompt runs (name is a tweet-era leak — these are POST prompts) | CS appendix; prompts copied to `prompt_optimization_h/qwen27_baseline/inputs/` |
| `optimized_phq9/*/{minimal,eval_on}_<tag>300/` | Qwen assessor on that generator's 300-block set (`run_llm_assessor_on_heldout.sh`) | paper multi-model table |
| `method_comparison/multimodel/` | generator × estimator summary (`python -m utils.visualization multimodel`) | paper multi-model table |
