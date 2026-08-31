# data/prompt_optimization_h/ — notes (cleanup 2026-08)

`_h` = human-in-the-loop prompt optimization (generate_posts_opt_h.py);
the automated TextGrad counterpart lives in `data/test_post/optimized_*`.

## qwen27_baseline/
- `iter_0/ … iter_10/` — the optimization trajectory. `iter_0` (minimal prompt)
  and `iter_10` (final/best prompt) are hot inputs everywhere (SA drivers,
  run_finetune.sh, run_bias_calibration.sh); `iter_1–9` = provenance.
- `inputs/` — local copies of the SA generation inputs (see `inputs/NOTES.md`).
- `prompt_sa/` — one CSV per prompt variant (sa_prompt_run.sh) + `prompt_cosine/`
  outputs of `sa_prompt.py`.
- `prompt_sa_reps/` — replicate design (sa_prompt_baseline_run.sh) +
  `prompt_cosine_reps/` outputs of `sa_analyze --prompt-reps`.
- `SA_prompt/` — `sa_prompt.py` default `--sa-dir`; its
  `prompt_Qwen_Qwen3.5-27B.csv` is also read by `run_finetune.sh:25`.

## Status flags
- The prompt-variant SA (`sa_prompt.py`, `prompt_sa/`, `prompt_sa_reps/` and
  their heatmaps) appears in NO manuscript — kept for possible SI use.
- Naming: `SA_prompt` vs `prompt_sa` vs `prompt_sa_reps` are three different
  things with near-identical names (flagged, deliberately not renamed —
  too many wired paths).
