# Qwen_Qwen3.5-27B — high-fidelity teacher dataset (provenance)

Generated ~2026-03-30 → 04-01 (repo at commit `84fa9c2`), teacher
Qwen3.5-27B with **thinking enabled** (`enable_thinking: True`).

- Prompts: `data/prompts_post.json` (the `FC.PROMPTS_FILE` of that era,
  `ABM_FORMAT=post`). The JSON was edited afterwards (May 11); the exact
  March bytes are unrecoverable (data/ is untracked).
- PHQ-9 labels are PRESCRIBED, not assessed: each agent walks a random
  permutation of scores 0–27, `check_point=10` posts per score. The `phq9`
  column in `tweets_with_phq9.csv` is therefore ground truth (which is what
  makes this valid regressor training data).
- Layout: `temp_0.8_top_p_0.6_cp_10_{inter,no_inter}/seed_{55,65,75}/` with
  `checkpoint.json` (source of truth) + `tweets_with_phq9.csv` (re-exportable
  via `src/utils/tools/checkpoint_to_csv.py`; CSV mtimes postdate generation
  because of such re-exports).

## Forced-flag note (recorded, not adjudicated)
The operator states the forced flag was true for ALL generation runs,
including `_inter`. The repo code at `84fa9c2` sets
`force_active = not interaction`, which would make the `_inter` dirs
(seeds 55, 65) non-forced (system_standard branch, NO_POST allowed,
@user mentions). These two statements conflict; double-check before
describing the `_inter` half in a manuscript.

## Downstream roles
- Training/eval source for the BERT regressors (`prompt_optimizer --mode bert`).
- Implicit neighbour pool for ALL SA/data generation
  (`DEFAULT_NEIGHBOR_ROOTS`, src/utils/create_data/tools.py:81).
- Excluded from persona sampling via `_DEFAULT_EVAL_EXCLUDE_DIRS`
  (load_personas.py:147).
