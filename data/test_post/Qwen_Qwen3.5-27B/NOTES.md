# Qwen_Qwen3.5-27B — high-fidelity teacher dataset (provenance)

Generated ~2026-03-30 → 04-01 (repo at commit `84fa9c2`), teacher
Qwen3.5-27B with **thinking enabled** (`enable_thinking: True`).

- Prompts: `data/prompts_post.json` (the `FC.PROMPTS_FILE` of that era,
  `ABM_FORMAT=post`). The JSON was edited afterwards (May 11); the exact
  March bytes are unrecoverable (data/ is untracked).
- `data/prompts_post.json` was trimmed 2026-08-27 to the generation prompt only
  (`tweet_gen.*`). All 9 obsolete `phq9.*` assessment keys were removed —
  full original kept at `bin/data/prompts_post.json.orig-full`.
- PHQ-9 labels are PRESCRIBED, not assessed: each agent walks a random
  permutation of scores 0–27, `check_point=10` posts per score. The `phq9`
  column in `tweets_with_phq9.csv` is therefore ground truth (which is what
  makes this valid regressor training data).
- Layout: `temp_0.8_top_p_0.6_cp_10_{inter,no_inter}/seed_{55,65,75}/` with
  `checkpoint.json` (source of truth) + `tweets_with_phq9.csv` (re-exportable
  via `src/utils/tools/checkpoint_to_csv.py`; CSV mtimes postdate generation
  because of such re-exports).

## Generation mode: `_no_inter` vs `_inter` (confirmed 2026-08-27)
The two halves were generated in different modes — `force_active = not interaction`
in `step_llm_tweet`, i.e. "the agent must post":

- `_no_inter` (seed 75) — FORCED: every agent posts every round (`tweet_gen.system_forced`).
- `_inter` (seeds 55, 65) — NOT forced: `tweet_gen.system_standard`, which lets an
  agent choose `NO_POST` and asks for `@user_<ID>` mentions when interacting.

Verified in the CSVs:

| run | rows | NO_POST | `@user_` |
|---|---|---|---|
| seed_55 (`_inter`) | 10,000 | 176 | 8,218 |
| seed_65 (`_inter`) | 20,000 | 52 | 4,775 |
| seed_75 (`_no_inter`) | 90,000 | 0 | 0 |

This is a stylistic difference in the POSTS only — PHQ-9 labels are prescribed by
the permutation walk in both, so the set is sound as regressor training data.
Just don't describe the `_inter` half as "forced generation".
(In the later simulation/SA pipeline generation is always forced: network.py:291,312.)

## Downstream roles
- Training/eval source for the BERT regressors (`prompt_optimizer --mode bert`).
- Implicit neighbour pool for ALL SA/data generation
  (`DEFAULT_NEIGHBOR_ROOTS`, src/utils/create_data/loaders.py:80).
- Excluded from persona sampling via `_DEFAULT_EVAL_EXCLUDE_DIRS`
  (load_personas.py:147).
