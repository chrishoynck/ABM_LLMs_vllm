# data/ — notes (cleanup 2026-08)

## Prompt JSONs
| File | Status |
|---|---|
| `prompts_optimal.json` | LIVE (`FC.PROMPTS_FILE`, format_config.py:41). Used keys: `phq9.{system_instruction,System_format,user_template_persona,system_persona}`, `tweet_gen.{system_forced,user_template_forced}`. |
| `prompts_post.json` | HISTORICAL — generated the high-fidelity teacher set (see `test_post/Qwen_Qwen3.5-27B/NOTES.md`). Edited after generation (May 11); March bytes unrecoverable (data/ untracked). Also read by `experiment.ipynb`. |
| `prompts_post_minimal.json` | LIVE — tweet-optimizer seed (prompt_optimizer.py:3428), `run_minimal_shift.sh`, `run_phq9_on_bert_testset.sh`. |
| `prompts.json`, `prompts_post1.json`, `prompts_post_no_thinking.json` | binned → `bin/data/` (zero references). |

Dead/broken keys still inside `prompts_optimal.json` (left in place — removing
them requires code edits, later stage):
- `tweet_gen.system_standard`, `tweet_gen.user_template` — byte-dups of the
  `_forced` variants; the non-forced branch never fires (network.py:291,312
  always pass `force_active=True`). `agent.py:282` reads `system_standard`
  unconditionally, so the key cannot simply be deleted.
- `phq9.system_user` — unreachable (persona always set), but read
  unconditionally at `agent.py:246`.
- `phq9.user_template_user` — unreachable; the only template that would inject
  the previous PHQ-9 into an assessment.
- `phq9.user_template_forced` — BROKEN: contains `{persona}` but neither caller
  supplies it → `KeyError` if ever reached (regression vs prompts_post.json).

## Other
- `methodology_paper/` — belongs to a DIFFERENT paper. Frozen; do not touch
  (its internal `prompts.json` included).
- `grok_posts/` — `posts_with_phq9.*` = generator defaults (generate_posts_grok.py);
  `posts_eval_grok.csv` binned; `eval_bert_on_csv.py:14` docstring names a
  never-created `posts_eval_grok_aligned.csv`.
- Personas: `personas_eval_1000_phq9.csv` = SA/eval anchor set (hottest file);
  `personas_finetune_phq9.csv` = finetune; `personas_short_10k.csv` /
  `personas_10k.csv` = pools. `personas_eval_1000.csv` is referenced by some
  defaults but MISSING on disk (known dangling default).
- `depressed.csv` binned (no references). `confidential/` = HELIUS (phq9.sav +
  derived phq9_filtered.csv, written by load_personas.py:265).
