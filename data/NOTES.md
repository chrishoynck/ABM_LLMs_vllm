# data/ — notes (cleanup 2026-08)

> Looking for **where a dataset comes from / what it is used for**? That lives
> in [README.md](README.md) (the provenance map). This file holds
> status/cleanup details.

## Prompt JSONs
| File | Status |
|---|---|
| `prompts_optimal.json` | LIVE (`FC.PROMPTS_FILE`, format_config.py:41). Used keys: `phq9.{system_instruction,System_format,user_template_persona,system_persona}`, `tweet_gen.{system_forced,user_template_forced}`. |
| `prompts_post.json` | HISTORICAL — generated the high-fidelity teacher set (see `test_post/Qwen_Qwen3.5-27B/NOTES.md`). Edited after generation (May 11); March bytes unrecoverable (data/ untracked). Also read by `experiment.ipynb`. |
| `prompts_post_minimal.json` | LIVE — tweet-optimizer seed (prompt_optimizer.py:3428), `run_minimal_shift.sh`, `run_phq9_on_bert_testset.sh`. |

Dead-key cleanup (2026-08-31): `tweet_gen.{system_standard,user_template}` and
`phq9.{system_user,user_template_user}` were removed together with their dead
code paths — `agent.py` now raises on non-forced tweet generation (legacy
TestLLMs interaction mode) and builds the PHQ-9 prompt persona-only.
`phq9.user_template_forced` stays (read by `prompt_optimizer._build_user_message`
when no persona is passed) but its dangling `{persona}` placeholder — a
guaranteed `KeyError` — was removed.

## Other
- `methodology_paper/` — belongs to a DIFFERENT paper. Frozen; do not touch
  (its internal `prompts.json` included).
- `grok_posts/` — `posts_with_phq9.*` = generator defaults (generate_posts_grok.py);
  `eval_bert_on_csv.py:14` docstring names a never-created
  `posts_eval_grok_aligned.csv`.
- Personas: `personas_eval_1000_phq9.csv` = SA/eval anchor set (hottest file);
  `personas_finetune_phq9.csv` = finetune; `personas_short_10k.csv` /
  `personas_10k.csv` = pools. `personas_eval_1000.csv` is referenced by some
  defaults but MISSING on disk (known dangling default).
- `confidential/` = HELIUS (phq9.sav + derived phq9_filtered.csv, written by
  load_personas.py:265).
