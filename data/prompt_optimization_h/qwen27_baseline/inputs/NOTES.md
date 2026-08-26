# qwen27_baseline/inputs/ — local prompt-SA generation inputs (2026-08-26)

Copies (`cp -p`, mtimes preserved) so the prompt-variant SA reads one folder
instead of three trees. Originals unchanged.

| File | Copied from | Consumers |
|---|---|---|
| `personas_eval_1000_phq9.csv` | `data/personas_eval_1000_phq9.csv` | sa_prompt_run.sh, sa_prompt_baseline_run.sh |
| `prompt_iter_0.txt` | `../iter_0/prompt.txt` (minimal) | both drivers |
| `prompt_iter_10.txt` | `../iter_10/prompt.txt` | both drivers |
| `textgrad_seed{24,25,28,29,53}.txt` | `data/test_post/optimized_tweets/Qwen3.5-27B_seed<N>/best_instruction_tweet.txt` | both drivers |

NOT duplicated: the implicit neighbour pool
(`DEFAULT_NEIGHBOR_ROOTS` → `data/test_post/Qwen_Qwen3.5-27B/...`).
