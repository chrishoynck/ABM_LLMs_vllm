# empirical/

The synthetic-data analyses re-run on real users: tweets from the `timelines` table
joined to the Qualtrics PHQ-9 survey (`qualtrics_numeric`). Data, embeddings and
results live outside the repo, in `~/data/social_twitter/`.

| File | What it does |
|---|---|
| `embed.py` | embeds every tweet **once** with S-BERT and MentalBERT → `embeddings/{sbert,mentalbert}_tweets.npz`; skips an encoder whose file exists |
| `ladder.py` | severity ladder with each user matched to the nearest same-gender, closest-age user in the next PHQ-9 band; adjacent-band cosine + bootstrap CI + random-partner baseline (S-BERT) |
| `score.py` | every MentalBERT+MLP assessor (arm × seed) on the users, from the saved MentalBERT vectors; same centroid and rounding as `bert-eval`, same output columns |
| `bias.py` | per-band signed bias and MAE per arm (mean ± SD over seeds) |
| `run_empirical.job` | SLURM (Big Red 200, CPU only): embed → ladder → score → bias |
| `figures.ipynb` | synthetic vs empirical figures in the style of `visualization.plot_model_linearity_bias` (a) and (b); PNG + CSV of the plotted numbers to `results/figures/` |

Run: set `-A <allocation>` in the job, then `sbatch empirical/run_empirical.job` from the
repo root, then the notebook. A resubmit reuses the embeddings; pass `--force` to
`embed.py` to re-encode.

If compute nodes can't reach Hugging Face, fetch both encoders once on a login node:

```bash
PYTHONPATH=src python -c "from utils.metrics import generate_sbert_model as g; g(mentalbert=False, device='cpu'); g(mentalbert=True, device='cpu')"
```

and uncomment `HF_HUB_OFFLINE=1` in the job.

## Inputs (`~/data/social_twitter/`)

- `260923_tweets_phq.csv`: `agent_id, phq9, tweet`, one row per tweet, rows grouped by user
- `260923_users.csv`: `agent_id, phq9, age, gender`, one row per user

Built in a notebook from the MariaDB pull:
- **Survey side:** complete PHQ-9 only. Items are coded 1–4, so the total is the item sum − 9.
- **Tweet filters:** English, no retweets, deduplicated by tweet id.
- **Tweet timing:** from the tweet id (`created_at` in `timelines` is empty).
- **Tweet selection:** the 10 most recent tweets in the 90 days before `RecordedDate`; users with fewer than 5 are dropped.
- **Cleaning:** URLs removed, `@handles` replaced with `@user`, HTML entities decoded.

The 2026-09-23 pull has 337 users: Minimal 120, Mild 91, Moderate 61, Mod. Severe 40, Severe 25.
