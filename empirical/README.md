# empirical/

The synthetic-data analyses re-run on real users: tweets from the `timelines` table
joined to the Qualtrics PHQ-9 survey (`qualtrics_numeric`). All data and results stay
under `data/confidential/empirical/` (gitignored).

| File | What it does |
|---|---|
| `embed.py` | S-BERT (or `--mentalbert`) embedding of every tweet → `.npz` |
| `ladder.py` | severity ladder with each user matched to the nearest same-gender, closest-age user in the next PHQ-9 band; adjacent-band cosine + bootstrap CI + random-partner baseline |
| `bias.py` | per-band signed bias and MAE of every MentalBERT+MLP assessor arm (mean ± SD over seeds) |
| `run_empirical.job` | SLURM: embed → ladder → `bert-eval` for every arm × seed → bias table |
| `figures.ipynb` | synthetic vs empirical figures in the style of `visualization.plot_model_linearity_bias` (a) and (b): the adjacent-band cosine ladder and bias per band for each assessor arm; PNG + CSV of the plotted numbers to `results/figures/` |

Run: `sbatch empirical/run_empirical.job` from the repo root, then the notebook (cluster venv).

## Inputs (`data/confidential/empirical/`)

- `260923_tweets_phq.csv`: `agent_id, phq9, tweet`, one row per tweet, rows grouped by user
- `260923_users.csv`: `agent_id, phq9, age, gender`, one row per user

Built in a notebook from the MariaDB pull:
- **Survey side:** complete PHQ-9 only. Items are coded 1–4, so the total is the item sum − 9.
- **Tweet filters:** English, no retweets, deduplicated by tweet id.
- **Tweet timing:** from the tweet id (`created_at` in `timelines` is empty).
- **Tweet selection:** the 10 most recent tweets in the 90 days before `RecordedDate`; users with fewer than 5 are dropped.
- **Cleaning:** URLs removed, `@handles` replaced with `@user`, HTML entities decoded.

The 2026-09-23 pull has 337 users: Minimal 120, Mild 91, Moderate 61, Mod. Severe 40, Severe 25.
