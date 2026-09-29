# empirical/

The synthetic-data analyses re-run on real users: tweets from the `timelines` table
joined to the Qualtrics PHQ-9 survey (`qualtrics_numeric`). Data, embeddings and
results live outside the repo, in `~/data/social_twitter/`.

| File | What it does |
|---|---|
| `embed.py` | embeds every tweet **once** with S-BERT and MentalBERT → `embeddings/{sbert,mentalbert}_tweets.npz`; skips an encoder whose file exists |
| `ladder.py` | severity ladder with each user matched to the nearest same-gender, closest-age user in the next PHQ-9 band, and the same matching within each band as the reference; cosine + bootstrap CI + random-partner baseline (S-BERT) → `ladder_matched.csv`, `ladder_matched_within.csv` |
| `score.py` | every MentalBERT+MLP assessor (arm × seed) on the users, from the saved MentalBERT vectors; same centroid and rounding as `bert-eval`, same output columns |
| `bias.py` | per-band signed bias and MAE per arm (mean ± SD over seeds) |
| `finetune_cv.py` | the teacher regressors fine-tuned on the users as `run_finetune.sh` fine-tuned them on synthetic posts, but 5-fold CV by user instead of a fixed test set; plus the same MLP trained from scratch |
| `cds.py` | the CDS check (`checks/check_cds_tracks_phq9.py`) on the users: % of tweets with CDS per PHQ-9 score and category × band, plus per-user Spearman of % CDS tweets vs PHQ-9, and the synthetic CDS figure with the empirical line added (`cds_vs_synthetic.png`); regex only, run anywhere with `PYTHONPATH=src:checks` |
| `run_empirical.job` | SLURM (Big Red 200, CPU only): embed → ladder → score → bias → fine-tune → bias |
| `figures.ipynb` | synthetic vs empirical figures in the style of `visualization.plot_model_linearity_bias` (a) and (b) plus the within-band reference, how far predictions track the true score, the fine-tuned bias against the synthetic arms, and depressed vs not (PHQ-9 ≥ 10: AUC, sensitivity, specificity, ROC) for the human-optimized arm; PNG + CSV to `results/figures/` |

One-time setup on a Big Red login node, from the repo root. The python module already
has pandas, scikit-learn, networkx, umap and matplotlib, so the venv on top of it only
adds torch and sentence-transformers; the last line downloads both
encoders so the job can run offline:

```bash
module load python
python3 -m venv --system-site-packages ~/venvs/empirical
source ~/venvs/empirical/bin/activate
pip install torch sentence-transformers seaborn
PYTHONPATH=src python -c "from utils.metrics import generate_sbert_model as g; g(mentalbert=False, device='cpu'); g(mentalbert=True, device='cpu'); print('ok')"
```

Run: `sbatch -A <allocation> --export=ALL,TAG=<pull> empirical/run_empirical.job` from the
repo root (CPU, `general` partition), then the notebook with the same `TAG`. Each pull
gets `~/data/social_twitter/<TAG>/{embeddings,results}`; a resubmit reuses the
embeddings, pass `--force` to `embed.py` to re-encode.

## Inputs (`~/data/social_twitter/`)

- `<TAG>_tweets_phq.csv`: `agent_id, phq9, tweet`, one row per tweet, rows grouped by user
- `<TAG>_users.csv`: `agent_id, phq9, age, gender`, one row per user

Built in a notebook from the MariaDB pull:
- **Survey side:** complete PHQ-9 only. Items are coded 1–4, so the total is the item sum − 9.
- **Tweet filters:** English, no retweets, deduplicated by tweet id.
- **Tweet timing:** from the tweet id (`created_at` in `timelines` is empty).
- **Tweet selection:** `260923`: the 10 most recent tweets in the 90 days before `RecordedDate`, users with fewer than 5 dropped (337 users). `260928`: no time window, 10 tweets drawn at random (seed 42) from users with at least 10. `260928w5`: as `260928`, but only tweets of at least 5 words (not counting `@user`) are eligible.
- **Cleaning:** URLs removed, `@handles` replaced with `@user`, HTML entities decoded.

