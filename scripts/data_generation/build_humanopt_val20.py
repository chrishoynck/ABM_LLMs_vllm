"""Build the 20-persona validation set for the human-in-the-loop prompt run.

The human loop never had a validation split: iterations were judged on the
7-agent working batch and only iters 0/7/8/9/10 were ever scored on the
100-persona test set. This draws 20 personas from the same eval pool, disjoint
from BOTH, so every iteration prompt can be re-evaluated on one fixed held-out
set (matching the 20 blocks the TextGrad generation runs validated on).

Excluded rows of data/personas_eval_1000_phq9.csv:
  - the 100 test personas          (rng(999).choice(1000, 100))
  - each iteration's working batch (rng(iter_n).choice(1000, 7), n = 0..10)

    ./.venv_vllm/bin/python scripts/data_generation/build_humanopt_val20.py
"""
import numpy as np
import pandas as pd

POOL   = "data/personas_eval_1000_phq9.csv"
OUT    = "data/prompt_optimization_h/qwen27_baseline/val20_personas.csv"
N_VAL  = 20
SEED   = 20260922

df = pd.read_csv(POOL)

used = set(np.random.default_rng(999).choice(len(df), size=100, replace=False).tolist())
n_test = len(used)
for iter_n in range(11):
    used |= set(np.random.default_rng(iter_n).choice(len(df), size=7, replace=False).tolist())

eligible = np.array([i for i in range(len(df)) if i not in used])

# Stratified by PHQ-9 severity band with proportional allocation: a plain random
# draw of 20 leaves the band mix well off the pool's, and post quality is scored
# per band, so an unrepresentative mix would shift the whole validation curve.
BANDS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 27)]
phq9 = df["phq9"].to_numpy()
rng = np.random.default_rng(SEED)
shares = np.array([((phq9 >= lo) & (phq9 <= hi)).sum() / len(df) for lo, hi in BANDS])
# Largest-remainder allocation so the quotas sum to exactly N_VAL.
exact = shares * N_VAL
take = np.floor(exact).astype(int)
for b in np.argsort(-(exact - take))[:N_VAL - take.sum()]:
    take[b] += 1

pick = []
for (lo, hi), n, share in zip(BANDS, take, shares):
    in_band = eligible[(phq9[eligible] >= lo) & (phq9[eligible] <= hi)]
    print(f"  band {lo:>2}-{hi:<2}: pool share {share:.3f} -> {n} of {len(in_band)} eligible")
    pick += rng.choice(in_band, size=n, replace=False).tolist()
pick = sorted(pick)
assert len(pick) == N_VAL, f"allocation gave {len(pick)} personas, not {N_VAL}"

val = df.iloc[pick].reset_index(drop=True)
val.to_csv(OUT, index=False)

print(f"pool {len(df)}, excluded {len(used)} ({n_test} test + working batches), "
      f"eligible {len(eligible)}")
print(f"picked rows: {pick}")
print(f"PHQ-9: {sorted(val['phq9'].tolist())}")
print(f"PHQ-9 mean {val['phq9'].mean():.2f} vs pool {df['phq9'].mean():.2f}")
print(f"wrote {OUT}")
