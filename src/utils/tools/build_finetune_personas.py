"""Build the (persona, PHQ-9) file used to fine-tune the BERT regressor.

Draws ``--n`` personas from the 10k pool, EXCLUDING:
  * the 120 test personas (first 120 rows of the eval file), so train and test
    never overlap, and
  * every persona in the regressors' training corpus
    (data/test_post/Qwen_Qwen3.5-27B), so fine-tuning sees only fresh people.

PHQ-9 scores are assigned balanced (cycle 0..27 then shuffle), matching
build_persona_phq9_eval. The eval file is read-only — never modified.

Usage:
    python -m utils.tools.build_finetune_personas --n 2000 \\
        --out data/personas_finetune_phq9.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

CORPUS_DIR = "data/test_post/Qwen_Qwen3.5-27B"   # regressors' training corpus
EVAL_FILE = "data/personas_eval_1000_phq9.csv"   # first 120 rows = test set
TEST_N = 120


def _corpus_personas(corpus_dir: str) -> set[str]:
    """Every persona appearing in any tweets_with_phq9.csv under corpus_dir."""
    seen: set[str] = set()
    for p in glob.glob(os.path.join(corpus_dir, "**", "*.csv"), recursive=True):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "persona" in df.columns:
            seen |= set(df["persona"].astype(str))
    return seen


def build(n: int, out_path: str, source: str, phq9_seed: int, persona_seed: int) -> None:
    test_personas = set(pd.read_csv(EVAL_FILE)["persona"].astype(str).head(TEST_N))
    forbidden = test_personas | _corpus_personas(CORPUS_DIR)

    pool = pd.read_csv(source)
    pcol = "persona" if "persona" in pool.columns else pool.columns[0]
    candidates = [p for p in pool[pcol].astype(str).tolist() if p not in forbidden]
    if len(candidates) < n:
        raise SystemExit(f"only {len(candidates)} personas available after exclusions, "
                         f"but --n={n} requested.")

    rng = np.random.default_rng(persona_seed)
    personas = [candidates[i] for i in rng.permutation(len(candidates))[:n]]
    scores = np.array([i % 28 for i in range(n)])
    np.random.default_rng(phq9_seed).shuffle(scores)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    pd.DataFrame({"persona": personas, "phq9": scores.astype(int)}).to_csv(out_path, index=False)
    print(f"[finetune-personas] wrote {out_path} ({n} rows); "
          f"excluded {len(test_personas)} test + corpus personas.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=2000, help="Number of training personas.")
    parser.add_argument("--out", default="data/personas_finetune_phq9.csv")
    parser.add_argument("--source", default="data/personas_short_10k.csv")
    parser.add_argument("--phq9-seed", type=int, default=2000)
    parser.add_argument("--persona-seed", type=int, default=2000)
    args = parser.parse_args()
    build(args.n, args.out, args.source, args.phq9_seed, args.persona_seed)


if __name__ == "__main__":
    main()
