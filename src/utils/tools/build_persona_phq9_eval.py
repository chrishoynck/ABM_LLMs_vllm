"""Build the shared (persona, PHQ-9) eval file used to compare models on identical inputs.

The file is sampled ONCE: pulls personas from the eval pool (built via
load_or_build_persona_pool if missing) and assigns each one a PHQ-9 score
0..27 round-robin then shuffled with a fixed seed. Every model run that
consumes the file (utils.create_data.generate_*, generate_posts_grok.py) sees the same
persona → PHQ-9 pairing in the same order, so per-model BERT predictions are
directly comparable.

Usage:
    python -m utils.tools.build_persona_phq9_eval \\
        --persona-pool data/personas_eval_1000.csv \\
        --pool-size 1000 \\
        --phq9-seed 1000 \\
        --out data/personas_eval_1000_phq9.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

try:
    import utils.tools.load_personas as lp
except ImportError:
    from . import load_personas as lp


def build(persona_pool: str, pool_size: int, phq9_seed: int, out_path: str) -> None:
    """Build the (persona, phq9) eval file at `out_path`. No-op if it already exists."""
    if os.path.isfile(out_path):
        df = pd.read_csv(out_path)
        print(f"[build] {out_path} already exists ({len(df)} rows); leaving it alone.")
        return

    personas = lp.load_or_build_persona_pool(
        n_needed=pool_size, pool_path=persona_pool, pool_size=pool_size,
    )

    # Same assignment scheme create_data.generate_posts_grok uses for its default round-robin
    # (cycle 0..27 across blocks then shuffle), so coverage is balanced (~36/score for 1000).
    scores = np.array([i % 28 for i in range(pool_size)])
    np.random.default_rng(phq9_seed).shuffle(scores)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    pd.DataFrame({"persona": personas, "phq9": scores.astype(int)}).to_csv(out_path, index=False)

    counts = pd.Series(scores).value_counts().sort_index()
    print(f"[build] wrote {out_path} ({pool_size} rows; phq9 seed={phq9_seed})")
    print(f"[build] per-PHQ-9 counts: min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--persona-pool", default="data/personas_eval_1000.csv")
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--phq9-seed", type=int, default=1000)
    parser.add_argument("--out", default="data/personas_eval_1000_phq9.csv")
    args = parser.parse_args()
    build(args.persona_pool, args.pool_size, args.phq9_seed, args.out)


if __name__ == "__main__":
    main()
