"""Build the (persona, PHQ-9) CSV files that generation and eval runs read.

Three subcommands, each writing one two-column CSV (persona, phq9):
  eval        the shared 1000-persona eval file; built once, no-op if it exists
  finetune    training personas for the regressor fine-tune, disjoint from the test set
  test-extra  extra test personas taken from the eval file, disjoint from training
Run: see src/README.md (Hand-run CLIs); `scripts/assessment/run_finetune.sh` calls
the last two.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

EVAL_FILE = "data/personas_eval_1000_phq9.csv"   # first TEST_N rows = the BERT test set
TRAIN_FILE = "data/personas_finetune_phq9.csv"
CORPUS_DIR = "data/test_post/Qwen_Qwen3.5-27B"   # the regressors' training corpus
TEST_N = 120


def _balanced_phq9(n: int, seed: int) -> np.ndarray:
    """Return n PHQ-9 scores: 0..27 round-robin, then shuffled with `seed`."""
    scores = np.array([i % 28 for i in range(n)])
    np.random.default_rng(seed).shuffle(scores)
    return scores.astype(int)


def _write_csv(personas, scores, out_path: str) -> None:
    """Write a (persona, phq9) CSV, creating the folder if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    pd.DataFrame({"persona": personas, "phq9": scores}).to_csv(out_path, index=False)


def build_eval(persona_pool: str, pool_size: int, phq9_seed: int, out_path: str) -> None:
    """Build the shared (persona, PHQ-9) eval file. No-op if it already exists.

    Every model and prompt evaluation reads this same file, so all comparisons
    use the same persona-to-PHQ-9 pairs in the same order.

    Args:
        persona_pool (str): persona-only pool CSV; built from PersonaHub if missing.
        pool_size (int): number of personas to keep.
        phq9_seed (int): seed for the PHQ-9 shuffle.
        out_path (str): where to write the CSV.
    """
    if os.path.isfile(out_path):
        df = pd.read_csv(out_path)
        print(f"[build] {out_path} already exists ({len(df)} rows); leaving it alone.")
        return
    import utils.tools.load_personas as lp  # slow import (datasets, langdetect); only needed here

    personas = lp.load_or_build_persona_pool(
        n_needed=pool_size, pool_path=persona_pool, pool_size=pool_size,
    )
    scores = _balanced_phq9(pool_size, phq9_seed)
    _write_csv(personas, scores, out_path)
    counts = pd.Series(scores).value_counts().sort_index()
    print(f"[build] wrote {out_path} ({pool_size} rows; phq9 seed={phq9_seed})")
    print(f"[build] per-PHQ-9 counts: min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")


def _corpus_personas(corpus_dir: str) -> set[str]:
    """Every persona that appears in any CSV under `corpus_dir`."""
    seen: set[str] = set()
    for p in glob.glob(os.path.join(corpus_dir, "**", "*.csv"), recursive=True):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "persona" in df.columns:
            seen |= set(df["persona"].astype(str))
    return seen


def build_finetune(n: int, out_path: str, source: str, phq9_seed: int, persona_seed: int) -> None:
    """Draw n training personas for the regressor fine-tune.

    Excludes the first TEST_N rows of the eval file (the test set) and every
    persona in the regressors' training corpus, so fine-tuning only sees new people.

    Args:
        n (int): number of training personas.
        out_path (str): where to write the CSV.
        source (str): the 10k persona pool CSV.
        phq9_seed (int): seed for the PHQ-9 shuffle.
        persona_seed (int): seed for the persona draw.
    """
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
    _write_csv(personas, _balanced_phq9(n, phq9_seed), out_path)
    print(f"[finetune-personas] wrote {out_path} ({n} rows); "
          f"excluded {len(test_personas)} test + corpus personas.")


def build_test_extra(n: int, keep: int, out_path: str) -> None:
    """Pick the extra test personas needed to grow the test set from `keep` to `n` blocks.

    Takes eval-file rows after the first `keep` (those already have posts), keeps
    their PHQ-9 labels, and skips any persona in the fine-tune training file.

    Args:
        n (int): total test blocks wanted.
        keep (int): existing test blocks that stay as they are.
        out_path (str): where to write the CSV.
    """
    eval_df = pd.read_csv(EVAL_FILE)
    train_personas = set(pd.read_csv(TRAIN_FILE)["persona"].astype(str))

    n_extra = n - keep
    if n_extra <= 0:
        raise SystemExit(f"--n ({n}) must exceed --keep ({keep}).")

    rest = eval_df.iloc[keep:]
    clean = rest[~rest["persona"].astype(str).isin(train_personas)]
    if len(clean) < n_extra:
        raise SystemExit(f"only {len(clean)} eval personas after row {keep} are not in "
                         f"training, but {n_extra} needed for an {n}-block test set.")

    extra = clean.head(n_extra)
    _write_csv(extra["persona"].tolist(), extra["phq9"].to_numpy(), out_path)
    print(f"[test-personas] wrote {out_path} ({len(extra)} rows); "
          f"existing {keep} + these {len(extra)} = {n}-block test set, all disjoint from training.")


def main(argv=None) -> None:
    """Parse the subcommand and build the requested CSV."""
    p = argparse.ArgumentParser(description="Build (persona, PHQ-9) CSV files.")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="shared 1000-persona eval file; no-op if it exists")
    e.add_argument("--persona-pool", default="data/personas_eval_1000.csv")
    e.add_argument("--pool-size", type=int, default=1000)
    e.add_argument("--phq9-seed", type=int, default=1000)
    e.add_argument("--out", default=EVAL_FILE)

    f = sub.add_parser("finetune", help="training personas for the regressor fine-tune")
    f.add_argument("--n", type=int, default=2000, help="Number of training personas.")
    f.add_argument("--out", default=TRAIN_FILE)
    f.add_argument("--source", default="data/personas_short_10k.csv")
    f.add_argument("--phq9-seed", type=int, default=2000)
    f.add_argument("--persona-seed", type=int, default=2000)

    t = sub.add_parser("test-extra", help="extra test personas from the eval file, disjoint from training")
    t.add_argument("--n", type=int, default=300, help="Total test blocks wanted.")
    t.add_argument("--keep", type=int, default=120,
                   help="Existing test blocks already generated (kept as-is).")
    t.add_argument("--out", default="data/finetune/personas_test_extra.csv")

    a = p.parse_args(argv)
    if a.cmd == "eval":
        build_eval(a.persona_pool, a.pool_size, a.phq9_seed, a.out)
    elif a.cmd == "finetune":
        build_finetune(a.n, a.out, a.source, a.phq9_seed, a.persona_seed)
    else:
        build_test_extra(a.n, a.keep, a.out)


if __name__ == "__main__":
    main()
