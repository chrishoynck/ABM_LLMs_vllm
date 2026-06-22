"""Reconstruct the exact held-out TEST blocks a BERT+MLP regressor was scored on.

The non-finetuned MentalBERT+MLP regressor (`train_BERT_model`) never stores its
test set as a file: the partition is implied by `(seed, the cached agent_ids,
agent-level 80/10/10 split)` — see `prompt_optimizer.split_embeddings_and_labels`.
This script replays that split deterministically (CPU-only, no model load, no
teacher) and writes the test blocks back out as a `tweets_with_phq9` CSV so the
LLM PHQ-9 prompt can be scored on the *same* blocks via:

    python -m utils.prompt_optimizer --mode phq9-rerun-test \
        --model Qwen/Qwen3.5-27B --seeds 23 \
        --instruction-filename minimal_instruction.txt \
        --posts-file <this CSV>

Each selected block is re-emitted as one row per tweet with a fresh, globally
unique agent_id and a constant phq9, so `parse_tweets_with_phq9_csv` re-groups it
into the identical block (the grouping key is consecutive (agent_id, phq9)). The
script asserts the rebuilt block count equals the regressor's recorded n_test.

Usage:
    python -m utils.tools.build_bert_testset --seed 35 \
        --out data/test_post/bert_regression/test_blocks_seed35.csv
"""

import argparse
import csv
import os

import numpy as np
import torch

# ---- minimal, self-contained copy of prompt_optimizer.parse_tweets_with_phq9_csv ----
# Inlined so this tool does not import prompt_optimizer (which pulls in vLLM/TextGrad).
# Must stay byte-for-byte equivalent to the original grouping logic.
def parse_tweets_with_phq9_csv(file_path: str):
    """(tweet_blocks, true_answers, personas, agent_ids), grouped by consecutive (agent_id, phq9)."""
    tweet_blocks, true_answers, personas, agent_ids = [], [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        current_agent = current_phq9 = current_persona = None
        current_tweets: list[str] = []
        for row in reader:
            agent_id = row.get("agent_id")
            try:
                phq9 = int(row.get("phq9")) if row.get("phq9") not in (None, "") else None
            except ValueError:
                continue
            persona = row.get("persona") if row.get("persona") not in (None, "") else None
            tweet = (row.get("tweet") or "").strip()
            if agent_id != current_agent or phq9 != current_phq9:
                if current_tweets and current_phq9 is not None and len(current_tweets) > 1:
                    tweet_blocks.append(current_tweets)
                    true_answers.append(current_phq9)
                    personas.append(current_persona)
                    agent_ids.append(current_agent)
                current_agent, current_phq9, current_persona = agent_id, phq9, persona
                current_tweets = [tweet] if tweet else []
            elif tweet:
                current_tweets.append(tweet)
        if current_tweets and current_phq9 is not None and len(current_tweets) > 1:
            tweet_blocks.append(current_tweets)
            true_answers.append(current_phq9)
            personas.append(current_persona)
            agent_ids.append(current_agent)
    return tweet_blocks, true_answers, personas, agent_ids


def test_agents_for_seed(agent_ids: list[str], seed: int,
                         train_frac: float = 0.8, val_frac: float = 0.1) -> set[str]:
    """Replay split_embeddings_and_labels' agent-level split and return the TEST agent keys."""
    rng = np.random.default_rng(seed)
    unique_agents = sorted(set(agent_ids))               # same ordering as the trainer
    n_agents = len(unique_agents)
    agent_perm = rng.permutation(n_agents)
    n_train_a = int(n_agents * train_frac)
    n_val_a = int(n_agents * val_frac)
    return {unique_agents[i] for i in agent_perm[n_train_a + n_val_a:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=35,
                    help="Regressor seed whose held-out test partition to reconstruct (default: 35, the deployed seed).")
    ap.add_argument("--cache", default="data/test/Qwen/Qwen3.5-27B/mentalbert_embeddings/embeddings_and_labels.pt",
                    help="Embeddings cache the regressor was trained from (provides agent_ids + the file::agent keys).")
    ap.add_argument("--out", default=None,
                    help="Output tweets_with_phq9 CSV. Default: data/test_post/bert_regression/test_blocks_seed<seed>.csv")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--expect-n", type=int, default=None,
                    help="Optional: assert the rebuilt block count equals this (the regressor's recorded n_test).")
    args = ap.parse_args()

    out = args.out or f"data/test_post/bert_regression/test_blocks_seed{args.seed}.csv"

    print(f"[testset] loading agent_ids from {args.cache}")
    cache = torch.load(args.cache, map_location="cpu")
    agent_ids = cache["agent_ids"]
    n_blocks = len(agent_ids)
    print(f"[testset] cache: {n_blocks} blocks, {len(set(agent_ids))} unique agents")

    test_agents = test_agents_for_seed(agent_ids, args.seed, args.train_frac, args.val_frac)
    n_expected = sum(1 for a in agent_ids if a in test_agents)
    print(f"[testset] seed {args.seed}: {len(test_agents)} test agents -> {n_expected} test blocks")

    # Group the test agent keys (file::agent_id) by their source file.
    files: dict[str, set[str]] = {}
    for key in test_agents:
        fp, aid = key.rsplit("::", 1)
        files.setdefault(fp, set()).add(aid)

    fields = ["agent_id", "persona", "age", "step", "phq9", "tweet", "interaction"]
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    gid = 0
    written_blocks = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for fp in sorted(files):
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"source file from cache key not found: {fp}")
            blocks, answers, personas, aids = parse_tweets_with_phq9_csv(fp)
            want = files[fp]
            for block, ans, persona, aid in zip(blocks, answers, personas, aids):
                if aid not in want:
                    continue
                for j, tweet in enumerate(block):
                    writer.writerow({
                        "agent_id": gid, "persona": persona or "", "age": "",
                        "step": j, "phq9": ans, "tweet": tweet, "interaction": "",
                    })
                gid += 1
                written_blocks += 1
    print(f"[testset] wrote {written_blocks} blocks ({gid} agents) -> {out}")

    # Verify: re-parse the output and confirm it reforms the same number of blocks.
    rb, _, _, _ = parse_tweets_with_phq9_csv(out)
    print(f"[testset] re-parse check: output yields {len(rb)} blocks "
          f"(expected {n_expected})")
    assert len(rb) == n_expected == written_blocks, (
        f"block-count mismatch: rebuilt={len(rb)} selected={n_expected} written={written_blocks}"
    )
    if args.expect_n is not None:
        assert written_blocks == args.expect_n, (
            f"got {written_blocks} blocks but regressor recorded n_test={args.expect_n}")
        print(f"[testset] matches regressor n_test={args.expect_n} ✓")
    print("[testset] OK — ready for --mode phq9-rerun-test --posts-file " + out)


if __name__ == "__main__":
    main()
