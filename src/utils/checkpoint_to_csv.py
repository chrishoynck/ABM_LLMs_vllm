"""
Convert TestLLMs checkpoint.json files into tweets_with_phq9.csv.

A checkpoint written by ``write_out_tester`` already contains every agent's
full tweet history and PHQ-9 sum-score series, so the CSV can be built straight
from the JSON - no LLM/tokenizer load and no TestLLMs reconstruction needed.

The CSV columns mirror TestLLMs.export_tweets_with_phq9_txt:
    agent_id, persona, age, step, phq9, tweet, interaction

Usage
-----
    python src/utils/checkpoint_to_csv.py            # default: seeds 75 & 83, no_inter
    python src/utils/checkpoint_to_csv.py path/to/checkpoint.json [more.json ...]
"""

import csv
import json
import os
import sys

# Default run directory for the no-interaction Qwen runs.
DEFAULT_RUN_DIR = (
    "data/test_post/Qwen_Qwen3.5-27B/temp_0.8_top_p_0.6_cp_10_no_inter"
)
DEFAULT_SEEDS = (75, 83)


def checkpoint_to_csv(checkpoint_path: str) -> str:
    """
    Read one checkpoint.json and write tweets_with_phq9.csv next to it.

    Returns the path of the CSV that was written.
    """
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    interaction = ckpt.get("interaction", False)
    csv_path = os.path.join(
        os.path.dirname(checkpoint_path), "tweets_with_phq9.csv"
    )

    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            ["agent_id", "persona", "age", "step", "phq9", "tweet", "interaction"]
        )

        for agent in ckpt["agents"]:
            phq_series = agent.get("all_phq9_sumscores", [])
            tweets = agent.get("tweethistory", [])

            for idx, value in enumerate(phq_series):
                tweet = tweets[idx] if idx < len(tweets) else ""
                tweet = tweet.replace("\n", " ").replace("\r", " ").strip()
                writer.writerow(
                    [
                        agent["id"],
                        agent.get("persona"),
                        agent.get("age"),  # not stored in checkpoints -> None
                        idx,
                        value,
                        tweet,
                        interaction,
                    ]
                )

    print(f"Wrote {csv_path}  ({len(ckpt['agents'])} agents)")
    return csv_path


def main(argv: list) -> None:
    if argv:
        checkpoints = argv
    else:
        checkpoints = [
            os.path.join(DEFAULT_RUN_DIR, f"seed_{s}", "checkpoint.json")
            for s in DEFAULT_SEEDS
        ]

    for path in checkpoints:
        if not os.path.isfile(path):
            print(f"[skip] checkpoint not found: {path}")
            continue
        checkpoint_to_csv(path)


if __name__ == "__main__":
    main(sys.argv[1:])
