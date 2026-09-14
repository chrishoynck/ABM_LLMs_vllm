"""Check the 3-block smoke run of each generator model before the big jobs.

Reads `data/finetune/smoke/smoke_<model>.csv` and its `.meta.json` and checks
what used to be eyeballed in the SLURM log: row count, agent count, leaked
chat-template tokens, NO_POST rate, post length and the decoding settings.
Prints PASS or FAIL per model and exits 1 if anything failed. CPU only.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

# Tokens that mean the chat template or the POST: parser leaked into the text.
LEAKS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|channel>",
         "<turn|>", "<start_of_turn>", "<end_of_turn>", "POST:"]
# Models whose decoding must not drift (regression check on loaders.STUDENT_DECODING).
EXPECTED_DECODING = {"qwen27": (0.7, 0.9)}
NO_CONTENT = {"NO_POST", "NO_TWEET"}


def check_model(csv_path, max_no_post, min_len, max_len):
    """Run all checks on one smoke CSV.

    Args:
        csv_path (str): the smoke CSV; its `.meta.json` sits next to it.
        max_no_post (float): highest NO_POST fraction that still passes.
        min_len (int): lowest median post length (chars) that still passes.
        max_len (int): highest median post length (chars) that still passes.

    Returns:
        list[tuple[str, bool, str]]: (check name, passed, detail) per check.
    """
    results = []
    meta_path = csv_path + ".meta.json"
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    df = pd.read_csv(csv_path)
    posts = df["tweet"].astype(str)
    real = posts[~posts.isin(NO_CONTENT)]

    n_agents = meta.get("num_agents", 3)
    n_rows = n_agents * meta.get("check_point", 10)
    results.append(("row count", len(df) == n_rows, f"{len(df)} rows, expected {n_rows}"))
    results.append(("agent count", df["agent_id"].nunique() == n_agents,
                    f"{df['agent_id'].nunique()} agent_ids, expected {n_agents}"))

    leaked = [tok for tok in LEAKS if real.str.contains(tok, regex=False).any()]
    results.append(("no leaked tokens", not leaked, "found " + ", ".join(leaked) if leaked else "clean"))

    no_post = 1 - len(real) / max(len(df), 1)
    results.append(("NO_POST rate", no_post <= max_no_post, f"{no_post:.0%} (max {max_no_post:.0%})"))

    med = real.str.len().median() if len(real) else 0
    results.append(("post length", min_len <= med <= max_len, f"median {med:.0f} chars (want {min_len}-{max_len})"))

    tag = meta.get("model_arg")
    if tag in EXPECTED_DECODING:
        want = EXPECTED_DECODING[tag]
        got = (meta.get("temp"), meta.get("top_p"))
        results.append(("decoding", got == want, f"temp/top_p {got}, expected {want}"))
    else:
        results.append(("decoding", True, f"temp/top_p ({meta.get('temp')}, {meta.get('top_p')})"))
    return results


def main():
    """Check every smoke CSV in the folder and exit 1 if any check fails."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="data/finetune/smoke", help="folder with smoke_<model>.csv files")
    ap.add_argument("--max-no-post", type=float, default=0.10)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=600)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.dir, "smoke_*.csv")))
    if not paths:
        sys.exit(f"no smoke_*.csv in {a.dir}")
    failed = False
    for p in paths:
        results = check_model(p, a.max_no_post, a.min_len, a.max_len)
        ok = all(r[1] for r in results)
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'}  {os.path.basename(p)}")
        for name, passed, detail in results:
            print(f"    [{'ok' if passed else 'FAIL'}] {name}: {detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
