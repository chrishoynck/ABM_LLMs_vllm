"""
Standalone post-generation pipeline using the xAI Grok API.

This is intentionally separate from `test_phq9_llms.py`: the local-vLLM test
loop generates one post per request across many rounds, whereas this pipeline
issues ONE request per block and asks Grok to produce all posts of a block
(same persona, same PHQ-9) in a single completion. That keeps the persona +
PHQ-9 context billed once per block instead of once per post.

Output is written in the same CSV schema as
`TestLLMs.export_tweets_with_phq9`, so the result feeds straight into
`prompt_optimizer.parse_tweets_with_phq9_csv` / `train_val_test_split`.

Each block is streamed to disk (with a flush) the moment it finishes, so an
interrupted run - rate-limit exhaustion, a kill, a crash - loses nothing. The
output file is APPENDED to: rerun, optionally with a different --seed, to
extend the dataset. Block ids are seed-prefixed so reruns never collide.

Usage
-----
    export XAI_API_KEY=xai-...                # from https://console.x.ai
    python -m utils.create_data.generate_posts_grok \
        --num-blocks 300 --posts-per-block 10 \
        --output data/grok_posts/posts_with_phq9.csv
    # interrupted? just run it again (same or different --seed) - it appends.

Environment variables
---------------------
    XAI_API_KEY       required - the xAI API key
    LLM_API_BASE_URL  optional - default https://api.x.ai/v1
    LLM_API_MODEL     optional - default grok-4 (overridden by --model)
"""

import argparse
import asyncio
import csv
import json
import os
import re
import time

import numpy as np
from openai import AsyncOpenAI


# --------------------------------------------------------------------------
#  Self-contained copies of the small Agent helpers this pipeline needs.
#  Inlined (rather than importing classes.agent) so this file stays standalone
#  and free of the heavy local-model import chain (sklearn, transformers, ...).
# --------------------------------------------------------------------------
def _phq9_severity_category(score):
    """Map a PHQ-9 sumscore to a standard severity label."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if s <= 4:
        return "none/minimal"
    elif s <= 9:
        return "mild"
    elif s <= 14:
        return "moderate"
    elif s <= 19:
        return "moderately severe"
    return "severe"


def _well_being_prompt(well_being):
    """Concise well-being line from a PHQ-9 dict (age + score + severity)."""
    score = well_being.get("phq9_sumscore")
    severity = _phq9_severity_category(score)
    return (
        f"You are {well_being.get('age')} years old."
        f"Current well-being: PHQ-9 score {float(score):.4f} "
        f"({severity} depression) "
    )


def _strip_model_thinking(text):
    """Strip chain-of-thought / <think> blocks some models emit."""
    cleaned = re.sub(r'<think>.*?</think>', '', text or '', flags=re.DOTALL).strip()
    if '</think>' in cleaned and '<think>' not in cleaned:
        cleaned = cleaned.split('</think>', 1)[1].strip()
    if '<think>' in cleaned:
        cleaned = cleaned[:cleaned.index('<think>')].strip()
    parts = re.split(r'(?i)\bThinking Process\s*:', cleaned)
    if len(parts) > 1:
        cleaned = parts[-1].strip()
    return cleaned


# --------------------------------------------------------------------------
#  Prompts.
#  The SYSTEM prompt (rules / tone / constraints) is loaded at runtime from a
#  prompt JSON - by default prompts_post_minimal.json -> tweet_gen.system_forced
#  - and its single-post output section is swapped for a batch one by
#  _build_system_prompt(). Editing that JSON therefore changes what Grok sees.
#
#  The USER message below is assembled in code: the minimal JSON's forced
#  user template has no slot for neighbour posts, so the structural envelope
#  (persona + PHQ-9 + neighbours) is kept here.
# --------------------------------------------------------------------------
BATCH_USER = """ID: {agent_id}
Persona: {persona}
Well-being State PHQ-9 (0-27): {well_being}

### RECENT POSTS FROM USERS YOU FOLLOW ###
{neighbor_block}

### TASK ###
Generate {n_posts} posts now using the MANDATORY OUTPUT FORMAT."""

# Splits a single-post system prompt at its output-format section.
_OUTPUT_FORMAT_MARKER = "### MANDATORY OUTPUT FORMAT ###"


def build_batch_system_prompt(prompts_file, n_posts, max_chars, tweet_instruction=None):
    """Load `tweet_gen.system_forced` from `prompts_file` and adapt it to a
    batch request: keep the rules/constraints, replace the single-post output
    section with one that asks for `n_posts` enumerated posts.

    When `tweet_instruction` is given (the optimised ### RULES ### block written
    by prompt_optimizer), the original rules block is swapped for it before the
    batch-format adapter runs — so Grok ends up using the optimised rules
    inside its existing system-prompt structure.
    """
    with open(prompts_file, encoding="utf-8") as f:
        tweet_gen = json.load(f)["tweet_gen"]

    system_forced = tweet_gen["system_forced"].replace("{max_chars}", str(max_chars))

    if tweet_instruction is not None:
        rules_marker = "### RULES ###"
        constraints_marker = "### CONSTRAINTS ###"
        if rules_marker in system_forced and constraints_marker in system_forced:
            intro, rest = system_forced.split(rules_marker, 1)
            _old_rules, tail = rest.split(constraints_marker, 1)
            opt = tweet_instruction.strip()
            if not opt.startswith(rules_marker):
                opt = f"{rules_marker}\n{opt}"
            system_forced = (intro.rstrip() + "\n\n" + opt.rstrip()
                             + "\n\n" + constraints_marker + tail)
        else:
            # No rules/constraints structure to splice into — use the optimised
            # instruction as the whole pre-output-format prompt.
            system_forced = tweet_instruction.strip()

    # Keep everything before the single-post output-format section.
    base = system_forced.split(_OUTPUT_FORMAT_MARKER)[0].rstrip()

    return (
        f"{base}\n\n"
        f"### POST-SET CONTEXT ###\n"
        f"You are writing {n_posts} separate posts that this SAME user published on "
        f"different days over the past two weeks. They all sit at the SAME well-being "
        f"level, but each post is a DIFFERENT, specific topic - do not reuse topics "
        f"or words across posts. The user message lists recent posts from accounts "
        f"this user follows; across your {n_posts} posts you may engage with several "
        f"DIFFERENT ones (or none) - vary which feed posts you react to, and do not "
        f"react to the same feed post twice.\n\n"
        f"{_OUTPUT_FORMAT_MARKER}\n"
        f"Output exactly {n_posts} posts, each on its own single line, prefixed "
        f"exactly as shown:\n"
        f"POST 1: <content>\n"
        f"POST 2: <content>\n"
        f"...\n"
        f"POST {n_posts}: <content>\n"
        f"Do NOT output any reasoning, headers, commentary, or blank lines."
    )


_PREFIX_RE = re.compile(r'^\s*(?:POST|TWEET)\s*#?\s*\d*\s*[:.)\-]\s*', re.IGNORECASE)


def load_neighbor_pool(path):
    """Load (agent_id, post_text) pairs from a tweets_with_phq9 CSV.

    Reads the `agent_id` + `tweet` columns. Leading POST:/TWEET: prefixes and
    NO_POST placeholders are stripped. agent_ids are kept so the caller can
    format neighbour posts as `@user_<id>: ...` for the model to reply to.
    """
    if not path.endswith(".csv"):
        raise ValueError(
            f"neighbour pool must be a .csv (got {path}); legacy .txt support removed"
        )
    pool = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            agent_id = (row.get("agent_id") or "").strip()
            text = (row.get("tweet") or "").strip()
            text = _PREFIX_RE.sub("", text).strip().strip('"').strip()
            if text and text.upper() not in {"NO_POST", "NO_TWEET"}:
                pool.append((agent_id, text))
    return pool


class GrokPostGenerator:
    """Generate `num_blocks` blocks of posts via the Grok API.

    Each block is one persona at one PHQ-9 score (0-27); PHQ-9 scores are
    spread as evenly as possible across blocks so the full 0-27 range is
    covered. One block == one API request.
    """

    # Matches "POST 1: ...", "POST: ...", "TWEET 3 - ...", "1. ..." etc.
    _POST_LINE = re.compile(
        r'^\s*(?:POST|TWEET)?\s*#?\s*\d*\s*[:.)\-]\s*(.+\S)\s*$',
        re.IGNORECASE,
    )

    def __init__(self, personas, well_beings, num_blocks=300, posts_per_block=10,
                 seed=42, model=None, max_concurrency=8,
                 neighbor_pool=None, num_neighbors=12,
                 prompts_file="data/prompts_post_minimal.json", max_chars=240,
                 tweet_instruction=None, phq9_assignments=None):
        if len(personas) < num_blocks:
            raise ValueError(
                f"Need >= {num_blocks} personas, got {len(personas)}."
            )
        if not well_beings:
            raise ValueError("well_beings must be non-empty (used for agent age).")
        if phq9_assignments is not None and len(phq9_assignments) < num_blocks:
            raise ValueError(
                f"phq9_assignments has {len(phq9_assignments)} entries but "
                f"num_blocks={num_blocks} requested."
            )

        self.num_blocks = num_blocks
        self.posts_per_block = posts_per_block
        self.seed = seed
        self.max_concurrency = max_concurrency
        self.personas = list(personas)
        self.well_beings = list(well_beings)
        self.neighbor_pool = list(neighbor_pool) if neighbor_pool else []
        self.num_neighbors = num_neighbors
        self.phq9_assignments = (
            [int(x) for x in phq9_assignments[:num_blocks]]
            if phq9_assignments is not None else None
        )

        self.model = model or os.environ.get("LLM_API_MODEL", "grok-4")
        self.client = self._setup_client()

        # System prompt loaded from the prompt JSON, adapted to a batch request;
        # the optimised instruction (if any) is spliced in before the adapter.
        self.system_prompt = build_batch_system_prompt(
            prompts_file, posts_per_block, max_chars,
            tweet_instruction=tweet_instruction,
        )

        self.blocks = self._build_blocks()
        self.results = []  # filled by generate()

    # ------------------------------------------------------------------
    #  Setup
    # ------------------------------------------------------------------
    @staticmethod
    def _setup_client():
        """Build the xAI Grok client (OpenAI-compatible API)."""
        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key found. Set XAI_API_KEY (from https://console.x.ai)."
            )
        base_url = os.environ.get("LLM_API_BASE_URL", "https://api.x.ai/v1")
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _build_blocks(self):
        """Assign each block a distinct persona and a PHQ-9 score.

        With `phq9_assignments` supplied, scores come from the file in order
        (so every model run sees the same persona → PHQ-9 mapping). Otherwise
        scores 0-27 are cycled then shuffled, balanced and unsorted.
        """
        rng = np.random.default_rng(self.seed)
        if self.phq9_assignments is not None:
            scores = np.array(self.phq9_assignments, dtype=int)
        else:
            scores = np.array([i % 28 for i in range(self.num_blocks)])
            rng.shuffle(scores)

        blocks = []
        for i in range(self.num_blocks):
            # Re-use real PHQ-9 survey rows only for the `age` field; the
            # PHQ-9 score itself is the experimentally controlled target.
            well_being = dict(self.well_beings[i % len(self.well_beings)])
            well_being["phq9_sumscore"] = int(scores[i])

            # Sample the "users you follow" context for this block.
            neighbors = []
            if self.neighbor_pool and self.num_neighbors > 0:
                n = min(self.num_neighbors, len(self.neighbor_pool))
                idx = rng.choice(len(self.neighbor_pool), size=n, replace=False)
                neighbors = [self.neighbor_pool[j] for j in idx]

            blocks.append({
                "agent_id": i,
                "persona": self.personas[i],
                "age": well_being.get("age"),
                "phq9": int(scores[i]),
                "well_being": well_being,
                "neighbors": neighbors,
            })
        return blocks

    # ------------------------------------------------------------------
    #  Prompt building / response parsing
    # ------------------------------------------------------------------
    def _build_messages(self, block):
        well_being_str = _well_being_prompt(block["well_being"])
        neighbors = block.get("neighbors") or []
        if neighbors:
            neighbor_block = "\n".join(
                f"- @user_{aid}: {t}" if aid else f"- {t}"
                for aid, t in neighbors
            )
        else:
            neighbor_block = "(no posts from users you follow)"
        # Atomic single-print so concurrent blocks don't interleave mid-message.
        print(
            f"[grok] block {block['agent_id']} sampled "
            f"{len(neighbors)} neighbor posts:\n{neighbor_block}\n",
            flush=True,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",
             "content": BATCH_USER.format(
                 agent_id=block["agent_id"],
                 persona=block["persona"],
                 well_being=well_being_str,
                 neighbor_block=neighbor_block,
                 n_posts=self.posts_per_block,
             )},
        ]

    def _parse_posts(self, text):
        """Extract individual posts from a batch completion."""
        text = _strip_model_thinking(text or "")
        posts = []
        for line in text.splitlines():
            match = self._POST_LINE.match(line)
            if not match:
                continue
            content = match.group(1).strip().strip('"').strip()
            if content and content.upper() not in {"NO_POST", "NO_TWEET"}:
                posts.append(content)
        return posts

    # ------------------------------------------------------------------
    #  Generation - each block is written to disk the moment it finishes.
    # ------------------------------------------------------------------
    async def _generate_block(self, sem, block, attempts=5):
        """Generate one block; retry on API errors or short/incomplete output.

        Returns (block, posts). `posts` may have fewer than posts_per_block
        entries if every attempt fell short. Never raises - all errors are
        caught and retried, so an exhausted block just yields what it has.
        """
        messages = self._build_messages(block)
        best = []
        async with sem:
            for attempt in range(attempts):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.7,
                        top_p=0.8,
                        # presence_penalty=0.4,
                        # frequency_penalty=0.3,
                        max_tokens=self.posts_per_block * 100 + 200,
                        seed=self.seed + block["agent_id"],
                    )
                    posts = self._parse_posts(resp.choices[0].message.content)
                    if len(posts) > len(best):
                        best = posts
                    if len(posts) >= self.posts_per_block:
                        return block, posts[:self.posts_per_block]
                    # Parsed too few posts - retry unless this was the last try.
                    if attempt < attempts - 1:
                        print(f"[grok] block {block['agent_id']}: "
                              f"{len(posts)}/{self.posts_per_block} posts, retrying")
                except Exception as exc:
                    wait = min(60, 2 ** attempt)
                    headers = getattr(getattr(exc, "response", None), "headers", None)
                    if headers:
                        try:
                            wait = float(headers.get("retry-after", wait))
                        except (TypeError, ValueError):
                            pass
                    print(f"[grok] block {block['agent_id']} failed "
                          f"(attempt {attempt + 1}/{attempts}): {exc} "
                          f"- retrying in {wait:.0f}s")
                    await asyncio.sleep(wait)

        if len(best) < self.posts_per_block:
            print(f"[grok] block {block['agent_id']}: kept {len(best)}/"
                  f"{self.posts_per_block} posts after {attempts} attempts")
        return block, best

    def _row_id(self, block):
        """Seed-prefixed block id, so reruns (incl. different seeds) never
        collide when appended to the same output file."""
        return f"{self.seed}_{block['agent_id']}"

    def _write_block(self, writer, block, posts):
        """Append one finished block's rows to the open CSV."""
        row_id = self._row_id(block)

        # Echo every generated post to stdout as the block is written.
        print(f"=== Agent {row_id} persona: {block['persona']} "
              f"phq9: {block['phq9']} ===")
        for step, post in enumerate(posts):
            print(f"  POST {step}: {post}")

        for step, post in enumerate(posts):
            clean = post.replace("\n", " ").replace("\r", " ").strip()
            writer.writerow([row_id, block["persona"], block["age"],
                             step, block["phq9"], clean, False])

    def generate(self, csv_path):
        """Generate every block, streaming each to disk as soon as it finishes.

        Rows are flushed to `csv_path` after every block, so an interrupted
        run - rate-limit exhaustion, a kill, a crash - keeps every block
        completed so far. The file is APPENDED to: rerun (e.g. with a
        different --seed) to extend the dataset. Block ids are seed-prefixed
        so reruns never collide.

        Returns self.results (the blocks written this run).
        """
        asyncio.run(self._generate_streaming(csv_path))
        return self.results

    async def _generate_streaming(self, csv_path):
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        csv_is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

        t0 = time.perf_counter()
        self.results = []
        dropped = 0
        done = 0
        n_blocks = len(self.blocks)

        # Append mode: a rerun extends the file instead of overwriting it.
        csv_f = open(csv_path, "a", newline="", encoding="utf-8")
        try:
            writer = csv.writer(csv_f)
            if csv_is_new:
                writer.writerow(["agent_id", "persona", "age", "step",
                                 "phq9", "tweet", "interaction"])
            csv_f.flush()

            sem = asyncio.Semaphore(self.max_concurrency)
            tasks = [asyncio.create_task(self._generate_block(sem, b))
                     for b in self.blocks]

            # as_completed -> handle blocks the instant they return, in any
            # order; each block's rows are still written contiguously.
            for coro in asyncio.as_completed(tasks):
                block, posts = await coro
                done += 1
                if len(posts) < 2:  # parse_tweets_with_phq9_csv needs > 1 post
                    dropped += 1
                    continue
                self._write_block(writer, block, posts)
                csv_f.flush()
                self.results.append({
                    "agent_id": self._row_id(block),
                    "persona": block["persona"],
                    "age": block["age"],
                    "phq9": block["phq9"],
                    "posts": posts,
                })
                if done % 25 == 0 or done == n_blocks:
                    print(f"[grok] {done}/{n_blocks} blocks done "
                          f"({len(self.results)} written, {dropped} dropped)")
        finally:
            csv_f.close()

        elapsed = time.perf_counter() - t0
        total_posts = sum(len(r["posts"]) for r in self.results)
        print(f"[grok] wrote {len(self.results)} blocks ({total_posts} posts) "
              f"to {csv_path} in {elapsed:.1f}s; "
              f"{dropped} blocks dropped (< 2 posts).")


def main():
    parser = argparse.ArgumentParser(
        description="Generate persona + PHQ-9 conditioned posts via the Grok API."
    )
    parser.add_argument("--num-blocks", type=int, default=300,
                        help="Number of (persona, PHQ-9) blocks to generate.")
    parser.add_argument("--posts-per-block", type=int, default=10,
                        help="Posts generated per block (one API request each).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None,
                        help="Grok model (default: $LLM_API_MODEL or grok-4).")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max simultaneous API requests (mind rate limits).")
    parser.add_argument("--personas-file", default="data/personas_short_10k.csv")
    parser.add_argument("--num-neighbors", type=int, default=12,
                        help="Feed posts shown as context per block (0 to disable).")
    parser.add_argument("--output", default="data/grok_posts/posts_with_phq9.csv")
    parser.add_argument(
        "--persona-pool", default="data/personas_eval_1000.csv",
        help="Shared eval-persona file; built on first call if missing. "
             "Pass '' to bypass and use the legacy --personas-file sampling.",
    )
    parser.add_argument(
        "--persona-phq9-file", default=None,
        help="CSV with `persona` + `phq9` columns (built by "
             "utils.tools.build_persona_phq9_eval). When set, both personas and "
             "PHQ-9 scores come from this file in order.",
    )
    parser.add_argument(
        "--tweet-instruction-file", default=None,
        help="Path to best_instruction_tweet.txt from a prompt_optimizer tweet "
             "run. When set, the optimised ### RULES ### block is spliced into "
             "the system prompt before Grok's batch-output adapter runs.",
    )
    args = parser.parse_args()

    # Hardcoded defaults (previously CLI flags; restore as args if you need to override).
    PHQ9_FILE = "data/confidential/phq9.sav"
    NEIGHBOR_SOURCE = "data/test_post/Qwen_Qwen3.5-27B/temp_0.8_top_p_0.6_cp_10_no_inter/seed_75/tweets_with_phq9.csv"
    PROMPTS_FILE = "data/prompts_post_minimal.json"
    MAX_CHARS = 240

    # Imported lazily so the GrokPostGenerator class can be used without pandas.
    try:
        import utils.tools.load_personas as lp
    except ImportError:  # allow running from inside src/utils
        from ..tools import load_personas as lp

    phq9_assignments = None
    if args.persona_phq9_file:
        import pandas as pd
        df_pp = pd.read_csv(args.persona_phq9_file)
        if len(df_pp) < args.num_blocks:
            raise ValueError(
                f"--persona-phq9-file has {len(df_pp)} rows but "
                f"--num-blocks={args.num_blocks} requested."
            )
        personas = df_pp["persona"].head(args.num_blocks).tolist()
        phq9_assignments = df_pp["phq9"].head(args.num_blocks).astype(int).tolist()
        print(f"[persona-phq9] using {args.num_blocks} pairs from {args.persona_phq9_file}")
    elif args.persona_pool:
        personas = lp.load_or_build_persona_pool(
            n_needed=args.num_blocks,
            pool_path=args.persona_pool,
            source=args.personas_file,
        )
    else:
        personas = lp.load_personas_from_file(
            args.personas_file, personass_to_load=args.num_blocks, seed=args.seed
        )

    tweet_instruction = None
    if args.tweet_instruction_file:
        try:
            from utils.create_data.test_phq9_llms import load_instruction_file
        except ImportError:
            from .test_phq9_llms import load_instruction_file
        tweet_instruction = load_instruction_file(args.tweet_instruction_file)
        print(f"[align] system prompt uses optimised instruction from "
              f"{args.tweet_instruction_file} ({len(tweet_instruction.split())} words)")
    # Only `age` is taken from this; loading fewer rows than blocks is fine
    # (ages are cycled). 28 covers every PHQ-9 score at least once.
    well_beings = lp.load_phq9(
        PHQ9_FILE, personass_to_load=max(28, args.num_blocks // 4),
        seed=args.seed,
    )

    neighbor_pool = []
    if args.num_neighbors > 0:
        neighbor_pool = load_neighbor_pool(NEIGHBOR_SOURCE)
        print(f"Loaded {len(neighbor_pool)} neighbor posts from {NEIGHBOR_SOURCE}")

    generator = GrokPostGenerator(
        personas=personas,
        well_beings=well_beings,
        num_blocks=args.num_blocks,
        posts_per_block=args.posts_per_block,
        seed=args.seed,
        model=args.model,
        max_concurrency=args.concurrency,
        neighbor_pool=neighbor_pool,
        num_neighbors=args.num_neighbors,
        prompts_file=PROMPTS_FILE,
        max_chars=MAX_CHARS,
        tweet_instruction=tweet_instruction,
        phq9_assignments=phq9_assignments,
    )
    # Streams each block to args.output as it finishes; appends if the file
    # already exists, so a rerun (e.g. different --seed) extends the dataset.
    generator.generate(args.output)


if __name__ == "__main__":
    main()
