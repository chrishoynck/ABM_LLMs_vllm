import textgrad as tg
from textgrad.engine.vllm import ChatVLLM
import json
import numpy as np
import os
import re
import csv
import torch
import glob
import torch.nn as nn
import copy
import torch.optim as optim
from utils.metrics import *
from utils.format_config import FC
from torch.optim.lr_scheduler import ReduceLROnPlateau
from classes.agent import Agent


# ── Thinking-model helpers ────────────────────────────────────────────────

def strip_teacher_thinking(text: str) -> str:
    """Strip ``<think>…</think>`` reasoning blocks from teacher-model output.
    No-op for models that don't produce thinking tags."""
    return Agent.strip_model_thinking(text)


def _make_engine_thinking_aware(engine):
    """Monkey-patch *engine*.generate so every output has thinking blocks
    stripped automatically.  Safe for non-thinking models."""
    _orig = engine.generate
    def _patched(*args, **kwargs):
        raw = _orig(*args, **kwargs)
        if isinstance(raw, str):
            return strip_teacher_thinking(raw)
        if isinstance(raw, list):
            return [strip_teacher_thinking(r) if isinstance(r, str) else r
                    for r in raw]
        return raw
    engine.generate = _patched
    return engine


def parse_tweets_with_phq9(file_path: str):
    """
    Parse a tweets_with_phq9.txt file and group consecutive tweets that share
    the same PHQ-9 score into blocks.

    Parameters:
        file_path (str): Path to a tweets_with_phq9.txt file.

    Returns:
        tweet_blocks (list[list[str]]): Each element is a list of tweet strings from one PHQ-9 period.
        true_answers (list[int]): The true PHQ-9 score for the corresponding tweet block.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tweet_blocks = []
    true_answers = []

    current_phq9 = None
    current_tweets = []

    step_pattern = re.compile(
        r'^step\s+\d+:\s+phq9=(\d+)\s+tweet="(.+)"'
    )

    for line in lines:
        line = line.rstrip("\n")

        # New agent header, flush current block and reset
        if line.startswith("=== Agent"):
            if current_tweets:
                if len(current_tweets) > 1:
                    tweet_blocks.append(current_tweets)
                    true_answers.append(current_phq9)
            current_phq9 = None
            current_tweets = []
            continue

        match = step_pattern.match(line)
        if not match:
            continue

        phq9 = int(match.group(1))
        tweet = match.group(2)

        # Strip trailing metadata like '  (CHANGED from X)'
        changed_idx = tweet.rfind("  (CHANGED from ")
        if changed_idx != -1:
            tweet = tweet[:changed_idx]

        if phq9 != current_phq9:
            # New PHQ-9 period save the previous block (if any)
            if current_tweets:
                if len(current_tweets) > 1:
                    tweet_blocks.append(current_tweets)
                    true_answers.append(current_phq9)
            current_phq9 = phq9
            current_tweets = [tweet]
        else:
            current_tweets.append(tweet)

    # Flush the last block
    if current_tweets:
        if len(current_tweets) > 1:
            tweet_blocks.append(current_tweets)
            true_answers.append(current_phq9)

    return tweet_blocks, true_answers


def parse_tweets_with_phq9_csv(file_path: str):
    """
    Parse the tweets_with_phq9.csv file written by TestLLMs.export_tweets_with_phq9_txt.

    CSV format (one row per agent/step):
        agent_id, persona, age, step, phq9, tweet, interaction

    We reconstruct the same (tweet_blocks, true_answers) structure as the
    text parser by grouping consecutive rows with the same (agent_id, phq9)
    into one block and using that PHQ-9 score as the label.
    """
    tweet_blocks: list[list[str]] = []
    true_answers: list[int] = []
    personas: list[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        current_agent = None
        current_phq9 = None
        current_tweets: list[str] = []

        for row in reader:
            agent_id = row.get("agent_id")
            # Robust to missing/empty phq9 field
            try:
                phq9 = int(row.get("phq9")) if row.get("phq9") not in (None, "") else None
            except ValueError:
                continue
            try:
                persona = row.get("persona") if row.get("persona") not in (None, "") else None
            except ValueError:
                continue

            tweet = (row.get("tweet") or "").strip()

            if agent_id != current_agent or phq9 != current_phq9:
                # Flush previous block for previous agent/score
                if current_tweets and current_phq9 is not None:
                    if len(current_tweets) > 1:
                        tweet_blocks.append(current_tweets)
                        true_answers.append(current_phq9)
                        personas.append(current_persona)

                current_agent = agent_id
                current_phq9 = phq9
                current_persona = persona
                current_tweets = [tweet] if tweet else []
            else:
                if tweet:
                    current_tweets.append(tweet)

        # Flush last block
        if current_tweets and current_phq9 is not None:
            if len(current_tweets) > 1:
                tweet_blocks.append(current_tweets)
                true_answers.append(current_phq9)
                personas.append(current_persona)
    return tweet_blocks, true_answers, personas


FORMAT_SPLIT_MARKER = "### OPTIONS ###"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"
QWEN_27 = "Qwen/Qwen3.5-27B"


# def _split_system_prompt(full_system: str):
#     """Split the PHQ-9 system prompt into an optimisable instruction part
#     and a fixed format part (options, questions, answer format)."""
#     idx = full_system.find(FORMAT_SPLIT_MARKER)
#     if idx == -1:
#         return full_system.rstrip(), ""
#     return full_system[:idx].rstrip(), full_system[idx:]


def _build_user_message(format_block: str, tweet_block: list, prompts: dict) -> str:
    """Compose the user message: fixed PHQ-9 format + tweet data."""
    tweets_text = prompts["phq9"]["user_template_forced"].format(
        tweets_block="\n".join(tweet_block)
    )
    return f"{format_block}\n\n{tweets_text}"




def _evaluate_instruction(engine, instruction_text: str, format_block: str,
                          blocks: list, answers: list, prompts: dict) -> float:
    """Run the current instruction on *blocks* and return the MAE."""
    total_ae = 0
    for tweet_block, true_answer in zip(blocks, answers):
        user_msg = _build_user_message(format_block, tweet_block, prompts)
        response = engine.generate(user_msg, system_prompt=instruction_text)
        predicted = parse_phq9_answers(response)
        total_ae += abs(predicted - true_answer)
    return total_ae / max(len(blocks), 1)


def _make_loss_prompt(true_answer: int, predicted: int) -> str:
    """Concise per-sample loss prompt for the backward engine."""
    error = predicted - true_answer
    if error > 0:
        direction = f"overestimated by {error}"
    elif error < 0:
        direction = f"underestimated by {abs(error)}"
    else:
        direction = "correct"
    return (
        f"True PHQ-9 sumscore = {true_answer}, predicted = {predicted} ({direction}). "
        f"Give concise, actionable feedback (1-2 sentences) on what the system "
        f"instruction should say differently to improve reasoning from tweets "
        f"to PHQ-9 scores. Focus on calibration patterns and reasoning gaps, "
        f"not on individual PHQ-9 item scores."
    )

def train_val_test_split(rng, file_paths:list[str],
                         val_fraction: float = 0.10,
                         test_fraction: float = 0.10):
    """Split tweet-block data into train / validation / test sets.

    Returns
    -------
    train_data, val_data, test_data – each a tuple (blocks, answers, personas)
    """
    tweet_blocks_list = []
    true_answers_list = []
    personas_list = []
    
    for file_path in file_paths:
        if file_path.endswith(".csv"):
            csv_path = file_path
            txt_path = file_path.replace(".csv", ".txt")
        else:
            txt_path = file_path
            csv_path = file_path.replace(".txt", ".csv") if file_path.endswith(".txt") else file_path + ".csv"

        if os.path.isfile(csv_path):
            tweet_blocks, true_answers, personas = parse_tweets_with_phq9_csv(csv_path)
            tweet_blocks_list.extend(tweet_blocks)
            true_answers_list.extend(true_answers)
            personas_list.extend(personas)
            print(f"Parsed {len(tweet_blocks)} tweet blocks from {csv_path}")
        else:
            tweet_blocks, true_answers = parse_tweets_with_phq9(txt_path)
            personas = [None] * len(tweet_blocks)
            tweet_blocks_list.extend(tweet_blocks)
            true_answers_list.extend(true_answers)
            personas_list.extend(personas)
            print(f"Parsed {len(tweet_blocks)} tweet blocks from {txt_path} (CSV not found, used TXT parser)")

    perm = rng.permutation(len(tweet_blocks_list))
    n = len(tweet_blocks_list)
    n_test = max(1, int(n * test_fraction))
    n_val  = max(1, int(n * val_fraction))

    test_idx  = perm[:n_test]
    val_idx   = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    def _select(indices):
        return ([tweet_blocks_list[i] for i in indices],
                [true_answers_list[i] for i in indices],
                [personas_list[i] for i in indices])

    train_data = _select(train_idx)
    val_data   = _select(val_idx)
    test_data  = _select(test_idx)
    print(f"Train: {len(train_idx)},  Val: {len(val_idx)},  Test: {len(test_idx)}")
    return train_data, val_data, test_data

def call_optimizer_phq9(
    file_paths: list[str],
    model_name: str = QWEN_27,
    batch_size: int = 4,
    max_instruction_words: int = 300,
    num_steps: int = None,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    validate_every: int = 5,
    val_sample_size: int = 10,
    seed: int = 42,
    **vllm_kwargs,
):
    """
    Optimise the PHQ-9 instruction (task framing + reasoning guidance)
    using TextGrad with batched gradient accumulation.

    Key design choices
    ------------------
    * Train / val / test split (default 80 / 10 / 10).
    * Validation is done on a *sampled subset* (``val_sample_size``) every
      ``validate_every`` steps to keep cost manageable.
    * Full test-set evaluation at the end with the best instruction.
    * Engine is patched to strip ``<think>`` blocks so Qwen3.5 (and similar
      reasoning models) work transparently as the teacher / backward engine.
    """

    rng = np.random.default_rng(seed)

    train_data, val_data, test_data = train_val_test_split(
        rng, file_paths, val_fraction, test_fraction,
    )
    train_blocks, train_answers, _train_personas = train_data
    val_blocks, val_answers, _val_personas = val_data
    test_blocks, test_answers, _test_personas = test_data

    # Engine (single model for forward + backward)
    tp = vllm_kwargs.pop("tensor_parallel_size", None) or torch.cuda.device_count()
    student_engine = ChatVLLM(
        model_string=model_name,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.45,
        max_model_len=16384,
        **vllm_kwargs,
    )
    if model_name == QWEN_27:
        student_engine.model.disable_thinking()
    
    teacher_engine = ChatVLLM(
        model_string=model_name,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.45,
        max_model_len=16384,
        **vllm_kwargs,
    )
    _make_engine_thinking_aware(teacher_engine)
    tg.set_backward_engine(teacher_engine, override=True)

    # --- split prompt into instruction (grad) + format (fixed) --------------
    with open(FC.PROMPTS_FILE, "r") as f:
        prompts = json.load(f)

    instruction_text = prompts["phq9"]["system_instruction"]
    format_block = prompts["phq9"]["System_format"]

    instruction = tg.Variable(
        instruction_text,
        role_description=(
            "System-prompt INSTRUCTION for PHQ-9 assessment from tweets. "
            "Contains task framing and step-by-step reasoning guidance only. "
            "The PHQ-9 questions, scoring options, and answer format are "
            "provided separately and must NOT be duplicated here."
        ),
        requires_grad=True,
    )

    optimizer = tg.TGD(
        parameters=[instruction],
        constraints=[
            f"The instruction MUST stay concise — at most {max_instruction_words} words.",
            "Do NOT include the PHQ-9 questions, scoring options (0-3), or "
            "answer format — they are provided separately in the user message.",
            "Focus on actionable reasoning guidance that helps calibrate "
            "PHQ-9 inference from tweet histories.",
        ],
        gradient_memory=0,
    )

    if num_steps is None:
        num_steps = min(50, max(1, len(train_blocks) // batch_size))

    best_val_mae = float("inf")
    best_instruction = instruction.value

    # ── training loop ──────────────────────────────────────────────────────
    for step in range(num_steps):
        batch_idx = rng.choice(
            len(train_blocks),
            size=min(batch_size, len(train_blocks)),
            replace=False,
        )
        batch_blocks = [train_blocks[i] for i in batch_idx]
        batch_answers = [train_answers[i] for i in batch_idx]

        optimizer.zero_grad()
        model = tg.BlackboxLLM(student_engine, system_prompt=instruction)

        losses = []
        step_errors = []

        for tweet_block, true_answer in zip(batch_blocks, batch_answers):
            user_msg = _build_user_message(format_block, tweet_block, prompts)
            question = tg.Variable(
                user_msg,
                role_description="PHQ-9 format, questions, and patient tweet history",
                requires_grad=False,
            )

            prediction = model(question)
            predicted_score = parse_phq9_answers(prediction.value)
            error = predicted_score - true_answer
            step_errors.append(error)

            loss_fn = tg.TextLoss(_make_loss_prompt(true_answer, predicted_score))
            loss = loss_fn(prediction)
            losses.append(loss)

        total_loss = tg.sum(losses)
        total_loss.backward()
        optimizer.step()

        batch_mae = np.mean(np.abs(step_errors))
        print(f"[Step {step+1}/{num_steps}]  batch MAE={batch_mae:.2f}  errors={step_errors}")

        if (step + 1) % 5 == 0:
            print(f"  Current instruction:\n    {instruction.value[:200]}...")

        # periodic validation on a *sampled* subset
        if (step + 1) % validate_every == 0 or step == num_steps - 1:
            n_sample = min(val_sample_size, len(val_blocks))
            sample_idx = rng.choice(len(val_blocks), size=n_sample, replace=False)
            sampled_blocks  = [val_blocks[i]  for i in sample_idx]
            sampled_answers = [val_answers[i] for i in sample_idx]

            val_mae = _evaluate_instruction(
                student_engine, instruction.value, format_block,
                sampled_blocks, sampled_answers, prompts,
            )
            print(f"  -> Val MAE ({n_sample} samples): {val_mae:.2f}  (best: {best_val_mae:.2f})")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_instruction = instruction.value
                print("  -> New best instruction saved!")

    # ── final test evaluation ──────────────────────────────────────────────
    test_mae = _evaluate_instruction(
        student_engine, best_instruction, format_block,
        test_blocks, test_answers, prompts,
    )
    print(f"\nTest MAE (n={len(test_blocks)}): {test_mae:.2f}")

    # ── save results ───────────────────────────────────────────────────────
    output_dir = os.path.dirname(file_path)

    instr_path = os.path.join(output_dir, "optimized_instruction.txt")
    with open(instr_path, "w", encoding="utf-8") as f:
        f.write(best_instruction)
    print(f"\nBest instruction (val MAE={best_val_mae:.2f}, test MAE={test_mae:.2f}) → {instr_path}")

    full_path = os.path.join(output_dir, "optimized_full_prompt.txt")
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(best_instruction + "\n\n" + format_block)
    print(f"Full re-assembled prompt → {full_path}")

    return best_instruction



def parse_phq9_answers(answers: str) -> int:
        """
        Parse the PHQ-9 answers from the LLM output and compute the sumscore.
        Looks for the first digit found after the colon in each line.
        """
        lines = answers.strip().split("\n")
        total_score = 0
        
        for line in lines:

            parts = line.split(":", 1) # Split only on the first colon
            
            if len(parts) != 2:
                continue
                
            answer_part = parts[1].strip()
            
            # Find the first single digit (0-9) in the answer text
            match = re.search(r'\d', answer_part)
            
            if match:
                try:
                    score = int(match.group())
                    
                    # 3. Validate range (PHQ-9 scores must be 0, 1, 2, or 3)
                    if 0 <= score <= 3:
                        total_score += score
                    else:
                        print(f"Score out of range (found {score}) in line: {line}")
                except ValueError:
                    print(f"Could not convert match to int in line: {line}")
            else:
                print(f"No number found in answer part: {line}")
        
        # if total_score_llm != total_score:
        #     print(f"Total score mismatch: {total_score_llm} != {total_score}")
        
        return total_score


###################### TWEET GENERATION OPTIMIZATION ######################


def _build_user_message_tweet(context_tweets: list[str], prompts: dict,
                              persona: str, phq9_score: int) -> str:
    """Build the user message for tweet generation from the forced template.

    ``context_tweets`` are the (ground-truth) tweets that precede the one
    the student model is asked to generate (teacher forcing).
    """
    if context_tweets:
        prev_block = "### PREVIOUS POSTS ###\n" + "\n".join(
            f"- {t}" for t in context_tweets
        )
    else:
        prev_block = "### PREVIOUS POSTS ###\n(none yet)"

    template = prompts["tweet_gen"]["user_template_forced"]
    return template.format(
        agent_id="AGENT",
        persona=persona or "unspecified",
        well_being=phq9_score,
        previous_tweet_block=prev_block,
    )


def parse_tweet_answers(raw_output: str) -> str:
    """Extract tweet content from raw LLM output, stripping thinking blocks.

    Uses the static ``Agent.strip_model_thinking`` so no Agent instance is
    needed.  Falls back to the first paragraph of cleaned text.
    """
    cleaned = Agent.strip_model_thinking(raw_output)
    lower = cleaned.lower()

    for prefix in [FC.CONTENT_PREFIX_LOWER, "tweet:", "post:"]:
        idx = lower.rfind(prefix)
        if idx != -1:
            content = cleaned[idx + len(prefix):].strip()
            first_para = re.split(r'\n\s*\n', content, maxsplit=1)[0].strip()
            first_para = first_para.strip('`"\' ')
            if first_para and first_para.lower() not in (
                '<content>', '[content]', '...', '.', '', 'content',
            ):
                return first_para

    return cleaned.split("\n\n")[0].strip()


def _make_loss_prompt_tweet(predicted_tweet: str, true_tweet: str,
                            persona: str, phq9_score: int) -> str:
    """Per-sample loss prompt sent to the backward engine for tweet quality."""
    return (
        f"The student model generated the following tweet:\n"
        f"\"{predicted_tweet}\"\n\n"
        f"A reference tweet for this agent (persona: {persona}, "
        f"PHQ-9 = {phq9_score}) is:\n\"{true_tweet}\"\n\n"
        "Evaluate the generated tweet on Tone/Mood alignment with the "
        "PHQ-9 score, Content originality, and Naturalness. "
        "Tweets should be diverse, emotionally unfiltered, and not too "
        "obvious about symptoms or personality traits. "
        "Sometimes mood can deviate from general well-being.\n"
        "Give concise, actionable feedback (1-2 sentences) on what the "
        "system instruction should say differently to improve tweet quality. "
        "Do NOT base feedback on very specific persona details — those "
        "differ from person to person."
    )


def _evaluate_tweet_instruction(engine, instruction_text: str, prompts: dict,
                                blocks: list, answers: list, personas: list,
                                rng, sample_size: int = None) -> float:
    """Score tweet instruction quality by having the teacher rate outputs.

    For each sample a tweet is generated (teacher-forced context), then the
    teacher model rates it 0-10.  Returns the average score.
    """
    if sample_size and len(blocks) > sample_size:
        idx = rng.choice(len(blocks), size=sample_size, replace=False)
        blocks  = [blocks[i]  for i in idx]
        answers = [answers[i] for i in idx]
        personas = [personas[i] for i in idx]

    scores = []
    for tweet_block, phq9, persona in zip(blocks, answers, personas):
        pos = min(1, len(tweet_block) - 1)
        context = tweet_block[:pos]
        true_tweet = tweet_block[pos]

        user_msg = _build_user_message_tweet(context, prompts, persona, phq9)
        response = engine.generate(user_msg, system_prompt=instruction_text)
        predicted_tweet = parse_tweet_answers(response)

        rating_prompt = (
            f"Rate this generated tweet on a scale of 0-10 "
            f"(0=terrible, 10=excellent) based on naturalness, emotional "
            f"alignment with PHQ-9 score {phq9}, and similarity to this "
            f"reference tweet.\n\nReference: \"{true_tweet}\"\n"
            f"Generated: \"{predicted_tweet}\"\n\n"
            f"Respond with ONLY a single number 0-10."
        )
        rating_response = engine.generate(rating_prompt)
        try:
            score = float(re.search(r'\d+', rating_response).group())
            score = min(10.0, max(0.0, score))
        except (AttributeError, ValueError):
            score = 5.0
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def call_optimizer_tweets(
    file_path: str,
    model_name: str = LLAMA_70B,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
    batch_size: int = 4,
    max_instruction_words: int = 300,
    num_steps: int = None,
    validate_every: int = 5,
    val_sample_size: int = 8,
    max_chars: int = 240,
    **vllm_kwargs,
):
    """
    Optimise the tweet generation system prompt using TextGrad.

    Design
    ------
    * **Teacher forcing**: for each tweet position *i* in a ground-truth
      block, the student sees the *true* prior tweets ``block[:i]`` and must
      generate tweet *i*.  Each prediction is individually compared with the
      ground truth via a ``TextLoss`` evaluated by the teacher / backward
      engine.  This gives clean gradient flow per tweet.
    * Multiple tweet blocks per batch prevent skewing towards a single PHQ-9
      level.
    * The engine is patched to strip ``<think>`` blocks so reasoning models
      (Qwen3.5) work as the teacher without leaking chain-of-thought into
      gradients.  Non-thinking models are unaffected.
    * Train / val / test split with sampled validation for efficiency.
    """
    rng = np.random.default_rng(seed)
    train_data, val_data, test_data = train_val_test_split(
        rng, file_path, val_fraction, test_fraction,
    )
    train_blocks, train_answers, train_personas = train_data
    val_blocks, val_answers, val_personas = val_data
    test_blocks, test_answers, test_personas = test_data

    # Engine (single model for forward + backward)
    tp = vllm_kwargs.pop("tensor_parallel_size", None) or torch.cuda.device_count()
    engine = ChatVLLM(
        model_string=model_name,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        **vllm_kwargs,
    )
    _make_engine_thinking_aware(engine)
    tg.set_backward_engine(engine, override=True)

    # ── prompts ────────────────────────────────────────────────────────────
    with open(FC.PROMPTS_FILE, "r") as f:
        prompts = json.load(f)

    raw_instruction = prompts["tweet_gen"]["system_forced"]
    instruction_text = raw_instruction.replace("{max_chars}", str(max_chars))

    instruction = tg.Variable(
        instruction_text,
        role_description=(
            "System-prompt INSTRUCTION for tweet generation. "
            "Contains task framing, tone/mood guidance, and constraints. "
            "The user template (persona, PHQ-9, history) is provided "
            "separately and must NOT be duplicated here."
        ),
        requires_grad=True,
    )

    optimizer = tg.TGD(
        parameters=[instruction],
        constraints=[
            f"The instruction MUST stay concise — at most {max_instruction_words} words.",
            "Do NOT include the user template (persona, history, etc.) — "
            "those are provided separately in the user message.",
            "Focus on actionable guidance for tone, mood, content diversity, "
            "and originality that produces natural, emotionally-aligned tweets.",
        ],
        gradient_memory=0,
    )

    if num_steps is None:
        num_steps = min(50, max(1, len(train_blocks) // batch_size))

    best_val_score = -float("inf")
    best_instruction = instruction.value

    # ── training loop ──────────────────────────────────────────────────────
    for step in range(num_steps):
        batch_idx = rng.choice(
            len(train_blocks),
            size=min(batch_size, len(train_blocks)),
            replace=False,
        )
        batch_blocks  = [train_blocks[i]  for i in batch_idx]
        batch_answers = [train_answers[i] for i in batch_idx]
        batch_personas = [train_personas[i] for i in batch_idx]

        optimizer.zero_grad()
        model = tg.BlackboxLLM(engine, system_prompt=instruction)

        losses = []
        n_tweets = 0

        for tweet_block, phq9, persona in zip(batch_blocks, batch_answers,
                                              batch_personas):
            # Teacher forcing: ground-truth prior tweets as context
            for i in range(len(tweet_block)):
                context = tweet_block[:i]
                true_tweet = tweet_block[i]

                user_msg = _build_user_message_tweet(
                    context, prompts, persona, phq9,
                )
                question = tg.Variable(
                    user_msg,
                    role_description=(
                        "User template with persona, PHQ-9 score, "
                        "and tweet history"
                    ),
                    requires_grad=False,
                )
                prediction = model(question)
                predicted_tweet = parse_tweet_answers(prediction.value)

                loss_fn = tg.TextLoss(
                    _make_loss_prompt_tweet(
                        predicted_tweet, true_tweet, persona, phq9,
                    )
                )
                loss = loss_fn(prediction)
                losses.append(loss)
                n_tweets += 1

        total_loss = tg.sum(losses)
        total_loss.backward()
        optimizer.step()

        print(f"[Step {step+1}/{num_steps}]  tweets evaluated: {n_tweets}")

        if (step + 1) % 5 == 0:
            print(f"  Current instruction:\n    {instruction.value[:200]}...")

        # periodic validation on a sampled subset
        if (step + 1) % validate_every == 0 or step == num_steps - 1:
            val_score = _evaluate_tweet_instruction(
                engine, instruction.value, prompts,
                val_blocks, val_answers, val_personas,
                rng, sample_size=val_sample_size,
            )
            print(
                f"  -> Val quality score: {val_score:.2f}/10  "
                f"(best: {best_val_score:.2f})"
            )
            if val_score > best_val_score:
                best_val_score = val_score
                best_instruction = instruction.value
                print("  -> New best instruction saved!")

    # ── final test evaluation ──────────────────────────────────────────────
    test_score = _evaluate_tweet_instruction(
        engine, best_instruction, prompts,
        test_blocks, test_answers, test_personas, rng,
    )
    print(f"\nTest quality score: {test_score:.2f}/10")

    # ── save results ───────────────────────────────────────────────────────
    output_dir = os.path.dirname(file_path)

    instr_path = os.path.join(output_dir, "optimized_instruction_tweet.txt")
    with open(instr_path, "w", encoding="utf-8") as f:
        f.write(best_instruction)
    print(
        f"Best tweet instruction "
        f"(val={best_val_score:.2f}, test={test_score:.2f}) → {instr_path}"
    )

    full_path = os.path.join(output_dir, "optimized_full_prompt_tweet.txt")
    with open(full_path, "w", encoding="utf-8") as f:
        user_template = prompts["tweet_gen"]["user_template_forced"]
        f.write(
            f"=== SYSTEM PROMPT ===\n{best_instruction}\n\n"
            f"=== USER TEMPLATE ===\n{user_template}"
        )
    print(f"Full prompt → {full_path}")

    return best_instruction




# =============================== BERT Model ===============================


def create_dataset(file_path: str):
    """
    Create a dataset from a tweets_with_phq9.txt file.
    """
    tweet_blocks, true_answers = parse_tweets_with_phq9(file_path)
    permutation_blocks = np.random.permutation(len(tweet_blocks))
    permuted_tweet_blocks = [tweet_blocks[i] for i in permutation_blocks]
    permuted_true_answers = [true_answers[i] for i in permutation_blocks]

    number_of_blocks = len(permuted_tweet_blocks)
    ten_percent = number_of_blocks // 10

    validation_blocks = permuted_tweet_blocks[0:ten_percent]
    validation_true_answers = permuted_true_answers[0:ten_percent]
    training_blocks = permuted_tweet_blocks[ten_percent:-ten_percent]
    training_true_answers = permuted_true_answers[ten_percent:-ten_percent]
    test_blocks = permuted_tweet_blocks[-ten_percent:]
    test_true_answers = permuted_true_answers[-ten_percent:]

    tweet_blocks = [training_blocks, validation_blocks, test_blocks]
    true_answers = [training_true_answers, validation_true_answers, test_true_answers]

    print(f"Training blocks: {len(training_blocks)}")
    return tweet_blocks, true_answers

def split_embeddings_and_labels(rng, embeddings, labels, train_frac=0.8, val_frac=0.1):
    """
    Split embeddings and labels into train/val/test after loading.
    Uses the same 80/10/10 proportions as create_dataset.
    Parameters:
        embeddings (torch.Tensor): The embeddings to split.
        labels (torch.Tensor): The labels to split.
        train_frac (float): The fraction of the data to use for training.
        val_frac (float): The fraction of the data to use for validation.
    Returns:
        train_embs (torch.Tensor): The embeddings for the training data.
        val_embs (torch.Tensor): The embeddings for the validation data.
        test_embs (torch.Tensor): The embeddings for the test data.
        train_labels (torch.Tensor): The labels for the training data.
        val_labels (torch.Tensor): The labels for the validation data.
        test_labels (torch.Tensor): The labels for the test data.
    """
    n = len(embeddings)
    perm = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_embs = embeddings[train_idx]
    val_embs = embeddings[val_idx]
    test_embs = embeddings[test_idx]

    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.float32)
    
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    return train_embs, val_embs, test_embs, train_labels, val_labels, test_labels


def drop_high_phq9(tweet_blocks, true_answers):
    """
    Drop the tweets with a PHQ-9 score greater than 15.
    """
    filter_tweets = [true_answer <= 18 for true_answer in true_answers]
    filtered_tweet_blocks = tweet_blocks[filter_tweets]
    filtered_true_answers = true_answers[filter_tweets]
    return filtered_tweet_blocks, filtered_true_answers

def setup_BERT_model(tweet_blocks, model, device):
    """
    Setup the BERT model for the PHQ-9 assessment.
    """
    centroids = []
    for tweet_block in tweet_blocks:
        embeddings = create_embedding(model, tweet_block).to(device)
        # print("shape of embeddings: ", embeddings.shape)
        mean_v = embeddings.mean(dim=0)
        max_v = embeddings.max(dim=0)[0]
        
        var_emb = embeddings.var(dim=0)

        if torch.isnan(var_emb).any():
            print("Tweet block: ", tweet_block)
            print("NaN in var_emb")
            var_emb = 0 

        std_v = torch.sqrt(var_emb + 1e-8)
        
        window_centroid = torch.cat([mean_v, max_v, std_v], dim=0)
        centroids.append(window_centroid)
    centroids = torch.stack(centroids).to(device)
    return centroids

def train_BERT_model(embeddings_path, base_model_name, device, mental_bert: bool = False, split_seed=42):
    """
    Train the BERT model for the PHQ-9 assessment.
    Loads embeddings and labels, then performs train/val/test split (80/10/10) before training.
    """
    with open(embeddings_path, "rb") as f:
        data = torch.load(f)
    
    rng = np.random.default_rng(split_seed)

    if "embeddings" in data and "labels" in data:
        # New format: single array; split after loading
        all_embs = data["embeddings"]
        all_labels = data["labels"]
        train_embs, val_embs, test_embs, train_labels, val_labels, test_labels = split_embeddings_and_labels(
            rng, all_embs, all_labels, train_frac=0.8, val_frac=0.1
        )
    else:
        # Legacy format: pre-split arrays
        train_embs = data["train_embs"]
        val_embs = data["val_embs"]
        test_embs = data["test_embs"]
        train_labels = data["train_labels"]
        val_labels = data["val_labels"]
        test_labels = data["test_labels"]

    train_embs, train_labels = drop_high_phq9(train_embs.to(device), train_labels.to(device))
    val_embs, val_labels = drop_high_phq9(val_embs.to(device), val_labels.to(device))
    test_embs, test_labels = drop_high_phq9(test_embs.to(device), test_labels.to(device))

    print(f"Training blocks: {len(train_embs)}, val: {len(val_embs)}, test: {len(test_embs)}")

    nn_model = neural_net_BERT(mentalbert=mental_bert).to(device)
    nn_model, best_loss, epoch_history = train_bert(nn_model, train_embs, train_labels, val_embs, val_labels, device)
    print("Testing model....")
    test_loss = evaluate_bert(nn_model, test_embs, test_labels, device, mae=True)
    print(f"Test Loss: {test_loss}\n\n")

    
    save_dir = os.path.join("data", "test", base_model_name, "sbertmodel")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "regressor.pt")
    torch.save(nn_model, save_path)
    print(f"Sbert regression saved to {save_path}")

    metrics_path = os.path.join(save_dir, "performance.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_val_loss": float(best_loss),
                "test_loss": float(test_loss),
                "epochs": epoch_history,  # list of {epoch, train_loss, val_loss}
            },
            f,
            indent=2,
        )

    return nn_model, best_loss, test_loss


class neural_net_BERT(nn.Module): 
    """
    Setup the neural network for the PHQ-9 assessment.
    """
    def __init__(self, mentalbert: bool = False, dropout_rate=0.2):
        if mentalbert:
            input_size = 768*3
        else:
            input_size = 384*3
        super(neural_net_BERT, self).__init__()
        self.model = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1), # 1 output for the PHQ-9 score
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)
    
def train_bert(model, 
                train_data, 
                train_labels, 
                val_data, 
                val_labels, 
                device, 
                epochs=30, 
                batch_size=8, 
                learning_rate=0.0001,
                patience=5):
    """
    Train the neural network for the PHQ-9 assessment.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=3)

    criterion = nn.HuberLoss(delta=1.0)
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)

    best_loss = float('inf')
    epoch_history = {"epoch": [], "train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        train_loss = 0
        for i in range(0, len(train_data), batch_size):
            inputs = train_data[i:i+batch_size]
            labels = train_labels[i:i+batch_size]     

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1) # squeeze the last dimension to get the score
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()*inputs.size(0)

        train_loss /= len(train_data)
        val_loss = evaluate_bert(model, val_data, val_labels, device)
        scheduler.step(val_loss)
    

        epoch_history["epoch"].append(epoch+1)
        epoch_history["train_loss"].append(train_loss)
        epoch_history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss} Validation Loss: {val_loss}\n\n")
  
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = copy.deepcopy(model)
    
    return best_model_state, best_loss, epoch_history

def evaluate_bert(model, test_data, test_labels, device, mae=False):
    """
    Evaluate the BERT model for the PHQ-9 assessment.
    """
    model.eval()
    if mae:
        criterion = nn.MSELoss()
    else:
        criterion = nn.HuberLoss(delta=1.0)

    test_data = torch.as_tensor(test_data, dtype=torch.float32).to(device)
    test_labels = torch.as_tensor(test_labels, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(test_data).squeeze(-1)
        loss = criterion(outputs, test_labels)
    return loss.item()

def save_embeddings_for_file(file_path: str, base_model_name: str, device, mentalbert: bool = False, out_dir=None):
    """
    - Parses tweets_with_phq9.txt into all blocks (no train/val/test split)
    - Encodes each block with SBERT
    - Saves single embeddings + labels to disk; split is done at train time after loading
    """
    tweet_blocks, true_answers = parse_tweets_with_phq9(file_path)
    sbert_model = generate_sbert_model(mentalbert=mentalbert).to(device)

    print("Encoding all blocks...")
    all_embs = setup_BERT_model(tweet_blocks, sbert_model, device)
    all_labels = torch.tensor(true_answers, dtype=torch.float32)

    if mentalbert:
        dir_name = "mentalbert_embeddings"
    else:
        dir_name = "sbert_embeddings"

    if out_dir is None:
        out_dir = os.path.join("data", "test", base_model_name, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    torch_path = os.path.join(out_dir, "embeddings_and_labels.pt")
    torch.save({"embeddings": all_embs, "labels": all_labels}, torch_path)
    print(f"Saved embeddings + labels to {torch_path} (n={len(all_embs)}); split will be done at train time.")
    

def _generate_file_path(base_dir: str, target_filename: str = "tweets_with_phq9.txt") -> List[str]:
    """
    Generate all file paths for a given condition directory by finding 
    all seed folders that contain the target file.
    """
    # Construct a wildcard pattern, e.g.: data/..._inter/seed_*/tweets_with_phq9.txt
    search_pattern = os.path.join(base_dir, "seed_*", target_filename)
    
    # glob.glob returns a list of all file paths that match the pattern
    found_paths = glob.glob(search_pattern)
    
    # Optional: sort them so seed_55 comes before seed_65, etc.
    return sorted(found_paths)
    
if __name__ == "__main__":
    create_new_embeddings = False
    mental_bert = True
    file_paths = []
    
    # Fixed the loop to iterate over a list of strings
    for inter in ["no_inter", "inter"]:
        base_dir = f"data/test_post/Qwen_Qwen3.5-27B/temp_0.8_top_p_0.6_cp_10_{inter}"
        
        # This will return a list of paths for all seeds in the current base_dir
        paths_for_condition = _generate_file_path(base_dir)
        
        # Use .extend() instead of .append() to add the items to the flat list
        file_paths.extend(paths_for_condition)

    for file_path in file_paths:
        print(file_path)

    base_model_name = "Qwen_Qwen3.5-27B"
    model_name = "Qwen/Qwen3.5-27B"
    run_prompt_optimizer = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if run_prompt_optimizer:
        call_optimizer_phq9(file_paths, model_name=model_name, batch_size=4)
    else:
        if create_new_embeddings:
            save_embeddings_for_file(file_path, base_model_name, device, mentalbert=mental_bert)
        
        if mental_bert:
            dir_name = "mentalbert_embeddings"
        else:
            dir_name = "sbert_embeddings"

        embeddings_path = os.path.join("data", "test", base_model_name, dir_name, "embeddings_and_labels.pt")
        train_BERT_model(embeddings_path, base_model_name, device, mental_bert=mental_bert)