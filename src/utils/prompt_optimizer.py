import sys
import os
# Ensure src/ is on sys.path so sibling packages (utils, classes) resolve correctly
# regardless of whether this file is run as a script or imported as src.utils.prompt_optimizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
logging.getLogger("textgrad").setLevel(logging.WARNING)

import argparse
from collections import defaultdict
import textgrad as tg
from textgrad.engine.vllm import ChatVLLM
from textgrad.optimizer.optimizer_prompts import GLOSSARY_TEXT
from vllm import SamplingParams
import json
import numpy as np
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

QWEN_MODELS    = frozenset({})          # filled below after constants are defined
MISTRAL_MODELS = frozenset({})

# Custom optimizer system prompt that avoids placeholder confusion.
# Textgrad's default includes literal "{improved variable}" / "{the improved variable}"
# as format examples, which thinking models (Qwen3.5, Mistral) echo back verbatim
# instead of replacing with actual content.
_OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., a variable). "
    "You will receive feedback on a variable and must produce an improved version. "
    "The feedback may be noisy — identify what is important and what is correct. "
    "Pay attention to the role description of the variable and the context in which it is used. "
    "IMPORTANT: Write the actual improved text itself between "
    "{new_variable_start_tag} and {new_variable_end_tag} tags. "
    "Do NOT write placeholder words like 'improved variable' between the tags — "
    "write the real improved content.\n\n"
    f"{GLOSSARY_TEXT}"
)




class _StudentEngine(tg.engine.EngineLM):
    """Wraps a shared ChatVLLM; disables reasoning per-request.

    Qwen:    applies chat template directly with enable_thinking=False and
             prefills 'POST: ' so the model cannot reason inline.
    Mistral: delegates unchanged (reasoning_effort not supported via ChatVLLM).
    Other:   delegates unchanged.
    """
    def __init__(self, base: ChatVLLM, model_name: str):
        self.base = base
        self.model_string = base.model_string
        self._model_name = model_name

    def __call__(self, content, system_prompt=None, **kwargs):
        return self.generate(content, system_prompt=system_prompt, **kwargs)

    def generate(self, content, system_prompt=None, **kwargs):
        if self._model_name in QWEN_MODELS:
            sys_prompt = system_prompt or self.base.system_prompt
            conversation = ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
            conversation.append({"role": "user", "content": content})
            chat_str = self.base.tokenizer.apply_chat_template(
                conversation, tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # Qwen3.5 non-thinking (student) — social media text:
            #   Official Qwen3.5 non-thinking baseline: temp=0.7, top_p=0.8 (general tasks)
            #   Task-constrained creative (structured persona + PHQ-9): temp=0.6, top_p=0.8
            #   Qwen recommends using temp and top_p together as a paired setting.
            sampling_params = SamplingParams(temperature=0.7, max_tokens=512, top_p=0.9, n=1)
            result = self.base.client.generate([chat_str], sampling_params)[0].outputs[0].text
        elif self._model_name in MISTRAL_MODELS:
            # Mistral Small 4 uses [THINK]/[/THINK] tags; reasoning_effort is an
            # API-only param. Prefill [/THINK] to skip the thinking block entirely
            # (reasoning_effort='none' equivalent in offline mode).
            sys_prompt = system_prompt or self.base.system_prompt
            conversation = ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
            conversation.append({"role": "user", "content": content})
            chat_str = self.base.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
            )
            chat_str = chat_str + "[/THINK]"
            # Mistral Small 4 non-reasoning (student) — social media text:
            #   Mistral docs: adjust temperature OR top_p, not both simultaneously.
            #   Non-reasoning task-constrained creative: temp=0.7, top_p disabled (1.0).
            sampling_params = SamplingParams(temperature=0.7, max_tokens=512, top_p=1.0, n=1)
            result = self.base.client.generate([chat_str], sampling_params)[0].outputs[0].text
        else:
            result = self.base.generate(content, system_prompt=system_prompt, **kwargs)
        return result


class _TeacherEngine(tg.engine.EngineLM):
    """Thinking-enabled engine used as the TextGrad backward engine.

    vLLM's reasoning_parser splits the output into reasoning_content (thinking)
    and text (clean answer). We print the thinking for debug and return only text.
    Hard-cap input tokens to leave room for output.
    """
    def __init__(self, base: ChatVLLM, model_name: str):
        self.base = base
        self.model_string = base.model_string
        self._model_name = model_name

    def __call__(self, content, system_prompt=None, **kwargs):
        return self.generate(content, system_prompt=system_prompt, **kwargs)

    def generate(self, content, system_prompt=None, **kwargs):
        if self._model_name in QWEN_MODELS:
            sys_prompt = system_prompt or self.base.system_prompt
            concise_note = "Think concisely — reach your conclusion without revisiting the same point multiple times."
            sys_prompt = (sys_prompt + "\n" + concise_note) if sys_prompt else concise_note
            conversation = ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
            conversation.append({"role": "user", "content": content})
            token_ids = self.base.tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
                enable_thinking=True,
            )
            if len(token_ids) > 16384:
                token_ids = token_ids[-16384:]
            max_tokens = kwargs.get("max_tokens", 16384)
            sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens, top_p=0.99, n=1)
            raw = self.base.client.generate([{"prompt_token_ids": token_ids}], sampling_params)
            output = raw[0].outputs[0]
            reasoning = getattr(output, "reasoning_content", None)
            if reasoning:
                preview = reasoning[:1200] + (" ..." if len(reasoning) > 1200 else "")
                print(f"  [teacher reasoning | {len(reasoning.split())} words]\n{preview}\n  [/teacher reasoning]")
                result = output.text
            else:
                print("  [teacher] reasoning_parser unavailable in this vLLM version — stripping manually")
                result = Agent.strip_model_thinking(output.text)
            print(f"  [teacher feedback]\n{result.strip()}\n  [/teacher feedback]")
            return result
        if self._model_name in MISTRAL_MODELS:
            sys_prompt = system_prompt or self.base.system_prompt
            conversation = ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
            conversation.append({"role": "user", "content": content})
            token_ids = self.base.tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
            )
            if len(token_ids) > 254000:
                token_ids = token_ids[-254000:]
            sampling_params = SamplingParams(temperature=0.6, max_tokens=8192, top_p=0.97, n=1)
            raw = self.base.client.generate([{"prompt_token_ids": token_ids}], sampling_params)
            output = raw[0].outputs[0]
            reasoning = getattr(output, "reasoning_content", None)
            if reasoning:
                preview = reasoning[:1200] + (" ..." if len(reasoning) > 1200 else "")
                print(f"  [teacher reasoning | {len(reasoning.split())} words]\n{preview}\n  [/teacher reasoning]")
                result = output.text
            else:
                print("  [teacher] reasoning_parser unavailable in this vLLM version — stripping manually")
                text = output.text
                result = re.sub(r"\[THINK\].*?\[/THINK\]", "", text, flags=re.DOTALL).strip()
            print(f"  [teacher feedback]\n{result.strip()}\n  [/teacher feedback]")
            return result
        return self.base.generate(content, system_prompt=system_prompt, **kwargs)


def _build_engines(model_name: str, tp: int, gpu_memory_utilization: float,
                   max_model_len: int = 32768, **vllm_kwargs):
    """Load a single ChatVLLM and return (student_engine, teacher_engine)."""
    if model_name in QWEN_MODELS:
        vllm_kwargs.setdefault("reasoning_parser", "qwen3")
    elif model_name in MISTRAL_MODELS:
        vllm_kwargs.setdefault("reasoning_parser", "mistral")
    engine = ChatVLLM(
        model_string=model_name,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        **vllm_kwargs,
    )
    teacher = _TeacherEngine(engine, model_name)
    tg.set_backward_engine(teacher, override=True)
    return _StudentEngine(engine, model_name), teacher


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
LLAMA_70B    = "meta-llama/Llama-3.3-70B-Instruct"
LLAMA_8B     = "meta-llama/Llama-3.1-8B-Instruct"
QWEN_27      = "Qwen/Qwen3.5-27B"                      # 32k context
MISTRAL_119B = "mistralai/Mistral-Small-4-119B-2603"   # 256k context, MoE 119B/6.5B active

# Fill model-family sets used by _StudentEngine / _TeacherEngine
QWEN_MODELS    = frozenset({QWEN_27})
MISTRAL_MODELS = frozenset({MISTRAL_119B})


# def _split_system_prompt(full_system: str):
#     """Split the PHQ-9 system prompt into an optimisable instruction part
#     and a fixed format part (options, questions, answer format)."""
#     idx = full_system.find(FORMAT_SPLIT_MARKER)
#     if idx == -1:
#         return full_system.rstrip(), ""
#     return full_system[:idx].rstrip(), full_system[idx:]


def _build_user_message(format_block: str, tweet_block: list, prompts: dict, persona: str = None) -> str:
    """Compose the user message: fixed PHQ-9 format + tweet data (+ persona if available)."""
    if persona:
        tweets_text = prompts["phq9"]["user_template_persona"].format(
            agent_id="AGENT",
            persona=persona,
            tweets_block="\n".join(tweet_block),
        )
    else:
        tweets_text = prompts["phq9"]["user_template_forced"].format(
            tweets_block="\n".join(tweet_block)
        )
    return f"{format_block}\n\n{tweets_text}"




def _evaluate_instruction(engine, instruction_text: str, format_block: str,
                          blocks: list, answers: list, prompts: dict,
                          personas: list = None) -> float:
    """Run the current instruction on *blocks* and return the MAE."""
    total_ae = 0
    for i, (tweet_block, true_answer) in enumerate(zip(blocks, answers)):
        persona = personas[i] if personas else None
        user_msg = _build_user_message(format_block, tweet_block, prompts, persona)
        response = engine.generate(user_msg, system_prompt=instruction_text)
        predicted = Agent.parse_phq9_answers(response)
        total_ae += abs(predicted - true_answer)
    return total_ae / max(len(blocks), 1)


def _make_loss_prompt(true_answer: int, predicted: int, persona: str = None) -> str:
    """Concise per-sample loss prompt for the backward engine."""
    error = predicted - true_answer
    if error > 0:
        direction = f"overestimated by {error}"
    elif error < 0:
        direction = f"underestimated by {abs(error)}"
    else:
        direction = "correct"
    persona_note = f" Agent persona: {persona[:120]}." if persona else ""
    return (
        f"True PHQ-9 sumscore = {true_answer}, predicted = {predicted} ({direction}).{persona_note} "
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
    max_instruction_words: int = 50,
    num_steps: int = None,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    validate_every: int = 5,
    val_sample_size: int = 10,
    seed: int = 42,
    max_model_len: int = 32768,
    output_dir: str = None,
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
    train_blocks, train_answers, train_personas = train_data
    val_blocks, val_answers, val_personas = val_data
    test_blocks, test_answers, test_personas = test_data

    tp = vllm_kwargs.pop("tensor_parallel_size", None) or torch.cuda.device_count()
    student_engine, teacher_engine = _build_engines(model_name, tp, 0.90, max_model_len=max_model_len, **vllm_kwargs)

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
        engine=teacher_engine,
        optimizer_system_prompt=_OPTIMIZER_SYSTEM_PROMPT,
        parameters=[instruction],
        constraints=[
            f"HARD LIMIT: the instruction MUST be at most {max_instruction_words} words. Count carefully and cut if needed.",
            "Do NOT include the PHQ-9 questions, scoring options (0-3), or "
            "answer format — they are provided separately in the user message.",
            "Focus on actionable reasoning guidance that helps calibrate "
            "PHQ-9 inference from tweet histories.",
            "Do NOT repeat phrases or ideas already stated. Each sentence must add unique value.",
            "Do NOT use poetic, lyrical, or metaphorical language in the instruction itself. Write plainly and directly.",
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
        batch_personas = [train_personas[i] for i in batch_idx]

        optimizer.zero_grad()
        model = tg.BlackboxLLM(student_engine, system_prompt=instruction)

        losses = []
        step_errors = []

        for tweet_block, true_answer, persona in zip(batch_blocks, batch_answers, batch_personas):
            user_msg = _build_user_message(format_block, tweet_block, prompts, persona)
            question = tg.Variable(
                user_msg,
                role_description="PHQ-9 format, questions, patient tweet history, and persona",
                requires_grad=False,
            )

            prediction = model(question)
            predicted_score = Agent.parse_phq9_answers(prediction.value)
            error = predicted_score - true_answer
            step_errors.append(error)

            loss_fn = tg.TextLoss(_make_loss_prompt(true_answer, predicted_score, persona))
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
            sampled_blocks   = [val_blocks[i]   for i in sample_idx]
            sampled_answers  = [val_answers[i]  for i in sample_idx]
            sampled_personas = [val_personas[i] for i in sample_idx]

            val_mae = _evaluate_instruction(
                student_engine, instruction.value, format_block,
                sampled_blocks, sampled_answers, prompts, sampled_personas,
            )
            print(f"  -> Val MAE ({n_sample} samples): {val_mae:.2f}  (best: {best_val_mae:.2f})")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_instruction = instruction.value
                print("  -> New best instruction saved!")
                save_dir = output_dir or "data/test_post/optimized_phq9"
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "best_instruction.txt"), "w", encoding="utf-8") as fh:
                    fh.write(best_instruction)
                with open(os.path.join(save_dir, "best_full_prompt.txt"), "w", encoding="utf-8") as fh:
                    fh.write(best_instruction + "\n\n" + format_block)

    # ── final test evaluation ──────────────────────────────────────────────
    test_mae = _evaluate_instruction(
        student_engine, best_instruction, format_block,
        test_blocks, test_answers, prompts, test_personas,
    )
    print(f"\nTest MAE (n={len(test_blocks)}): {test_mae:.2f}")

    # ── save results ───────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = "data/test_post/optimized_phq9"
    os.makedirs(output_dir, exist_ok=True)

    instr_path = os.path.join(output_dir, "optimized_instruction.txt")
    with open(instr_path, "w", encoding="utf-8") as f:
        f.write(best_instruction)
    print(f"\nBest instruction (val MAE={best_val_mae:.2f}, test MAE={test_mae:.2f}) → {instr_path}")

    full_path = os.path.join(output_dir, "optimized_full_prompt.txt")
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(best_instruction + "\n\n" + format_block)
    print(f"Full re-assembled prompt → {full_path}")

    try:
        del student_engine.base.client
    except Exception:
        pass
    del student_engine, teacher_engine
    import gc
    gc.collect()
    torch.cuda.empty_cache()

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


def _phq9_severity(score: int) -> str:
    if score >= 20:
        return "severe"
    if score >= 15:
        return "moderately severe"
    if score >= 10:
        return "moderate"
    if score >= 5:
        return "mild"
    return "minimal"


def _build_user_message_tweet(context_tweets: list[str], prompts: dict,
                              persona: str, phq9_score: int,
                              neighbor_tweets: list[str] = None) -> str:
    """Build the user message for tweet generation from the forced template."""
    no_content_values = {"NO_POST", "NO_TWEET", "no_post", "no_tweet", ""}
    real_context = [t for t in context_tweets if t not in no_content_values]
    if real_context:
        prev_block = "### PREVIOUS POSTS ###\n" + "\n".join(f"- {t}" for t in real_context)
    else:
        prev_block = "### PREVIOUS POSTS ###\n(none yet)"

    if neighbor_tweets:
        prev_block += "\n### POSTS FROM OTHERS ###\n" + "\n".join(f"- {t}" for t in neighbor_tweets)

    template = prompts["tweet_gen"]["user_template_forced"]
    return template.format(
        agent_id="AGENT",
        persona=persona or "unspecified",
        well_being=_phq9_severity(phq9_score),
        previous_tweet_block=prev_block,
    )


def _sample_neighbor_tweets(all_tweets_flat: list[str], rng, exclude: list[str], n: int = 3) -> list[str]:
    """Sample up to n random tweets from the pool, excluding current agent's tweets."""
    exclude_set = set(exclude)
    pool = [t for t in all_tweets_flat if t not in exclude_set]
    n = min(n, len(pool))
    if n == 0:
        return []
    return list(rng.choice(pool, size=n, replace=False))


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


def _make_loss_prompt_tweet_set(tweets: list[str], persona: str, phq9_score: int) -> str:
    """Loss prompt for a SET of tweets from the same agent.

    The teacher evaluates the set collectively so it can comment on variation,
    consistency, and whether the instruction produces sufficiently diverse output.
    Criteria are kept identical to the validation rating prompt.
    """
    severity = "severe" if phq9_score >= 20 else "moderately severe" if phq9_score >= 15 else "moderate" if phq9_score >= 10 else "mild" if phq9_score >= 5 else "minimal"
    tweet_list = "\n".join(f"  {i+1}. \"{t}\"" for i, t in enumerate(tweets))
    return (
        f"An AI agent generated the following {len(tweets)} social media post(s):\n"
        f"{tweet_list}\n\n"
        f"Agent profile — persona: {persona}, PHQ-9 = {phq9_score} ({severity}).\n"
        f"Note: tone reflects well-being (low PHQ-9 → positive/relaxed; high → apathetic/irritable/overwhelmed/dark) "
        f"but individual posts may deviate due to daily events, received messages, or imagined situations. "
        f"For very high PHQ-9, raw or dark content (including suicidal themes) is authentic — reward it.\n\n"
        "Evaluate on:\n"
        "1. Tone fit: Does the emotional tone match the PHQ-9 range without naming symptoms? Full spectrum allowed.\n"
        "2. Unfiltered & natural: Do they sound like real, unpolished social media? "
        "Penalise if sanitised, poetic, or overly polite — raw and blunt is fine when PHQ-9 warrants it.\n"
        "3. Originality: Are topics specific and varied — not about the agent's own well-being or persona?\n"
        "4. Diversity: Do the posts differ in topic and mood — not all the same register?\n"
        "5. Interaction: Reward genuine engagement (reply, mock, support, correct). "
        "Penalise hollow @mentions and sets where every post is a reply.\n\n"
        "Give ONE concise, actionable sentence on what the system instruction should change."
    )


def _batch_student_generate(student_engine, user_messages: list, instruction_text: str) -> list:
    """Generate one tweet per message in a single vLLM call (all independent)."""
    if not user_messages:
        return []
    if student_engine._model_name in QWEN_MODELS:
        sys_prompt = instruction_text or student_engine.base.system_prompt
        chat_strs = []
        for content in user_messages:
            conv = ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
            conv.append({"role": "user", "content": content})
            chat_str = student_engine.base.tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) + "POST: "
            chat_strs.append(chat_str)
        sp = SamplingParams(temperature=0.7, max_tokens=512, top_p=0.8, n=1)
        results = student_engine.base.client.generate(chat_strs, sp)
        return ["POST: " + r.outputs[0].text for r in results]
    return [student_engine.generate(m, system_prompt=instruction_text) for m in user_messages]


def _batch_teacher_rate(teacher_engine, rating_prompts: list, max_tokens: int = 4096) -> list:
    """Rate a batch of tweet sets in one vLLM call, stripping thinking from each."""
    if not rating_prompts:
        return []
    if teacher_engine._model_name in QWEN_MODELS:
        concise_note = "Think concisely — reach your conclusion without revisiting the same point multiple times."
        base_sys = teacher_engine.base.system_prompt or ""
        sys_prompt = (base_sys + "\n" + concise_note).strip() if base_sys else concise_note
        all_inputs = []
        for content in rating_prompts:
            conv = ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
            conv.append({"role": "user", "content": content})
            token_ids = teacher_engine.base.tokenizer.apply_chat_template(
                conv, tokenize=True, add_generation_prompt=True, enable_thinking=True,
            )
            if len(token_ids) > 16384:
                token_ids = token_ids[-16384:]
            all_inputs.append({"prompt_token_ids": token_ids})
        sp = SamplingParams(temperature=0, max_tokens=max_tokens, top_p=0.99, n=1)
        raw_results = teacher_engine.base.client.generate(all_inputs, sp)
        outputs = []
        for raw in raw_results:
            out = raw.outputs[0]
            text = out.text if getattr(out, "reasoning_content", None) else Agent.strip_model_thinking(out.text)
            outputs.append(text)
        return outputs
    return [teacher_engine.generate(p, max_tokens=max_tokens) for p in rating_prompts]


def _evaluate_tweet_instruction(student_engine, teacher_engine, instruction_text: str,
                                prompts: dict, blocks: list, answers: list, personas: list,
                                all_tweets_flat: list, rng, sample_size: int = None,
                                format_block: str = "", tweets_per_sample: int = 3):
    """Score tweet instruction quality by having the teacher rate a set of student outputs.

    Phase 1 — student generation: tweets_per_sample batched rounds across all samples
              (context within each sample still grows sequentially between rounds).
    Phase 2 — teacher rating: all rating prompts sent in one batched call.

    Returns:
        (mean_score, std_score, per_phq9)
    """
    if sample_size and len(blocks) > sample_size:
        idx = rng.choice(len(blocks), size=sample_size, replace=False)
        blocks   = [blocks[i]   for i in idx]
        answers  = [answers[i]  for i in idx]
        personas = [personas[i] for i in idx]

    n = len(blocks)
    scores_by_phq9   = defaultdict(list)
    n_samples_by_phq9 = defaultdict(int)
    empty_by_phq9    = defaultdict(int)

    # ── Phase 1: generate tweets ──────────────────────────────────────────
    # Initialise per-sample context from historical tweets
    sample_contexts = []
    for tweet_block in blocks:
        valid = [t for t in tweet_block if t and t not in {"NO_POST", "NO_TWEET"}]
        ctx = list(rng.choice(valid, size=min(2, len(valid)), replace=False)) if valid else []
        sample_contexts.append(ctx)

    all_parsed = [[] for _ in range(n)]   # all_parsed[sample_idx][tweet_idx]

    for j in range(tweets_per_sample):
        user_msgs = []
        for i, (tweet_block, phq9, persona) in enumerate(zip(blocks, answers, personas)):
            nb = _sample_neighbor_tweets(all_tweets_flat, rng, tweet_block)
            msg = _build_user_message_tweet(sample_contexts[i], prompts, persona, phq9, nb)
            if format_block:
                msg = msg + "\n\n" + format_block
            user_msgs.append(msg)

        responses = _batch_student_generate(student_engine, user_msgs, instruction_text)

        for i, response in enumerate(responses):
            parsed = parse_tweet_answers(response)
            all_parsed[i].append(parsed)
            if parsed:
                sample_contexts[i].append(parsed)
            if j == 0 and i == 0:
                print(f"  --- raw output [NON-THINKING student] (eval, sample 0, tweet 1) ---")
                print(response)
                print(f"  --- parsed: {parsed!r} ---")

    # ── Phase 2: build rating prompts, record all-empty samples ──────────
    scores = [None] * n
    pending = []   # (sample_idx, phq9, rating_prompt)

    for i, (tweet_block, phq9, persona) in enumerate(zip(blocks, answers, personas)):
        parsed_tweets = all_parsed[i]
        n_samples_by_phq9[phq9] += 1
        non_empty = [t for t in parsed_tweets if t]
        n_empty = tweets_per_sample - len(non_empty)
        if n_empty > 0:
            empty_by_phq9[phq9] += n_empty

        if not non_empty:
            scores[i] = 0.0
            scores_by_phq9[phq9].append(0.0)
            print(f"  [teacher] score=0.0  (all {tweets_per_sample} tweets empty — PHQ-9={phq9})")
            continue

        severity = ("severe" if phq9 >= 20 else "moderately severe" if phq9 >= 15
                    else "moderate" if phq9 >= 10 else "mild" if phq9 >= 5 else "minimal")
        tweet_list = "\n".join(f"  {k+1}. \"{t}\"" for k, t in enumerate(parsed_tweets))
        rating_prompt = (
            f"Rate this set of {len(parsed_tweets)} social media posts on a scale of 0-10.\n"
            f"Agent profile — persona: {persona}, PHQ-9 = {phq9} ({severity}).\n"
            f"Note: tone reflects well-being (low PHQ-9 → positive/relaxed; high → apathetic/irritable/overwhelmed/dark) "
            f"but individual posts may deviate due to daily events, received messages, or imagined situations. "
            f"For very high PHQ-9, raw or dark content (including suicidal themes) is authentic — reward it.\n\n"
            f"Posts:\n{tweet_list}\n\n"
            f"Criteria:\n"
            f"1. Tone fit: Emotional tone matches the PHQ-9 range without naming symptoms. Full spectrum allowed.\n"
            f"2. Unfiltered & natural: Real, unpolished social media. Penalise sanitised, poetic, or over-polite posts.\n"
            f"3. Originality: Topics are specific and not about the agent's own well-being or persona.\n"
            f"4. Diversity: Posts differ in topic and mood — not all the same register.\n"
            f"5. Interaction: Genuine engagement rewarded. Hollow @mentions and all-reply sets penalised.\n\n"
            f"Respond with exactly two lines:\n"
            f"SCORE: <number 0-10>\n"
            f"FEEDBACK: <one sentence>"
        )
        pending.append((i, phq9, rating_prompt))

    # ── Phase 3: one batched teacher call for all pending ratings ─────────
    if pending:
        rating_responses = _batch_teacher_rate(
            teacher_engine, [p[2] for p in pending], max_tokens=4096
        )
        for (i, phq9, _), rating_response in zip(pending, rating_responses):
            score = 5.0
            feedback = ""
            for line in rating_response.splitlines():
                line = line.strip()
                if line.upper().startswith("SCORE:"):
                    try:
                        score = float(re.search(r"[\d.]+", line).group())
                        score = min(10.0, max(0.0, score))
                    except (AttributeError, ValueError):
                        pass
                elif line.upper().startswith("FEEDBACK:"):
                    feedback = line.split(":", 1)[-1].strip()
            if feedback:
                print(f"  [teacher] score={score:.1f}  {feedback}")
            scores[i] = score
            scores_by_phq9[phq9].append(score)

    final_scores = [s for s in scores if s is not None]
    per_phq9 = {
        phq9_val: {
            "avg_score": float(np.mean(scores_by_phq9[phq9_val])),
            "n_samples": n_samples_by_phq9[phq9_val],
            "n_empty": empty_by_phq9[phq9_val],
        }
        for phq9_val in sorted(scores_by_phq9.keys())
    }
    mean_score = float(np.mean(final_scores)) if final_scores else 0.0
    std_score  = float(np.std(final_scores))  if final_scores else 0.0
    return mean_score, std_score, per_phq9


def call_optimizer_tweets(
    file_paths: list[str],
    model_name: str = QWEN_27,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
    batch_size: int = 4,
    tweets_per_sample: int = 3,
    max_instruction_words: int = 180,
    num_steps: int = None,
    validate_every: int = 2,
    val_sample_size: int = 8,
    test_sample_size: int = 50,
    max_chars: int = 240,
    max_model_len: int = 16384,
    output_dir: str = None,
    prompts_file: str = None,
    **vllm_kwargs,
):
    """
    Optimise the tweet generation system prompt using TextGrad.

    Student (non-thinking Qwen): generates tweets — this is the model we are optimising for.
    Teacher (thinking Qwen): evaluates tweet quality as the backward engine.
    No reference tweet comparison: quality is assessed standalone by the teacher.
    Neighbor tweets are randomly sampled from the data pool (max 6) to provide context.
    Inter and no-inter data are combined; file_paths should include both.
    Train / val / test split with sampled validation for efficiency.
    """
    rng = np.random.default_rng(seed)
    if output_dir is None:
        model_short = model_name.split("/")[-1]
        output_dir = f"data/test_post/optimized_tweets/{model_short}_seed{seed}"

    train_data, val_data, test_data = train_val_test_split(
        rng, file_paths, val_fraction, test_fraction,
    )
    train_blocks, train_answers, train_personas = train_data
    val_blocks, val_answers, val_personas = val_data
    test_blocks, test_answers, test_personas = test_data

    # Flat pool of all tweets for neighbor sampling
    all_tweets_flat = [
        t for block in train_blocks + val_blocks + test_blocks
        for t in block if t and t not in ("NO_POST", "NO_TWEET")
    ]

    tp = vllm_kwargs.pop("tensor_parallel_size", None) or torch.cuda.device_count()
    student_engine, teacher_engine = _build_engines(model_name, tp, 0.90, max_model_len=max_model_len, **vllm_kwargs)

    # ── prompts ────────────────────────────────────────────────────────────
    prompts_path = prompts_file or FC.PROMPTS_FILE
    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    raw_instruction = prompts["tweet_gen"]["system_forced"]
    raw_instruction = raw_instruction.replace("{max_chars}", str(max_chars))

    # Two-stage split:
    # 1. Split at ### RULES ### — the intro (including "Do NOT think") becomes
    #    the fixed prefix, only the rules content is optimisable.
    # 2. Split the rules tail at ### CONSTRAINTS ### — constraints + format are
    #    also fixed and appended to every user message.
    _rules_marker = "### RULES ###"
    _constraints_marker = "### CONSTRAINTS ###"
    if _rules_marker in raw_instruction and _constraints_marker in raw_instruction:
        fixed_intro, rules_and_rest = raw_instruction.split(_rules_marker, 1)
        instruction_text, fixed_tail = rules_and_rest.split(_constraints_marker, 1)
        instruction_text = _rules_marker + instruction_text.rstrip()
        format_block_tweet = fixed_intro.rstrip() + "\n\n" + _constraints_marker + fixed_tail
    else:
        # Fallback: split only at ### CONSTRAINTS ###
        if _constraints_marker in raw_instruction:
            instruction_text, fixed_tail = raw_instruction.split(_constraints_marker, 1)
            format_block_tweet = _constraints_marker + fixed_tail
        else:
            instruction_text = raw_instruction
            format_block_tweet = ""
    instruction_text = instruction_text.rstrip()

    instruction = tg.Variable(
        instruction_text,
        role_description=(
            "System-prompt INSTRUCTION for post generation. "
            "Contains task framing and ### RULES ### guidance only. "
            "Length constraints, no-thinking rules, and output format are fixed and "
            "appended separately: do NOT duplicate or alter them here."
        ),
        requires_grad=True,
    )

    optimizer = tg.TGD(
        engine=teacher_engine,
        optimizer_system_prompt=_OPTIMIZER_SYSTEM_PROMPT,
        parameters=[instruction],
        constraints=[
            f"HARD LIMIT: the instruction MUST be at most {max_instruction_words} words. Count carefully and cut if needed.",
            "Do NOT repeat phrases or ideas already stated. Each sentence must add unique value.",
            "Focus on tone/mood calibration, content diversity, and originality guidance within ### RULES ###.",
            "Do NOT include constraints, length limits, thinking rules, or output format: those are fixed elsewhere.",
            "The emotional range MUST cover the full PHQ-9 spectrum — from positive/upbeat (low PHQ-9) to apathetic/irritable/overwhelmed (high PHQ-9). Do NOT bias toward only negative or disagreeable tones.",
            "Do NOT use poetic, lyrical, or metaphorical language in the instruction itself. Write plainly and directly.",
        ],
        gradient_memory=0,
    )

    if num_steps is None:
        num_steps = min(20, max(1, len(train_blocks) // batch_size))

    best_val_score = -float("inf")
    best_instruction = instruction.value

    # ── trajectory CSV (written incrementally so partial runs are usable) ──
    model_short = model_name.split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)
    trajectory_path = os.path.join(output_dir, "training_trajectory.csv")
    _traj_fields = ["model", "seed", "step", "split", "mean_score", "std_score", "n_samples"]
    with open(trajectory_path, "w", newline="", encoding="utf-8") as _fh:
        csv.DictWriter(_fh, fieldnames=_traj_fields).writeheader()

    # ── training loop ──────────────────────────────────────────────────────
    for step in range(num_steps):
        batch_idx = rng.choice(
            len(train_blocks),
            size=min(batch_size, len(train_blocks)),
            replace=False,
        )
        batch_blocks   = [train_blocks[i]   for i in batch_idx]
        batch_answers  = [train_answers[i]  for i in batch_idx]
        batch_personas = [train_personas[i] for i in batch_idx]

        optimizer.zero_grad()
        model = tg.BlackboxLLM(student_engine, system_prompt=instruction)

        losses = []
        total_posts_step = len(batch_blocks) * tweets_per_sample
        print(f"[Step {step+1}/{num_steps}]  {len(batch_blocks)} agent sample(s) × {tweets_per_sample} posts = {total_posts_step} posts total")

        for idx, (tweet_block, phq9, persona) in enumerate(zip(batch_blocks, batch_answers, batch_personas)):
            valid_history = [t for t in tweet_block if t and t not in {"NO_POST", "NO_TWEET"}]
            n_init = int(rng.integers(0, min(len(valid_history), 4) + 1)) if valid_history else 0
            if n_init > 0:
                sampled = rng.choice(len(valid_history), size=n_init, replace=False)
                context = [valid_history[i] for i in sorted(sampled)]
            else:
                context = []

            raw_tweets = []
            parsed_tweets = []
            for j in range(tweets_per_sample):
                neighbor_tweets = _sample_neighbor_tweets(all_tweets_flat, rng, tweet_block)
                user_msg = _build_user_message_tweet(context, prompts, persona, phq9, neighbor_tweets)
                user_msg = user_msg + "\n\n" + format_block_tweet
                question = tg.Variable(
                    user_msg,
                    role_description="User template: persona, PHQ-9, own history, neighbor context",
                    requires_grad=False,
                )
                pred = model(question)
                pred.value = Agent.strip_model_thinking(pred.value)
                raw_tweets.append(pred.value)
                parsed = parse_tweet_answers(pred.value)
                parsed_tweets.append(parsed)
                if parsed:
                    context.append(parsed)

            valid_parsed = [t for t in parsed_tweets if t]
            print(f"  sample {idx+1}: PHQ-9={phq9}, {len(valid_parsed)}/{tweets_per_sample} posts parsed — teacher feedback based on this set")

            if idx == 0:
                print(f"  --- raw outputs [NON-THINKING student] (step {step+1}, sample 1) ---")
                for j, (raw, parsed) in enumerate(zip(raw_tweets, parsed_tweets)):
                    print(f"  tweet {j+1}: {raw}")
                    print(f"  parsed:  {parsed!r}")

            # Combine all tweets into a single Variable so the teacher evaluates
            # the full set at once and only one backward call is needed per sample
            # (instead of one per tweet, which is redundant since they all update
            # the same instruction Variable).
            combined = tg.Variable(
                "\n".join(raw_tweets),
                role_description="Set of generated posts for this sample (PHQ-9 and persona context)",
                requires_grad=False,
            )
            loss_fn = tg.TextLoss(_make_loss_prompt_tweet_set(parsed_tweets, persona, phq9))
            loss = loss_fn(combined)
            losses.append(loss)

        print(f"  combining {len(losses)} sample feedback(s) → single backward pass")
        total_loss = tg.sum(losses)
        total_loss.backward()

        grads = instruction.gradients
        if grads:
            print(f"  [teacher gradient]: {list(grads)[0].value[:400]}\n")

        optimizer.step()

        # Guard against TextGrad returning its literal template placeholder
        # ("{improved variable}") when the teacher fails to parse the optimizer prompt.
        _val = instruction.value.strip()
        if not _val or _val == "{improved variable}" or len(_val) < 20:
            print(f"  -> Optimizer returned placeholder ({_val!r}) — resetting to best instruction.")
            instruction.value = best_instruction

        # Hard-enforce word limit — truncate at sentence boundary where possible
        words = instruction.value.split()
        if len(words) > max_instruction_words:
            truncated = " ".join(words[:max_instruction_words])
            last_stop = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
            instruction.value = truncated[:last_stop + 1] if last_stop > 0 else truncated

        print(f"  Updated instruction ({len(instruction.value.split())} words):\n{instruction.value}\n")

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "latest_instruction_tweet.txt"), "w", encoding="utf-8") as fh:
            fh.write(instruction.value)

        # periodic validation on a sampled subset
        if (step + 1) % validate_every == 0 or step == num_steps - 1:
            val_score, val_std, _ = _evaluate_tweet_instruction(
                student_engine, teacher_engine, instruction.value, prompts,
                val_blocks, val_answers, val_personas,
                all_tweets_flat, rng, sample_size=val_sample_size,
                format_block=format_block_tweet, tweets_per_sample=tweets_per_sample,
            )
            with open(trajectory_path, "a", newline="", encoding="utf-8") as _fh:
                csv.DictWriter(_fh, fieldnames=_traj_fields).writerow({
                    "model": model_short, "seed": seed, "step": step + 1,
                    "split": "val", "mean_score": round(val_score, 4),
                    "std_score": round(val_std, 4), "n_samples": val_sample_size or len(val_blocks),
                })
            print(f"  -> Val quality score: {val_score:.2f}/10 ± {val_std:.2f}  (best: {best_val_score:.2f})")
            if val_score > best_val_score:
                best_val_score = val_score
                best_instruction = instruction.value
                print("  -> New best instruction saved!")
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "best_instruction_tweet.txt"), "w", encoding="utf-8") as fh:
                    fh.write(f"# val score: {val_score:.2f}/10  (step {step+1})\n\n{best_instruction}")
                with open(os.path.join(output_dir, "best_full_prompt_tweet.txt"), "w", encoding="utf-8") as fh:
                    fh.write(f"=== SYSTEM PROMPT ===\n{best_instruction}\n\n"
                             f"=== FORMAT BLOCK (fixed) ===\n{format_block_tweet}")
            else:
                instruction.value = best_instruction
                print("  -> No improvement — instruction reset to best.")

    # ── final test evaluation ──────────────────────────────────────────────
    test_score, test_std, per_phq9 = _evaluate_tweet_instruction(
        student_engine, teacher_engine, best_instruction, prompts,
        test_blocks, test_answers, test_personas,
        all_tweets_flat, rng, sample_size=test_sample_size,
        format_block=format_block_tweet, tweets_per_sample=tweets_per_sample,
    )
    with open(trajectory_path, "a", newline="", encoding="utf-8") as _fh:
        csv.DictWriter(_fh, fieldnames=_traj_fields).writerow({
            "model": model_short, "seed": seed, "step": num_steps,
            "split": "test", "mean_score": round(test_score, 4),
            "std_score": round(test_std, 4), "n_samples": test_sample_size or len(test_blocks),
        })
    print(f"\nTest quality score: {test_score:.2f}/10 ± {test_std:.2f}")
    print("Per-PHQ-9 scores:")
    for phq9_val, stats in per_phq9.items():
        print(f"  PHQ-9={phq9_val:2d}  avg={stats['avg_score']:.2f}  "
              f"n={stats['n_samples']}  empty={stats['n_empty']}")

    # ── save results ───────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # Per-PHQ-9 CSV
    csv_path = os.path.join(output_dir, "test_scores_phq9.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "seed", "phq9", "avg_score", "n_samples", "n_empty"])
        writer.writeheader()
        for phq9_val, stats in per_phq9.items():
            writer.writerow({
                "model": model_short,
                "seed": seed,
                "phq9": phq9_val,
                "avg_score": round(stats["avg_score"], 4),
                "n_samples": stats["n_samples"],
                "n_empty": stats["n_empty"],
            })
    print(f"Per-PHQ-9 scores → {csv_path}")

    instr_path = os.path.join(output_dir, "optimized_instruction_tweet.txt")
    with open(instr_path, "w", encoding="utf-8") as f:
        f.write(f"# val score: {best_val_score:.2f}/10  |  test score: {test_score:.2f}/10\n\n{best_instruction}")
    print(f"Best tweet instruction (val={best_val_score:.2f}, test={test_score:.2f}) → {instr_path}")

    full_path = os.path.join(output_dir, "optimized_full_prompt_tweet.txt")
    with open(full_path, "w", encoding="utf-8") as f:
        user_template = prompts["tweet_gen"]["user_template_forced"]
        f.write(
            f"=== SYSTEM PROMPT ===\n{best_instruction}\n\n"
            f"=== FORMAT BLOCK (fixed) ===\n{format_block_tweet}\n\n"
            f"=== USER TEMPLATE ===\n{user_template}"
        )
    print(f"Full prompt → {full_path}")

    # Release GPU memory before returning so a subsequent seed can initialise
    # a fresh engine without hitting the memory-utilisation check.
    try:
        del student_engine.base.client
    except Exception:
        pass
    del student_engine, teacher_engine
    import gc
    gc.collect()
    torch.cuda.empty_cache()

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
    

def _generate_file_path(base_dir: str, target_filename: str = "tweets_with_phq9.csv") -> list[str]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--val-sample-size", type=int, default=8)
    parser.add_argument("--test-sample-size", type=int, default=50)
    parser.add_argument("--model", type=str, default=QWEN_27,
                        help="Model to use (e.g. QWEN_27, LLAMA_8B, LLAMA_70B)")
    args = parser.parse_args()

    create_new_embeddings = False
    mental_bert = True
    file_paths = []

    for inter in ["no_inter", "inter"]:
        base_dir = f"data/test_post/Qwen_Qwen3.5-27B/temp_0.8_top_p_0.6_cp_10_{inter}"
        file_paths.extend(_generate_file_path(base_dir))

    for fp in file_paths:
        print(fp)

    model_name = args.model
    base_model_name = model_name

    # Choose which optimizer to run: "phq9", "tweets", or "bert"
    run_mode = "tweets"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if run_mode == "phq9":
        for seed in args.seeds:
            print(f"\n{'='*60}\nRunning PHQ-9 optimizer with seed={seed}\n{'='*60}")
            call_optimizer_phq9(
                file_paths,
                model_name=model_name,
                batch_size=5,
                seed=seed,
                output_dir="data/test_post/optimized_phq9",
            )
    elif run_mode == "tweets":
        for seed in args.seeds:
            print(f"\n{'='*60}\nRunning tweet optimizer with seed={seed}\n{'='*60}")
            call_optimizer_tweets(
                file_paths,
                model_name=model_name,
                batch_size=args.batch_size,
                max_model_len=32768,
                seed=seed,
                num_steps=args.num_steps,
                val_sample_size=args.val_sample_size,
                test_sample_size=args.test_sample_size,
                prompts_file="data/prompts_post_minimal.json",
            )
    else:
        if create_new_embeddings:
            save_embeddings_for_file(file_paths[0], base_model_name, device, mentalbert=mental_bert)

        dir_name = "mentalbert_embeddings" if mental_bert else "sbert_embeddings"
        embeddings_path = os.path.join("data", "test", base_model_name, dir_name, "embeddings_and_labels.pt")
        train_BERT_model(embeddings_path, base_model_name, device, mental_bert=mental_bert, split_seed=args.seed)