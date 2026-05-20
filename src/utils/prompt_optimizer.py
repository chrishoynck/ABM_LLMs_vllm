import os
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
from .metrics import *
from .format_config import FC
from .visualization import plot_optimizer_trajectory, plot_test_scores_by_phq9
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..classes.agent import Agent


QWEN_MODELS    = frozenset({})          # filled below after constants are defined
MISTRAL_MODELS = frozenset({})

def _extract_score_feedback(raw: str) -> str:
    """Return the last SCORE/FEEDBACK pair in `raw`; scans the whole string so a thinking dump still resolves.

    Args:
        raw: teacher output, possibly containing un-stripped thinking text.

    Returns:
        "SCORE: X\\nFEEDBACK: Y" if both present, "FEEDBACK: Y" if only FEEDBACK, else "".
    """
    score_matches = list(re.finditer(r"SCORE\s*:\s*([\d.]+)", raw, re.IGNORECASE))
    feedback_matches = list(re.finditer(r"FEEDBACK\s*:\s*(.+)", raw, re.IGNORECASE))
    if not feedback_matches:
        return ""
    feedback_val = feedback_matches[-1].group(1).strip()
    if score_matches:
        score_val = score_matches[-1].group(1).strip()
        return f"SCORE: {score_val}\nFEEDBACK: {feedback_val}"
    return f"FEEDBACK: {feedback_val}"


def _teacher_call_kind(content: str) -> str:
    """Classify a teacher engine call by inspecting markers TextGrad embeds in the prompt.

    Args:
        content: full user message passed to the teacher engine.

    Returns:
        'backward' (gradient computation), 'optimizer' (instruction rewrite), or 'loss' (SCORE/FEEDBACK rating).
    """
    if "<CONVERSATION>" in content:
        return "backward"
    if "IMPROVED_VARIABLE" in content:
        return "optimizer"
    return "loss"

# Custom optimizer system prompt — TextGrad's default includes literal "{improved variable}"
# placeholders that thinking models (Qwen3.5, Mistral) echo back verbatim.
_OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., a variable). "
    "You will receive feedback on a variable and must produce an improved version. "
    "The feedback may be noisy — identify what is important and what is correct. "
    "Pay attention to the role description of the variable and the context in which it is used. "
    "YOUR JOB IS A SMALL, INCREMENTAL EDIT. Pick the single most important point in the "
    "feedback and address it by modifying or adding at most one short clause or sentence. "
    "Keep every other sentence exactly as it was. "
    "It is fine (and often the right move) for the rewrite to end up slightly longer than the "
    "input — do not hesitate to add a brief clarifying phrase when that is what the feedback "
    "calls for. Do not shorten the input unless the feedback explicitly tells you to remove "
    "something. "
    "If the feedback is vague, contradictory, or already handled, return the variable unchanged. "
    "IMPORTANT: Write the actual improved text itself between "
    "{new_variable_start_tag} and {new_variable_end_tag} tags. "
    "Do NOT write placeholder words like 'improved variable' between the tags — "
    "write the real improved content.\n\n"
    f"{GLOSSARY_TEXT}"
)




class _StudentEngine(tg.engine.EngineLM):
    """ChatVLLM wrapper that disables reasoning per-request (Qwen/Mistral specific)."""
    def __init__(self, base: ChatVLLM, model_name: str,
                 temperature: float = 0.7, max_tokens: int = 512):
        self.base = base
        self.model_string = base.model_string
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

    def __call__(self, content, system_prompt=None, **kwargs):
        return self.generate(content, system_prompt=system_prompt, **kwargs)

    def generate(self, content, system_prompt=None, temperature=None, max_tokens=None, **kwargs):
        """Generate one student response with thinking disabled for the active model family.

        Args:
            content: user message text.
            system_prompt: optional override; defaults to the engine's system prompt.
            temperature: sampling temperature override; defaults to the value passed at init.
            max_tokens: token budget override; defaults to the value passed at init.

        Returns:
            Generated text with any spurious role headers stripped.
        """
        temperature = temperature if temperature is not None else self._temperature
        max_tokens  = max_tokens  if max_tokens  is not None else self._max_tokens
        if self._model_name in QWEN_MODELS:
            sys_prompt = system_prompt or self.base.system_prompt
            conversation = ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
            conversation.append({"role": "user", "content": content})
            chat_str = self.base.tokenizer.apply_chat_template(
                conversation, tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=0.9, n=1)
            result = self.base.client.generate([chat_str], sampling_params)[0].outputs[0].text
            # Qwen3.5 sometimes re-emits the "assistant" role header when enable_thinking=False without a prefill.
            stripped = result.lstrip()
            if stripped.lower().startswith("assistant"):
                result = stripped[len("assistant"):].lstrip("\n :")
        elif self._model_name in MISTRAL_MODELS:
            # Mistral Small 4: prefill [/THINK] to skip the thinking block (reasoning_effort is API-only).
            sys_prompt = system_prompt or self.base.system_prompt
            conversation = ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
            conversation.append({"role": "user", "content": content})
            chat_str = self.base.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
            )
            chat_str = chat_str + "[/THINK]"
            # Mistral docs: tune temperature OR top_p, not both.
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=1.0, n=1)
            result = self.base.client.generate([chat_str], sampling_params)[0].outputs[0].text
        else:
            result = self.base.generate(content, system_prompt=system_prompt, **kwargs)
        return result


class _TeacherEngine(tg.engine.EngineLM):
    """Thinking-enabled engine used as the TextGrad backward/optimizer engine."""
    def __init__(self, base: ChatVLLM, model_name: str):
        self.base = base
        self.model_string = base.model_string
        self._model_name = model_name

    def __call__(self, content, system_prompt=None, **kwargs):
        return self.generate(content, system_prompt=system_prompt, **kwargs)

    def generate(self, content, system_prompt=None, **kwargs):
        """Run the teacher with thinking enabled and return clean answer text only.

        Args:
            content: user message text (loss prompt, backward trace, or optimizer prompt).
            system_prompt: optional override; defaults to the engine's system prompt.
            **kwargs: forwarded; `max_tokens` selects a backward- vs optimizer-sized budget.

        Returns:
            Cleaned answer text with thinking blocks removed; empty if reasoning was truncated.
        """
        if self._model_name in QWEN_MODELS:
            sys_prompt = system_prompt or self.base.system_prompt
            concise_note = "Be brief in your thinking. Identify the key point and act on it immediately — do not re-examine the same idea multiple times or draft more than once."
            sys_prompt = (sys_prompt + "\n" + concise_note) if sys_prompt else concise_note
            conversation = ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
            conversation.append({"role": "user", "content": content})
            token_ids = self.base.tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
                enable_thinking=True,
            )
            if len(token_ids) > 16384:
                token_ids = token_ids[-16384:]
            # Backward calls need >8k so instruction-variable gradients aren't truncated mid-thinking.
            default_tokens = 8192 if "IMPROVED_VARIABLE" in content else 10240
            base_max_tokens = kwargs.get("max_tokens", default_tokens)
            # Only retry on optimizer calls (IMPROVED_VARIABLE) — an empty response
            # there crashes TextGrad's TGD tag parser. Loss/backward calls tolerate
            # empty results, so retrying them would just waste compute.
            max_attempts = 2 if "IMPROVED_VARIABLE" in content else 1

            result = ""
            for attempt in range(max_attempts):
                cur_max_tokens = base_max_tokens * (2 if attempt else 1)
                if attempt:
                    print(f"  [teacher] retrying optimizer call with max_tokens={cur_max_tokens} (was {base_max_tokens})")
                sampling_params = SamplingParams(temperature=0, max_tokens=cur_max_tokens, top_p=0.99, n=1)
                raw = self.base.client.generate([{"prompt_token_ids": token_ids}], sampling_params)
                output = raw[0].outputs[0]
                reasoning = getattr(output, "reasoning_content", None)
                if reasoning:
                    preview = reasoning[:1200] + (" ..." if len(reasoning) > 1200 else "")
                    print(f"  [teacher reasoning | {len(reasoning.split())} words]\n{preview}\n  [/teacher reasoning]")
                    result = output.text
                    break
                if attempt == 0:
                    print("  [teacher] reasoning_parser unavailable — stripping manually")
                raw_text = output.text or ""
                # Missing </think> means thinking was truncated mid-stream.
                if "</think>" not in raw_text:
                    if attempt < max_attempts - 1:
                        print("  [teacher] thinking did not finish — will retry with larger budget")
                        continue
                    print("  [teacher] thinking did not finish — discarding feedback")
                    result = ""
                    break
                answer = raw_text.rsplit("</think>", 1)[1].strip()
                if "<CONVERSATION>" not in content and "IMPROVED_VARIABLE" not in content:
                    result = _extract_score_feedback(answer)
                    if not result:
                        print("  [teacher] FEEDBACK not found — discarding feedback")
                        result = ""
                else:
                    result = answer
                break

            kind = _teacher_call_kind(content)
            print(f"  [teacher feedback | {kind}]\n{result.strip()}\n  [/teacher feedback]")
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
            base_max_tokens = kwargs.get("max_tokens", 8192)
            # Only retry optimizer calls (see Qwen branch for rationale).
            max_attempts = 2 if "IMPROVED_VARIABLE" in content else 1

            result = ""
            for attempt in range(max_attempts):
                cur_max_tokens = base_max_tokens * (2 if attempt else 1)
                if attempt:
                    print(f"  [teacher] retrying optimizer call with max_tokens={cur_max_tokens} (was {base_max_tokens})")
                sampling_params = SamplingParams(temperature=0.6, max_tokens=cur_max_tokens, top_p=0.97, n=1)
                raw = self.base.client.generate([{"prompt_token_ids": token_ids}], sampling_params)
                output = raw[0].outputs[0]
                reasoning = getattr(output, "reasoning_content", None)
                if reasoning:
                    preview = reasoning[:1200] + (" ..." if len(reasoning) > 1200 else "")
                    print(f"  [teacher reasoning | {len(reasoning.split())} words]\n{preview}\n  [/teacher reasoning]")
                    result = output.text
                    break
                if attempt == 0:
                    print("  [teacher] reasoning_parser unavailable in this vLLM version — stripping manually")
                raw_text = output.text or ""
                if "[THINK]" in raw_text and "[/THINK]" not in raw_text:
                    if attempt < max_attempts - 1:
                        print("  [teacher] thinking did not finish — will retry with larger budget")
                        continue
                    print("  [teacher] thinking did not finish — discarding feedback")
                    result = ""
                    break
                result = re.sub(r"\[THINK\].*?\[/THINK\]", "", raw_text,
                                flags=re.DOTALL).strip()
                break

            kind = _teacher_call_kind(content)
            print(f"  [teacher feedback | {kind}]\n{result.strip()}\n  [/teacher feedback]")
            return result
        return self.base.generate(content, system_prompt=system_prompt, **kwargs)


def _build_engines(model_name: str, tp: int, gpu_memory_utilization: float,
                   max_model_len: int = 32768,
                   student_temperature: float = 0.7, student_max_tokens: int = 512,
                   **vllm_kwargs):
    """Load a single shared ChatVLLM and wrap it as both student and teacher.

    Args:
        model_name: HuggingFace model id (Qwen or Mistral family).
        tp: tensor-parallel size for vLLM.
        gpu_memory_utilization: fraction of GPU memory vLLM may use.
        max_model_len: max context length for the engine.
        student_temperature: sampling temperature for the student.
        student_max_tokens: token budget for the student.
        **vllm_kwargs: extra args forwarded to ChatVLLM (e.g. reasoning_parser).

    Returns:
        (student_engine, teacher_engine). The teacher is also set as TextGrad's backward engine.
    """
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
    return _StudentEngine(engine, model_name, student_temperature, student_max_tokens), teacher


def parse_tweets_with_phq9(file_path: str):
    """Parse a tweets_with_phq9.txt file, grouping consecutive same-PHQ-9 tweets into blocks.

    Args:
        file_path: path to a tweets_with_phq9.txt file written by TestLLMs.

    Returns:
        (tweet_blocks, true_answers) where each block is a list of tweets sharing one PHQ-9 score.
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

        if line.startswith("=== Agent"):
            if current_tweets and len(current_tweets) > 1:
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

        # Strip trailing '(CHANGED from X)' metadata.
        changed_idx = tweet.rfind("  (CHANGED from ")
        if changed_idx != -1:
            tweet = tweet[:changed_idx]

        if phq9 != current_phq9:
            if current_tweets and len(current_tweets) > 1:
                tweet_blocks.append(current_tweets)
                true_answers.append(current_phq9)
            current_phq9 = phq9
            current_tweets = [tweet]
        else:
            current_tweets.append(tweet)

    if current_tweets and len(current_tweets) > 1:
        tweet_blocks.append(current_tweets)
        true_answers.append(current_phq9)

    return tweet_blocks, true_answers


def parse_tweets_with_phq9_csv(file_path: str):
    """Parse a tweets_with_phq9.csv into the same block structure as the .txt parser, plus personas/ids.

    Args:
        file_path: path to a tweets_with_phq9.csv written by TestLLMs.

    Returns:
        (tweet_blocks, true_answers, personas, agent_ids), grouped by consecutive (agent_id, phq9) rows.
    """
    tweet_blocks: list[list[str]] = []
    true_answers: list[int] = []
    personas: list[str] = []
    agent_ids: list[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        current_agent = None
        current_phq9 = None
        current_tweets: list[str] = []

        for row in reader:
            agent_id = row.get("agent_id")
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
                if current_tweets and current_phq9 is not None and len(current_tweets) > 1:
                    tweet_blocks.append(current_tweets)
                    true_answers.append(current_phq9)
                    personas.append(current_persona)
                    agent_ids.append(current_agent)

                current_agent = agent_id
                current_phq9 = phq9
                current_persona = persona
                current_tweets = [tweet] if tweet else []
            else:
                if tweet:
                    current_tweets.append(tweet)

        if current_tweets and current_phq9 is not None and len(current_tweets) > 1:
            tweet_blocks.append(current_tweets)
            true_answers.append(current_phq9)
            personas.append(current_persona)
            agent_ids.append(current_agent)
    return tweet_blocks, true_answers, personas, agent_ids


QWEN_27      = "Qwen/Qwen3.5-27B"                      # 32k context
MISTRAL_119B = "mistralai/Mistral-Small-4-119B-2603"   # 256k context, MoE 119B/6.5B active

# Fill model-family sets used by _StudentEngine / _TeacherEngine
QWEN_MODELS    = frozenset({QWEN_27})
MISTRAL_MODELS = frozenset({MISTRAL_119B})


def _build_user_message(format_block: str, tweet_block: list, prompts: dict, persona: str = None) -> str:
    """Compose the PHQ-9 user message: fixed format block + tweet data + optional persona.

    Args:
        format_block: fixed PHQ-9 questions/scoring options/answer format.
        tweet_block: list of tweet strings for one agent.
        prompts: parsed prompts JSON (provides the user templates).
        persona: optional persona text; selects the persona-aware template when present.

    Returns:
        Concatenated user message string.
    """
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
                          personas: list = None,
                          temperature: float = 0.2, max_tokens: int = 256) -> tuple:
    """Run the current PHQ-9 instruction on all blocks and return aggregated MAE.

    Args:
        engine: student _StudentEngine.
        instruction_text: current system instruction being evaluated.
        format_block: fixed PHQ-9 format block appended to every user message.
        blocks: list of tweet blocks (one per agent).
        answers: ground-truth PHQ-9 score per block.
        prompts: parsed prompts JSON.
        personas: optional personas aligned with `blocks`.
        temperature: student sampling temperature.
        max_tokens: student token budget.

    Returns:
        (mean_mae, std_mae, per_phq9) where per_phq9 = {true_score: {"avg_mae", "n_samples"}}.
    """
    user_msgs = []
    for i, tweet_block in enumerate(blocks):
        persona = personas[i] if personas else None
        user_msgs.append(_build_user_message(format_block, tweet_block, prompts, persona))

    responses = _batch_student_generate(engine, user_msgs, instruction_text,
                                        temperature=temperature, max_tokens=max_tokens)

    abs_errors = []
    per_phq9_errors = defaultdict(list)
    for response, true_answer in zip(responses, answers):
        predicted = Agent.parse_phq9_answers(response)
        ae = abs(predicted - true_answer)
        abs_errors.append(ae)
        per_phq9_errors[true_answer].append(ae)
    arr = np.array(abs_errors) if abs_errors else np.array([0.0])
    per_phq9 = {k: {"avg_mae": float(np.mean(v)), "n_samples": len(v)}
                for k, v in per_phq9_errors.items()}
    return float(arr.mean()), float(arr.std()), per_phq9


def _make_loss_prompt(true_answer: int, predicted: int,
                      tweet_block: list[str] = None, persona: str = None) -> str:
    """Build the per-sample loss prompt the teacher uses to critique the PHQ-9 instruction.

    Args:
        true_answer: ground-truth PHQ-9 score.
        predicted: student's parsed PHQ-9 score.
        tweet_block: tweets the prediction was based on (first 5 are shown).
        persona: optional persona snippet (truncated to 120 chars).

    Returns:
        Loss prompt text asking the teacher for a single FEEDBACK sentence.
    """
    error = predicted - true_answer
    if error > 0:
        direction = f"overestimated by {error}"
    elif error < 0:
        direction = f"underestimated by {abs(error)}"
    else:
        direction = "correct"
    persona_note = f"Persona: {persona[:120]}." if persona else ""
    valid_posts = [t for t in (tweet_block or []) if t and t not in {"NO_POST", "NO_TWEET"}]
    sample = valid_posts[:5]
    posts_block = ("Sample posts:\n" + "\n".join(f"- {t}" for t in sample)) if sample else ""
    if error == 0:
        guidance = (
            f"Think briefly. The prediction is EXACT — the instruction is doing this case well. "
            f"Your feedback should describe what the instruction did well so it can be preserved "
            f"(do NOT invent issues to correct)."
        )
    elif abs(error) <= 3:
        guidance = (
            f"Think briefly. The error is small (±3 PHQ-9 points) — this is essentially a good "
            f"prediction. Focus your feedback on what the instruction did well and should keep "
            f"doing, with at most a minor refinement."
        )
    else:
        guidance = (
            f"Think briefly — identify the single most important issue and act on it immediately."
        )
    return (
        f"{posts_block}\n\n{persona_note}\n"
        f"True PHQ-9 = {true_answer}, predicted = {predicted} ({direction}). MAE = {abs(error)}.\n\n"
        f"{guidance}\n\n"
        f"Respond with exactly:\n"
        f"FEEDBACK: <one sentence — for exact or near-correct predictions, describe what the "
        f"instruction is doing well that should be preserved; otherwise describe what the "
        f"instruction should say differently (e.g. distinguishing persistent depressive symptoms "
        f"from personality traits or daily mood fluctuations, avoiding over- or under-weighting "
        f"single emotional posts, or correcting systematic bias for this PHQ-9 range)>"
    ).strip()

def train_val_test_split(rng, file_paths:list[str],
                         val_fraction: float = 0.10,
                         test_fraction: float = 0.10):
    """Parse one or more tweets_with_phq9 files and split into train/val/test.

    Args:
        rng: numpy Generator for the permutation.
        file_paths: paths to .csv (preferred) or .txt files.
        val_fraction: fraction held out for validation.
        test_fraction: fraction held out for test (drawn first, then val, then train).

    Returns:
        (train_data, val_data, test_data) — each a tuple (blocks, answers, personas, agent_ids).
    """
    tweet_blocks_list = []
    true_answers_list = []
    personas_list = []
    agent_ids_list = []

    for file_path in file_paths:
        if file_path.endswith(".csv"):
            csv_path = file_path
            txt_path = file_path.replace(".csv", ".txt")
        else:
            txt_path = file_path
            csv_path = file_path.replace(".txt", ".csv") if file_path.endswith(".txt") else file_path + ".csv"

        if os.path.isfile(csv_path):
            tweet_blocks, true_answers, personas, agent_ids = parse_tweets_with_phq9_csv(csv_path)
            tweet_blocks_list.extend(tweet_blocks)
            true_answers_list.extend(true_answers)
            personas_list.extend(personas)
            agent_ids_list.extend(agent_ids)
            print(f"Parsed {len(tweet_blocks)} tweet blocks from {csv_path}")
        else:
            tweet_blocks, true_answers = parse_tweets_with_phq9(txt_path)
            personas = [None] * len(tweet_blocks)
            tweet_blocks_list.extend(tweet_blocks)
            true_answers_list.extend(true_answers)
            personas_list.extend(personas)
            agent_ids_list.extend(["unknown"] * len(tweet_blocks))
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
                [personas_list[i] for i in indices],
                [agent_ids_list[i] for i in indices])

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
    validate_every: int = 2,
    val_sample_size: int = 10,
    test_sample_size: int = None,
    seed: int = 42,
    max_model_len: int = 32768,
    output_dir: str = None,
    **vllm_kwargs,
):
    """Optimise the PHQ-9 system instruction via TextGrad with batched gradient accumulation.

    Args:
        file_paths: tweets_with_phq9 files (.csv or .txt) to draw training data from.
        model_name: HuggingFace model id used as both student and teacher.
        batch_size: number of samples per gradient-accumulation step.
        max_instruction_words: soft length budget enforced via the optimizer constraint.
        num_steps: number of training steps; defaults to `len(train) // batch_size`, capped at 50.
        val_fraction: fraction of data reserved for validation.
        test_fraction: fraction of data reserved for test.
        validate_every: validate every N training steps (and at the last step).
        val_sample_size: size of the fixed validation subset.
        test_sample_size: optional cap on the final test set.
        seed: RNG seed.
        max_model_len: vLLM context budget.
        output_dir: where to write checkpoints, trajectory CSV, and plots.
        **vllm_kwargs: forwarded to ChatVLLM.

    Returns:
        The best instruction string seen (also written to `best_instruction.txt`).
    """
    rng = np.random.default_rng(seed)

    # Load + split data.
    train_data, val_data, test_data = train_val_test_split(
        rng, file_paths, val_fraction, test_fraction,
    )
    train_blocks, train_answers, train_personas, _train_aids = train_data
    val_blocks, val_answers, val_personas, _val_aids = val_data
    test_blocks, test_answers, test_personas, _test_aids = test_data

    # Build shared vLLM engines (student + teacher).
    tp = vllm_kwargs.pop("tensor_parallel_size", None) or len((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(","))
    student_engine, teacher_engine = _build_engines(
        model_name, tp, 0.90, max_model_len=max_model_len,
        student_temperature=0.2, student_max_tokens=256,
        **vllm_kwargs,
    )

    # Split prompt into instruction (grad) + format (fixed).
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
            f"Keep the instruction under {max_instruction_words} words.",
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

    model_short = model_name.split("/")[-1]
    if output_dir is None:
        output_dir = f"data/test_post/optimized_phq9/{model_short}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    trajectory_path = os.path.join(output_dir, "training_trajectory.csv")
    _traj_fields = ["model", "seed", "step", "split", "mean_score", "std_score", "n_samples"]
    with open(trajectory_path, "w", newline="", encoding="utf-8") as _fh:
        csv.DictWriter(_fh, fieldnames=_traj_fields).writeheader()

    # Fixed validation subset: sample once and reuse so step-to-step comparisons
    # are paired. A fresh draw each time was masking improvements under sampling noise.
    n_val = min(val_sample_size, len(val_blocks))
    val_idx = rng.choice(len(val_blocks), size=n_val, replace=False)
    fixed_val_blocks   = [val_blocks[i]   for i in val_idx]
    fixed_val_answers  = [val_answers[i]  for i in val_idx]
    fixed_val_personas = [val_personas[i] for i in val_idx]
    print(f"[val] fixed evaluation subset: {n_val} agents (reused every validation)")

    # Step 0: baseline evaluation before any training.
    s0_mae, s0_std, _ = _evaluate_instruction(
        student_engine, instruction.value, format_block,
        fixed_val_blocks, fixed_val_answers, prompts, fixed_val_personas,
    )
    with open(trajectory_path, "a", newline="", encoding="utf-8") as _fh:
        csv.DictWriter(_fh, fieldnames=_traj_fields).writerow({
            "model": model_short, "seed": seed, "step": 0,
            "split": "val", "mean_score": round(s0_mae, 4),
            "std_score": round(s0_std, 4), "n_samples": n_val,
        })
    print(f"[Step 0] Baseline val MAE: {s0_mae:.2f} ± {s0_std:.2f}")
    best_val_mae = s0_mae
    best_instruction = instruction.value
    with open(os.path.join(output_dir, "best_instruction.txt"), "w", encoding="utf-8") as fh:
        fh.write(best_instruction)
    with open(os.path.join(output_dir, "best_full_prompt.txt"), "w", encoding="utf-8") as fh:
        fh.write(best_instruction + "\n\n" + format_block)

    # Training loop.
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

        step_errors = []

        for sample_idx, (tweet_block, true_answer, persona) in enumerate(
                zip(batch_blocks, batch_answers, batch_personas)):
            user_msg = _build_user_message(format_block, tweet_block, prompts, persona)

            if sample_idx == 0:
                print(f"\n  --- [PHQ-9 step {step+1}, sample 1] system prompt ---")
                print(instruction.value)
                print(f"  --- user message ---")
                print(user_msg)
                print(f"  --- end prompt ---\n")

            question = tg.Variable(
                user_msg,
                role_description="PHQ-9 format, questions, patient tweet history, and persona",
                requires_grad=False,
            )

            prediction = model(question)
            # Drop the user-message predecessor so TextGrad doesn't spend a teacher
            # call "improving" it (it's requires_grad=False and discarded anyway).
            prediction.predecessors = {p for p in prediction.predecessors if p is not question}
            predicted_score = Agent.parse_phq9_answers(prediction.value)
            error = predicted_score - true_answer
            step_errors.append(error)

            if sample_idx == 0:
                print(f"  --- [PHQ-9 step {step+1}, sample 1] raw student output ---")
                print(prediction.value)
                print(f"  --- parsed: {predicted_score}  |  true: {true_answer}  |  error: {error:+d} ---\n")

            # Run the teacher even when the prediction is exact: positive feedback
            # on what the instruction is doing well is also a useful gradient
            # signal (reinforces the existing behaviour rather than chasing change).
            prediction.value = f"My assessment of the patient's PHQ-9 total score is {predicted_score}."

            loss_fn = tg.TextLoss(_make_loss_prompt(true_answer, predicted_score, tweet_block, persona))
            loss = loss_fn(prediction)
            loss.backward()

        # Even after the engine's retry, the teacher can return empty on the
        # optimizer call (e.g. thinking still didn't finish). TextGrad's TGD
        # parser raises IndexError on empty/malformed responses — catch it so
        # an unparseable rewrite simply means no instruction update this step.
        try:
            optimizer.step()
        except (IndexError, ValueError) as exc:
            print(f"  [optimizer] step skipped — response unparseable ({exc.__class__.__name__}): instruction left unchanged")

        abs_errors = np.abs(step_errors)
        batch_mae = float(abs_errors.mean())
        batch_std = float(abs_errors.std())
        print(f"[Step {step+1}/{num_steps}]  batch MAE={batch_mae:.2f} ± {batch_std:.2f}  errors={step_errors}")
        with open(trajectory_path, "a", newline="", encoding="utf-8") as _fh:
            csv.DictWriter(_fh, fieldnames=_traj_fields).writerow({
                "model": model_short, "seed": seed, "step": step + 1,
                "split": "train", "mean_score": round(batch_mae, 4),
                "std_score": round(batch_std, 4), "n_samples": len(batch_idx),
            })

        with open(os.path.join(output_dir, "latest_instruction_phq9.txt"), "w", encoding="utf-8") as fh:
            fh.write(instruction.value)

        # Periodic validation on the fixed subset.
        if (step + 1) % validate_every == 0 or step == num_steps - 1:
            val_mae, val_std, _ = _evaluate_instruction(
                student_engine, instruction.value, format_block,
                fixed_val_blocks, fixed_val_answers, prompts, fixed_val_personas,
            )
            with open(trajectory_path, "a", newline="", encoding="utf-8") as _fh:
                csv.DictWriter(_fh, fieldnames=_traj_fields).writerow({
                    "model": model_short, "seed": seed, "step": step + 1,
                    "split": "val", "mean_score": round(val_mae, 4),
                    "std_score": round(val_std, 4), "n_samples": n_val,
                })
            print(f"  -> Val MAE ({n_val} samples, fixed): {val_mae:.2f} ± {val_std:.2f}  (best: {best_val_mae:.2f})")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_instruction = instruction.value
                print("  -> New best instruction saved!")
                with open(os.path.join(output_dir, "best_instruction.txt"), "w", encoding="utf-8") as fh:
                    fh.write(best_instruction)
                with open(os.path.join(output_dir, "best_full_prompt.txt"), "w", encoding="utf-8") as fh:
                    fh.write(best_instruction + "\n\n" + format_block)
            else:
                instruction.value = best_instruction
                print("  -> No improvement — instruction reset to best.")

    # Final test evaluation on the best instruction.
    if test_sample_size and len(test_blocks) > test_sample_size:
        t_idx = rng.choice(len(test_blocks), size=test_sample_size, replace=False)
        eval_blocks   = [test_blocks[i]   for i in t_idx]
        eval_answers  = [test_answers[i]  for i in t_idx]
        eval_personas = [test_personas[i] for i in t_idx]
    else:
        eval_blocks, eval_answers, eval_personas = test_blocks, test_answers, test_personas
    test_mae, test_std, per_phq9 = _evaluate_instruction(
        student_engine, best_instruction, format_block,
        eval_blocks, eval_answers, prompts, eval_personas,
    )
    with open(trajectory_path, "a", newline="", encoding="utf-8") as _fh:
        csv.DictWriter(_fh, fieldnames=_traj_fields).writerow({
            "model": model_short, "seed": seed, "step": num_steps,
            "split": "test", "mean_score": round(test_mae, 4),
            "std_score": round(test_std, 4), "n_samples": len(test_blocks),
        })
    print(f"\nTest MAE (n={len(test_blocks)}): {test_mae:.2f} ± {test_std:.2f}")

    # Save results.
    instr_path = os.path.join(output_dir, "optimized_instruction.txt")
    with open(instr_path, "w", encoding="utf-8") as f:
        f.write(best_instruction)
    print(f"\nBest instruction (val MAE={best_val_mae:.2f}, test MAE={test_mae:.2f}) → {instr_path}")

    full_path = os.path.join(output_dir, "optimized_full_prompt.txt")
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(best_instruction + "\n\n" + format_block)
    print(f"Full re-assembled prompt → {full_path}")

    csv_path = os.path.join(output_dir, "test_scores_phq9.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "seed", "phq9", "avg_mae", "n_samples"])
        writer.writeheader()
        for phq9_val, stats in sorted(per_phq9.items()):
            writer.writerow({"model": model_short, "seed": seed, "phq9": phq9_val,
                             "avg_mae": round(stats["avg_mae"], 4), "n_samples": stats["n_samples"]})
    print(f"Per-PHQ-9 MAE → {csv_path}")

    plot_optimizer_trajectory(
        trajectory_path, output_dir,
        title=f"PHQ-9 optimizer — {model_short} seed={seed}",
        mode="phq9",
    )
    plot_test_scores_by_phq9(
        {k: {"avg_mae": v["avg_mae"], "n_samples": v["n_samples"]} for k, v in per_phq9.items()},
        output_dir,
        title=f"Test MAE by PHQ-9 — {model_short} seed={seed}",
        mode="phq9",
    )

    try:
        del student_engine.base.client
    except Exception:
        pass
    del student_engine, teacher_engine
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return best_instruction



###################### TWEET GENERATION OPTIMIZATION ######################


def _phq9_severity(score: int) -> str:
    """Map a PHQ-9 score to a coarse severity label.

    Args:
        score: PHQ-9 total (0-27).

    Returns:
        'severe' / 'moderately severe' / 'moderate' / 'mild' / 'minimal/none'.
    """
    if score >= 20:
        return "severe"
    if score >= 15:
        return "moderately severe"
    if score >= 10:
        return "moderate"
    if score >= 5:
        return "mild"
    return "minimal/none"


def _build_user_message_tweet(context_tweets: list[str], prompts: dict,
                              persona: str, phq9_score: int,
                              neighbor_tweets: list[tuple[str, str]] = None) -> str:
    """Build the tweet-generation user message from the forced template.

    Args:
        context_tweets: this agent's prior tweets, used as PREVIOUS POSTS context.
        prompts: parsed prompts JSON (provides `tweet_gen.user_template_forced`).
        persona: persona text for the agent.
        phq9_score: current PHQ-9 score (drives severity label).
        neighbor_tweets: optional (tweet, agent_id) tuples shown as POSTS FROM OTHERS.

    Returns:
        Fully formatted user message string.
    """
    no_content_values = {"NO_POST", "NO_TWEET", "no_post", "no_tweet", ""}
    real_context = [t for t in context_tweets if t not in no_content_values]
    if real_context:
        prev_block = "### PREVIOUS POSTS ###\n" + "\n".join(f"- {t}" for t in real_context)
    else:
        prev_block = "### PREVIOUS POSTS ###\n(none yet)"

    if neighbor_tweets:
        prev_block += "\n### POSTS FROM OTHERS ###\n" + "\n".join(
            f"- @user_{aid}: {t}" for t, aid in neighbor_tweets
        )

    template = prompts["tweet_gen"]["user_template_forced"]
    return template.format(
        agent_id="AGENT",
        persona=persona or "unspecified",
        well_being=f"{_phq9_severity(phq9_score)} (PHQ-9: {phq9_score})",
        previous_tweet_block=prev_block,
    )


def _sample_neighbor_tweets(all_tweets_flat: list[tuple[str, str]], rng, exclude: list[str], n: int = 3) -> list[tuple[str, str]]:
    """Sample up to `n` random (tweet, agent_id) pairs from the global pool.

    Args:
        all_tweets_flat: flattened pool of (tweet, agent_id) across all splits.
        rng: numpy Generator.
        exclude: tweets belonging to the current agent (skipped to avoid self-leakage).
        n: maximum number of neighbor tweets to return.

    Returns:
        Up to `n` sampled (tweet, agent_id) tuples; fewer if the pool is small.
    """
    exclude_set = set(exclude)
    pool = [(t, aid) for t, aid in all_tweets_flat if t not in exclude_set]
    n = min(n, len(pool))
    if n == 0:
        return []
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in idx]


def parse_tweet_answers(raw_output: str) -> str:
    """Extract the tweet body from raw student output.

    Args:
        raw_output: full student response (may contain thinking blocks and POST:/TWEET: prefixes).

    Returns:
        Cleaned tweet text — first paragraph after the last recognised prefix, or the first paragraph
        of the cleaned text if no prefix is found.
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
    """Loss prompt asking the teacher to rate a set of tweets from one agent.

    Args:
        tweets: parsed tweets for the agent (set rated collectively).
        persona: agent persona text for context.
        phq9_score: ground-truth PHQ-9 score for the agent.

    Returns:
        Prompt text requiring the teacher to reply with SCORE + FEEDBACK lines.
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
        "4. Diversity: Do the posts cover somewhat different topics or moods? Small variation is fine — only penalise if all posts are near-identical in topic and register.\n"
        "5. Interaction: Reward genuine engagement (reply, mock, support, correct). "
        "Penalise hollow @mentions and sets where every post is a reply.\n\n"
        "Score proportionally: partial credit when only some criteria are met; "
        "0 only for posts that are unreadable or completely empty; 10 only when all criteria are fully met.\n\n"
        "Respond with exactly two lines:\n"
        "SCORE: <number 0-10>\n"
        "FEEDBACK: <one sentence — if your SCORE is 7 or higher, describe what the system "
        "instruction is doing well that should be PRESERVED (do NOT invent issues to fix); "
        "otherwise describe what the system instruction should change>"
    )


def _batch_student_generate(student_engine, user_messages: list, instruction_text: str,
                            temperature: float = 0.7, max_tokens: int = 512) -> list:
    """Generate one response per message in a single vLLM call.

    Args:
        student_engine: _StudentEngine wrapping the shared ChatVLLM.
        user_messages: independent user prompts; each becomes one generation.
        instruction_text: system prompt applied to every message.
        temperature: sampling temperature.
        max_tokens: per-response token budget.

    Returns:
        List of generated strings, aligned with `user_messages`.
    """
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
            )
            chat_strs.append(chat_str)
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=0.9, n=1)
        results = student_engine.base.client.generate(chat_strs, sp)
        raw_outputs = [r.outputs[0].text for r in results]
        # Strip spurious "assistant" role header that Qwen sometimes emits without a prefill.
        cleaned = []
        for text in raw_outputs:
            s = text.lstrip()
            if s.lower().startswith("assistant"):
                s = s[len("assistant"):].lstrip("\n :")
            cleaned.append(s)
        return cleaned
    return [student_engine.generate(m, system_prompt=instruction_text,
                                    temperature=temperature, max_tokens=max_tokens)
            for m in user_messages]


def _batch_teacher_rate(teacher_engine, rating_prompts: list, max_tokens: int = 6144) -> list:
    """Rate a batch of tweet sets in one vLLM call, stripping thinking from each output.

    Args:
        teacher_engine: _TeacherEngine wrapping the shared ChatVLLM.
        rating_prompts: rating prompts, one per agent set.
        max_tokens: per-response token budget (large enough for thinking + answer).

    Returns:
        List of cleaned SCORE/FEEDBACK strings aligned with `rating_prompts`.
    """
    if not rating_prompts:
        return []
    if teacher_engine._model_name in QWEN_MODELS:
        concise_note = "Be brief in your thinking. Identify the key point and act on it immediately — do not re-examine the same idea multiple times or draft more than once."
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
            if getattr(out, "reasoning_content", None):
                text = out.text
            else:
                stripped = Agent.strip_model_thinking(out.text)
                text = _extract_score_feedback(stripped) or _extract_score_feedback(out.text)
                if not text:
                    print("  [teacher batch] SCORE/FEEDBACK not found — discarding feedback")
            outputs.append(text)
        return outputs
    return [teacher_engine.generate(p, max_tokens=max_tokens) for p in rating_prompts]


def _evaluate_tweet_instruction(student_engine, teacher_engine, instruction_text: str,
                                prompts: dict, blocks: list, answers: list, personas: list,
                                all_tweets_flat: list, rng, sample_size: int = None,
                                format_block: str = "", tweets_per_sample: int = 3,
                                agent_ids: list = None):
    """Score tweet instruction quality via batched generation + batched teacher rating.

    Args:
        student_engine: _StudentEngine.
        teacher_engine: _TeacherEngine.
        instruction_text: system prompt being evaluated.
        prompts: parsed prompts JSON.
        blocks: tweet history per agent (used for context seeding).
        answers: ground-truth PHQ-9 score per agent.
        personas: persona text per agent.
        all_tweets_flat: pool of (tweet, agent_id) for neighbor sampling.
        rng: numpy Generator.
        sample_size: optional cap on the number of agents evaluated.
        format_block: fixed format block appended to every user message.
        tweets_per_sample: how many tweets each agent generates.
        agent_ids: optional agent ids aligned with `blocks`.

    Returns:
        (mean_score, std_score, per_phq9) where per_phq9[score] = {"avg_score", "n_samples", "n_empty"}.
    """
    if sample_size and len(blocks) > sample_size:
        idx = rng.choice(len(blocks), size=sample_size, replace=False)
        blocks     = [blocks[i]     for i in idx]
        answers    = [answers[i]    for i in idx]
        personas   = [personas[i]   for i in idx]
        agent_ids  = [agent_ids[i]  for i in idx] if agent_ids else None

    n = len(blocks)
    scores_by_phq9   = defaultdict(list)
    n_samples_by_phq9 = defaultdict(int)
    empty_by_phq9    = defaultdict(int)

    # Phase 1 — generate tweets. Each agent has its own growing context seeded with 0-4
    # historical tweets (matching training); batching across agents keeps contexts isolated.
    all_parsed = [[] for _ in range(n)]   # all_parsed[sample_idx][tweet_idx]

    sample_contexts = []
    for tweet_block in blocks:
        valid = [t for t in tweet_block if t and t not in {"NO_POST", "NO_TWEET"}]
        n_ctx = int(rng.integers(0, min(len(valid), 4) + 1)) if valid else 0
        if n_ctx > 0:
            ctx_idx = rng.choice(len(valid), size=n_ctx, replace=False)
            sample_contexts.append([valid[k] for k in sorted(ctx_idx)])
        else:
            sample_contexts.append([])

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
            # Grow this agent's context with the tweet just generated
            if parsed:
                sample_contexts[i].append(parsed)
            if j == 0 and i == 0:
                print(f"  --- raw output [NON-THINKING student] (eval, sample 0, tweet 1) ---")
                print(response)
                print(f"  --- parsed: {parsed!r} ---")

    # Phase 2 — build rating prompts; record all-empty samples as score 0.
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
            f"4. Diversity: Posts cover somewhat different topics or moods. Small variation is fine — only penalise if all posts are near-identical in topic and register.\n"
            f"5. Interaction: Genuine engagement rewarded. Hollow @mentions and all-reply sets penalised.\n\n"
            f"Score proportionally: partial credit when only some criteria are met; "
            f"0 only for posts that are unreadable or completely empty; 10 only when all criteria are fully met.\n\n"
            f"Respond with exactly two lines:\n"
            f"SCORE: <number 0-10>\n"
            f"FEEDBACK: <one sentence>"
        )
        pending.append((i, phq9, rating_prompt))

    # Phase 3 — one batched teacher call rates all pending sets.
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
    max_instruction_words: int = 200,
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
    """Optimise the tweet-generation system prompt via TextGrad.

    Student (non-thinking) generates tweets; teacher (thinking) rates each set standalone — no reference
    tweets. Neighbor posts (max 6) are sampled from the combined inter/no_inter pool for context.

    Args:
        file_paths: tweets_with_phq9 files (.csv or .txt) to draw data from.
        model_name: HuggingFace model id used as both student and teacher.
        val_fraction: fraction held out for validation.
        test_fraction: fraction held out for test.
        seed: RNG seed.
        batch_size: agents per gradient-accumulation step.
        tweets_per_sample: tweets each agent generates per training/eval sample.
        max_instruction_words: soft length budget enforced via the optimizer constraint.
        num_steps: training steps; defaults to `len(train) // batch_size`, capped at 20.
        validate_every: validate every N steps (and at the last step).
        val_sample_size: number of agents in the periodic validation subset.
        test_sample_size: cap on the final test set.
        max_chars: per-tweet character budget injected into the prompt template.
        max_model_len: vLLM context budget.
        output_dir: where to write checkpoints, trajectory CSV, and plots.
        prompts_file: path to the prompts JSON (defaults to FC.PROMPTS_FILE).
        **vllm_kwargs: forwarded to ChatVLLM.

    Returns:
        The best instruction string seen (also written to `best_instruction_tweet.txt`).
    """
    rng = np.random.default_rng(seed)
    if output_dir is None:
        model_short = model_name.split("/")[-1]
        output_dir = f"data/test_post/optimized_tweets/{model_short}_seed{seed}"

    train_data, val_data, test_data = train_val_test_split(
        rng, file_paths, val_fraction, test_fraction,
    )
    train_blocks, train_answers, train_personas, train_agent_ids = train_data
    val_blocks, val_answers, val_personas, val_agent_ids = val_data
    test_blocks, test_answers, test_personas, test_agent_ids = test_data

    # Flat pool of all (tweet, agent_id) tuples for neighbor sampling
    all_tweets_flat = [
        (t, aid)
        for block, aid in zip(
            train_blocks + val_blocks + test_blocks,
            train_agent_ids + val_agent_ids + test_agent_ids,
        )
        for t in block if t and t not in ("NO_POST", "NO_TWEET")
    ]

    tp = vllm_kwargs.pop("tensor_parallel_size", None) or len((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(","))
    student_engine, teacher_engine = _build_engines(model_name, tp, 0.90, max_model_len=max_model_len, **vllm_kwargs)

    # Load prompts and split into optimisable instruction vs fixed format/constraints.
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
            "Length constraints and output format are fixed and "
            "appended separately: do NOT duplicate or alter them here."
        ),
        requires_grad=True,
    )

    optimizer = tg.TGD(
        engine=teacher_engine,
        optimizer_system_prompt=_OPTIMIZER_SYSTEM_PROMPT,
        parameters=[instruction],
        constraints=[
            f"Keep the instruction under {max_instruction_words} words.",
            "Do NOT repeat phrases or ideas already stated. Each sentence must add unique value.",
            "Focus on tone/mood calibration, content diversity, and originality guidance within ### RULES ###.",
            "Do NOT include constraints, length limits, or output format: those are fixed elsewhere.",
            "Tone scales with PHQ-9: low → positive/upbeat, high → apathetic/irritable/overwhelmed/raw. High PHQ-9 negativity is correct — only prevent flattening all agents to the same tone.",
            "Do NOT use poetic, lyrical, or metaphorical language in the instruction itself. Write plainly and directly.",
        ],
        gradient_memory=0,
    )

    if num_steps is None:
        num_steps = min(20, max(1, len(train_blocks) // batch_size))

    best_val_score = -float("inf")
    best_instruction = instruction.value

    # Trajectory CSV — written incrementally so partial runs stay usable.
    model_short = model_name.split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)
    trajectory_path = os.path.join(output_dir, "training_trajectory.csv")
    _traj_fields = ["model", "seed", "step", "split", "mean_score", "std_score", "n_samples"]
    with open(trajectory_path, "w", newline="", encoding="utf-8") as _fh:
        csv.DictWriter(_fh, fieldnames=_traj_fields).writeheader()

    # step=0: evaluate and checkpoint the initial instruction as the first best
    _s0, _s0_std, _ = _evaluate_tweet_instruction(
        student_engine, teacher_engine, instruction.value, prompts,
        val_blocks, val_answers, val_personas,
        all_tweets_flat, rng, sample_size=val_sample_size,
        format_block=format_block_tweet, tweets_per_sample=tweets_per_sample,
        agent_ids=val_agent_ids,
    )
    with open(trajectory_path, "a", newline="", encoding="utf-8") as _fh:
        csv.DictWriter(_fh, fieldnames=_traj_fields).writerow({
            "model": model_short, "seed": seed, "step": 0,
            "split": "val", "mean_score": round(_s0, 4),
            "std_score": round(_s0_std, 4), "n_samples": val_sample_size or len(val_blocks),
        })
    print(f"[Step 0] Baseline val score: {_s0:.2f}/10 ± {_s0_std:.2f}")
    best_val_score = _s0
    best_instruction = instruction.value
    with open(os.path.join(output_dir, "best_instruction_tweet.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# val score: {_s0:.2f}/10  (step 0)\n\n{best_instruction}")
    with open(os.path.join(output_dir, "best_full_prompt_tweet.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"=== SYSTEM PROMPT ===\n{best_instruction}\n\n"
                 f"=== FORMAT BLOCK (fixed) ===\n{format_block_tweet}")

    # Training loop.
    for step in range(num_steps):
        batch_idx = rng.choice(
            len(train_blocks),
            size=min(batch_size, len(train_blocks)),
            replace=False,
        )
        batch_blocks      = [train_blocks[i]      for i in batch_idx]
        batch_answers     = [train_answers[i]     for i in batch_idx]
        batch_personas    = [train_personas[i]    for i in batch_idx]
        batch_agent_ids   = [train_agent_ids[i]   for i in batch_idx]

        optimizer.zero_grad()
        model = tg.BlackboxLLM(student_engine, system_prompt=instruction)

        train_scores = []
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

            # Combine all tweets into one Variable so the teacher rates the full set
            # in a single backward call (one per tweet would just redundantly update
            # the same instruction Variable).
            combined = tg.Variable(
                "\n".join(raw_tweets),
                role_description="Set of generated posts for this sample (PHQ-9 and persona context)",
                requires_grad=False,
            )
            loss_fn = tg.TextLoss(_make_loss_prompt_tweet_set(parsed_tweets, persona, phq9))
            loss = loss_fn(combined)

            for line in loss.value.splitlines():
                if line.strip().upper().startswith("SCORE:"):
                    try:
                        train_scores.append(min(10.0, max(0.0, float(re.search(r"[\d.]+", line).group()))))
                    except (AttributeError, ValueError):
                        pass
                    break

            loss.backward()

        grads = instruction.gradients
        if grads:
            print(f"  [teacher gradient]: {list(grads)[0].value[:400]}\n")

        # See PHQ-9 branch — guard against TextGrad's TGD parser crashing on
        # an empty/malformed optimizer response.
        try:
            optimizer.step()
        except (IndexError, ValueError) as exc:
            print(f"  [optimizer] step skipped — response unparseable ({exc.__class__.__name__}): instruction left unchanged")

        if train_scores:
            train_arr = np.array(train_scores)
            with open(trajectory_path, "a", newline="", encoding="utf-8") as _fh:
                csv.DictWriter(_fh, fieldnames=_traj_fields).writerow({
                    "model": model_short, "seed": seed, "step": step + 1,
                    "split": "train", "mean_score": round(float(train_arr.mean()), 4),
                    "std_score": round(float(train_arr.std()), 4), "n_samples": len(train_scores),
                })

        # Guard against the teacher returning TextGrad's literal "{improved variable}" placeholder.
        _val = instruction.value.strip()
        if not _val or _val == "{improved variable}" or len(_val) < 20:
            print(f"  -> Optimizer returned placeholder ({_val!r}) — resetting to best instruction.")
            instruction.value = best_instruction

        # 230-word tolerance keeps a closing sentence intact; only trims when genuinely over-long.
        hard_limit = max_instruction_words + 230
        words = instruction.value.split()
        if len(words) > hard_limit:
            print(f"  [trunc] instruction was {len(words)} words, trimming to ~{hard_limit}")
            truncated = " ".join(words[:hard_limit])
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
                agent_ids=val_agent_ids,
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

    # Final test evaluation on the best instruction.
    test_score, test_std, per_phq9 = _evaluate_tweet_instruction(
        student_engine, teacher_engine, best_instruction, prompts,
        test_blocks, test_answers, test_personas,
        all_tweets_flat, rng, sample_size=test_sample_size,
        format_block=format_block_tweet, tweets_per_sample=tweets_per_sample,
        agent_ids=test_agent_ids,
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

    # Save results.
    os.makedirs(output_dir, exist_ok=True)

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

    plot_optimizer_trajectory(
        trajectory_path, output_dir,
        title=f"Tweet optimizer — {model_short} seed={seed}",
        mode="tweets",
    )
    plot_test_scores_by_phq9(
        {k: {"avg_score": v["avg_score"], "n_samples": v["n_samples"]} for k, v in per_phq9.items()},
        output_dir,
        title=f"Test score by PHQ-9 — {model_short} seed={seed}",
        mode="tweets",
    )

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
    """Build an 80/10/10 train/val/test split of tweet blocks from a tweets_with_phq9.txt file.

    Args:
        file_path: path to a tweets_with_phq9.txt file.

    Returns:
        ([train_blocks, val_blocks, test_blocks], [train_answers, val_answers, test_answers]).
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
    """Split embeddings + labels into train/val/test tensors (remainder is test).

    Args:
        rng: numpy Generator for the permutation.
        embeddings: per-block centroid tensor.
        labels: ground-truth PHQ-9 scores (converted to float tensor if needed).
        train_frac: fraction for training.
        val_frac: fraction for validation (test gets `1 - train_frac - val_frac`).

    Returns:
        (train_embs, val_embs, test_embs, train_labels, val_labels, test_labels).
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


def setup_BERT_model(tweet_blocks, model, device):
    """Encode each tweet block into a centroid vector (mean ∥ max ∥ std of token embeddings).

    Args:
        tweet_blocks: list of tweet blocks (one per agent).
        model: an SBERT/MentalBERT encoder.
        device: torch device.

    Returns:
        Centroid tensor of shape (n_blocks, 3 * embedding_dim).
    """
    centroids = []
    for tweet_block in tweet_blocks:
        embeddings = create_embedding(model, tweet_block).to(device)
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
    """Load cached embeddings, split 80/10/10, train the regressor, and save model + metrics.

    Args:
        embeddings_path: path to a `.pt` containing either {embeddings, labels} or pre-split arrays.
        base_model_name: model id used to construct the save directory.
        device: torch device.
        mental_bert: True if embeddings came from MentalBERT (controls regressor input size).
        split_seed: RNG seed for the train/val/test split.

    Returns:
        (best_model, best_val_loss, test_loss).
    """
    with open(embeddings_path, "rb") as f:
        data = torch.load(f)
    
    rng = np.random.default_rng(split_seed)

    if "embeddings" in data and "labels" in data:
        # Single-array format: split after loading.
        all_embs = data["embeddings"]
        all_labels = data["labels"]
        train_embs, val_embs, test_embs, train_labels, val_labels, test_labels = split_embeddings_and_labels(
            rng, all_embs, all_labels, train_frac=0.8, val_frac=0.1
        )
    else:
        # Legacy pre-split format.
        train_embs = data["train_embs"]
        val_embs = data["val_embs"]
        test_embs = data["test_embs"]
        train_labels = data["train_labels"]
        val_labels = data["val_labels"]
        test_labels = data["test_labels"]

    train_embs  = train_embs.to(device);  train_labels  = train_labels.to(device)
    val_embs    = val_embs.to(device);    val_labels    = val_labels.to(device)
    test_embs   = test_embs.to(device);   test_labels   = test_labels.to(device)

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
    """MLP regressor mapping a centroid embedding (mean ∥ max ∥ std) to a PHQ-9 score.

    Args:
        mentalbert: True if input embeddings are 768-dim (MentalBERT) instead of 384-dim (SBERT).
        dropout_rate: dropout applied at the input and after the first hidden layer.
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
    """Train the MLP regressor with Huber loss and ReduceLROnPlateau.

    Args:
        model: regressor (already on device).
        train_data: train embeddings.
        train_labels: train PHQ-9 scores.
        val_data: val embeddings.
        val_labels: val PHQ-9 scores.
        device: torch device.
        epochs: max training epochs.
        batch_size: minibatch size.
        learning_rate: initial Adam learning rate.
        patience: currently unused (scheduler manages LR drops internally).

    Returns:
        (best_model, best_val_loss, epoch_history) where history holds per-epoch train/val losses.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=3)

    criterion = nn.HuberLoss(delta=1.0)
    train_data = torch.as_tensor(train_data, dtype=torch.float32).to(device)
    train_labels = torch.as_tensor(train_labels, dtype=torch.float32).to(device)

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
    """Run the regressor on `test_data` and return aggregated loss.

    Args:
        model: trained regressor.
        test_data: evaluation embeddings.
        test_labels: ground-truth PHQ-9 scores.
        device: torch device.
        mae: if True use L1 (MAE); otherwise Huber.

    Returns:
        Scalar loss value.
    """
    model.eval()
    if mae:
        criterion = nn.L1Loss()
    else:
        criterion = nn.HuberLoss(delta=1.0)

    test_data = torch.as_tensor(test_data, dtype=torch.float32).to(device)
    test_labels = torch.as_tensor(test_labels, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(test_data).squeeze(-1)
        loss = criterion(outputs, test_labels)
    return loss.item()

def save_embeddings_for_file(file_path, base_model_name: str, device, mentalbert: bool = False, out_dir=None):
    """Parse tweets_with_phq9 files, encode blocks with SBERT/MentalBERT, and cache embeddings + labels.

    The split is deferred until training (see `split_embeddings_and_labels`), so the cache stays
    seed-agnostic.

    Args:
        file_path: single path or list of paths to tweets_with_phq9 files (.csv or .txt).
        base_model_name: model id used to construct the default output directory.
        device: torch device for the encoder.
        mentalbert: if True use MentalBERT (768-dim); otherwise SBERT (384-dim).
        out_dir: optional override for the save directory.
    """
    paths = [file_path] if isinstance(file_path, str) else list(file_path)
    tweet_blocks, true_answers = [], []
    for fp in paths:
        if fp.endswith(".csv"):
            blocks, answers, _, _ = parse_tweets_with_phq9_csv(fp)
        else:
            blocks, answers = parse_tweets_with_phq9(fp)
        tweet_blocks.extend(blocks)
        true_answers.extend(answers)
    print(f"Loaded {len(tweet_blocks)} blocks from {len(paths)} file(s)")

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
    """Return sorted paths matching `{base_dir}/seed_*/{target_filename}`.

    Args:
        base_dir: condition directory containing per-seed subfolders.
        target_filename: file to find inside each seed_* folder.

    Returns:
        Sorted list of matching file paths.
    """
    search_pattern = os.path.join(base_dir, "seed_*", target_filename)
    found_paths = glob.glob(search_pattern)

    return sorted(found_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--val-sample-size", type=int, default=8)
    parser.add_argument("--test-sample-size", type=int, default=50)
    parser.add_argument("--model", type=str, default=QWEN_27,
                        help="HuggingFace model id (e.g. Qwen/Qwen3.5-27B)")
    parser.add_argument("--mode", type=str, default="tweets", choices=["tweets", "phq9", "bert"],
                        help="Which optimizer to run: tweets, phq9, or bert")
    args = parser.parse_args()

    create_new_embeddings = False
    mental_bert = True
    file_paths = []

    for inter in ["inter", "no_inter"]:
        base_dir = f"data/test_post/Qwen_Qwen3.5-27B/temp_0.8_top_p_0.6_cp_10_{inter}"
        file_paths.extend(_generate_file_path(base_dir))

    for fp in file_paths:
        print(fp)

    model_name = args.model
    base_model_name = model_name

    run_mode = args.mode

    if run_mode == "phq9":
        for seed in args.seeds:
            print(f"\n{'='*60}\nRunning PHQ-9 optimizer with seed={seed}\n{'='*60}")
            call_optimizer_phq9(
                file_paths,
                model_name=model_name,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                val_sample_size=args.val_sample_size,
                test_sample_size=args.test_sample_size,
                seed=seed,
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if create_new_embeddings:
            save_embeddings_for_file(file_paths, base_model_name, device, mentalbert=mental_bert)

        dir_name = "mentalbert_embeddings" if mental_bert else "sbert_embeddings"
        embeddings_path = os.path.join("data", "test", base_model_name, dir_name, "embeddings_and_labels.pt")
        train_BERT_model(embeddings_path, base_model_name, device, mental_bert=mental_bert, split_seed=args.seeds[0])