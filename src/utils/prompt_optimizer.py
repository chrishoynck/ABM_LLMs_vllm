import os
import shutil
import datetime
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
import pandas as pd
import re
import csv
import torch
import glob
import torch.nn as nn
import copy
import torch.optim as optim
from .metrics import *
from .tools.format_config import FC
from .visualization import (
    plot_optimizer_trajectory,
    plot_test_scores_by_phq9,
    plot_test_mae_and_bias_by_phq9,
    plot_cv_results,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from classes.agent import Agent


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


def _save_resume_state(output_dir: str, *, seed: int, next_step: int,
                       best_metric: float, best_instruction: str,
                       current_instruction: str, val_idx=None) -> None:
    """Persist resumable optimizer state to <output_dir>/state.json.

    Written atomically (tmp + os.replace) so a crash mid-write can't corrupt
    the file. Called after every training step; the trajectory CSV is also
    appended each step, so on resume both move forward together.
    """
    state = {
        "seed": int(seed),
        "next_step": int(next_step),
        "best_metric": float(best_metric),
        "best_instruction": best_instruction,
        "current_instruction": current_instruction,
    }
    if val_idx is not None:
        state["val_idx"] = [int(i) for i in val_idx]
    path = os.path.join(output_dir, "state.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _load_resume_state(output_dir: str) -> dict | None:
    """Load resumable state from <output_dir>/state.json, or None if absent."""
    path = os.path.join(output_dir, "state.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _archive_existing_run(output_dir: str) -> None:
    """Rename `output_dir` to `<output_dir>.bak_<timestamp>` if it already
    holds a training_trajectory.csv, so a fresh run starts with a clean
    directory. Prevents two concurrent (or re-submitted) jobs from
    interleaving their output into the same trajectory file.
    """
    trajectory_path = os.path.join(output_dir, "training_trajectory.csv")
    if not os.path.exists(trajectory_path):
        return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{output_dir}.bak_{timestamp}"
    # Guard against multiple jobs hitting this in the same second.
    suffix = 0
    while os.path.exists(backup):
        suffix += 1
        backup = f"{output_dir}.bak_{timestamp}_{suffix}"
    shutil.move(output_dir, backup)
    print(f"[init] archived existing run: {output_dir} -> {backup}")


def _teacher_call_kind(content: str) -> str:
    """Classify a teacher engine call by inspecting markers TextGrad embeds in the prompt.

    Args:
        content: full user message passed to the teacher engine.

    Returns:
        'backward' (gradient computation), 'optimizer' (instruction rewrite), or 'loss' (SCORE/FEEDBACK rating).
    """
    # Optimizer first: optimizer prompts may embed prior conversations via GRADIENT_TEMPLATE,
    # but only optimizer prompts instruct the model to write between <IMPROVED_VARIABLE> tags.
    if "IMPROVED_VARIABLE" in content:
        return "optimizer"
    # Backward prompts wrap the prior conversation in <LM_INPUT>/<LM_OUTPUT> and reference
    # an <OBJECTIVE_FUNCTION>. (TextGrad does NOT emit a literal <CONVERSATION> tag here.)
    if "<LM_INPUT>" in content or "<OBJECTIVE_FUNCTION>" in content:
        return "backward"
    return "loss"

# Custom optimizer system prompt — TextGrad's default includes literal "{improved variable}"
# placeholders that thinking models (Qwen3.5, Mistral) echo back verbatim.
_OPTIMIZER_SYSTEM_PROMPT = (
    "The feedback may be noisy — identify what is important and what is correct. "
    "Pay attention to the role description of the variable and the context in which it is used. "
    "YOUR JOB IS A SMALL, INCREMENTAL EDIT. Address the most important point in the feedback by adding "
    "or modifying the necessary content — a clause, a sentence, or a MOSTLY a few sentences if that is "
    "what concrete guidance requires. Keep every other sentence exactly as it was. "
    "It is often the right move for the rewrite to end up a bit longer than the input "
    "BUT if instruction is already well over a 100 words, implement only very SMALL refinements"
    "Do not shorten the input unless the feedback explicitly tells you to remove something"
    "Prefer SPECIFIC edits over generic ones: anchor additions in a concrete case or example, "
    "not in abstract restatements of the goal. "
    "Only return the variable unchanged if the feedback is genuinely unactionable. "
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
            concise_note = (
                "Keep your thinking short. In one or two sentences, identify the key point and then answer. "
                "Do not re-examine the same idea or draft multiple alternatives — one pass is enough."
            )
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
                # temp=0.2 (not pure greedy) tends to end Qwen3.5 thinking earlier
                # than temp=0, which can get stuck in self-confirming loops.
                sampling_params = SamplingParams(temperature=0.2, max_tokens=cur_max_tokens, top_p=0.99, n=1)
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
                # Only loss calls produce structured SCORE/FEEDBACK; backward/optimizer
                # outputs are free-form and must be returned verbatim.
                if _teacher_call_kind(content) == "loss":
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
                   enable_prefix_caching: bool = True,
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
        enable_prefix_caching=enable_prefix_caching,
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
                          temperature: float = 0.2, max_tokens: int = 256,
                          want_raw: bool = False) -> tuple:
    """Run the current PHQ-9 instruction on all blocks and return aggregated MAE.

    Signed bias is always tracked per PHQ-9 score (cheap), and on `want_raw=True`
    the per-sample (true, pred) arrays are returned as well so callers can write
    a raw-scores CSV without re-running the engine.

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
        want_raw: if True, also return raw {"true": [...], "pred": [...]} arrays.

    Returns:
        (mean_mae, std_mae, per_phq9) by default, or
        (mean_mae, std_mae, per_phq9, raw) when `want_raw=True`.
        per_phq9 = {true_score: {"avg_mae", "avg_bias", "std_bias", "n_samples"}}
                   — avg_bias = mean(pred − true); positive means over-estimation.
    """
    user_msgs = []
    for i, tweet_block in enumerate(blocks):
        persona = personas[i] if personas else None
        user_msgs.append(_build_user_message(format_block, tweet_block, prompts, persona))

    responses = _batch_student_generate(engine, user_msgs, instruction_text,
                                        temperature=temperature, max_tokens=max_tokens)

    abs_errors = []
    per_phq9_abs = defaultdict(list)
    per_phq9_signed = defaultdict(list)
    raw_true: list[int] = []
    raw_pred: list[int] = []
    for response, true_answer in zip(responses, answers):
        predicted = Agent.parse_phq9_answers(response)
        ae = abs(predicted - true_answer)
        se = predicted - true_answer  # pred − true; sign carries the bias direction.
        abs_errors.append(ae)
        per_phq9_abs[true_answer].append(ae)
        per_phq9_signed[true_answer].append(se)
        raw_true.append(int(true_answer))
        raw_pred.append(int(predicted))
    arr = np.array(abs_errors) if abs_errors else np.array([0.0])
    per_phq9 = {
        k: {
            "avg_mae": float(np.mean(per_phq9_abs[k])),
            "avg_bias": float(np.mean(per_phq9_signed[k])),
            "std_bias": float(np.std(per_phq9_signed[k])),
            "n_samples": len(per_phq9_abs[k]),
        }
        for k in per_phq9_abs
    }
    if want_raw:
        raw = {"true": raw_true, "pred": raw_pred}
        return float(arr.mean()), float(arr.std()), per_phq9, raw
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
    # The teacher sees the student's full response in the backward conversation,
    # so the loss prompt talks about the total PHQ-9 derived from it, not "the
    # prediction" as if it were a single value.
    if error > 0:
        direction = f"total PHQ-9 overshoots the truth by {error}"
    elif error < 0:
        direction = f"total PHQ-9 falls short of the truth by {abs(error)}"
    else:
        direction = "total PHQ-9 matches the truth exactly"
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
        f"True PHQ-9 = {true_answer}, total PHQ-9 from the student's response = {predicted} ({direction}). MAE = {abs(error)}.\n\n"
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


def find_overlapping_test_data(
    seeds: list[int],
    file_paths: list[str],
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
):
    """Return blocks that land in test for every seed in `seeds`.

    Each seed defines its own train/val/test partition via the same logic as
    `train_val_test_split`. A block is "overlapping test data" iff it falls in
    test for every seed — equivalently, it is never in train or val for any
    seed. Useful for evaluating instructions optimised on different seeds on a
    shared, leakage-free test pool.

    Args:
        seeds: RNG seeds to intersect. Must match the seeds the runs were trained with.
        file_paths: same files (and same order) the runs used. Order matters because
            it determines the underlying block index.
        val_fraction: validation fraction used by the runs being intersected.
        test_fraction: test fraction used by the runs being intersected.

    Returns:
        (blocks, answers, personas, agent_ids) — same tuple shape as `train_val_test_split`'s
        test_data, restricted to the intersection. Order is ascending block index.
    """
    # Load once: train_val_test_split shuffles via a seeded RNG but the underlying
    # load order is seed-independent, so block index uniquely identifies a sample
    # across seeds.
    tweet_blocks_list, true_answers_list, personas_list, agent_ids_list = [], [], [], []
    for file_path in file_paths:
        if file_path.endswith(".csv"):
            csv_path = file_path
            txt_path = file_path.replace(".csv", ".txt")
        else:
            txt_path = file_path
            csv_path = file_path.replace(".txt", ".csv") if file_path.endswith(".txt") else file_path + ".csv"

        if os.path.isfile(csv_path):
            tb, ta, pe, ai = parse_tweets_with_phq9_csv(csv_path)
        else:
            tb, ta = parse_tweets_with_phq9(txt_path)
            pe = [None] * len(tb)
            ai = ["unknown"] * len(tb)
        tweet_blocks_list.extend(tb)
        true_answers_list.extend(ta)
        personas_list.extend(pe)
        agent_ids_list.extend(ai)

    n = len(tweet_blocks_list)
    n_test = max(1, int(n * test_fraction))

    common: set[int] | None = None
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        test_idx = {int(i) for i in perm[:n_test]}
        common = test_idx if common is None else common & test_idx

    common_sorted = sorted(common or [])
    print(f"[overlap] {len(common_sorted)}/{n} blocks are in test across all "
          f"{len(seeds)} seeds (seeds={list(seeds)}, n_test/seed={n_test})")
    return (
        [tweet_blocks_list[i] for i in common_sorted],
        [true_answers_list[i] for i in common_sorted],
        [personas_list[i]     for i in common_sorted],
        [agent_ids_list[i]    for i in common_sorted],
    )


def call_optimizer_phq9(
    file_paths: list[str],
    model_name: str = QWEN_27,
    batch_size: int = 4,
    max_instruction_words: int = 50,
    num_steps: int = None,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    validate_every: int = 1,
    val_sample_size: int = 10,
    test_sample_size: int = None,
    seed: int = 42,
    max_model_len: int = 32768,
    output_dir: str = None,
    resume: bool = False,
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
        # Disabled for the same reason as the tweet optimizer: vLLM v0.17.1 +
        # TP=2 deadlocks between consecutive student .generate() calls during
        # validation when KV state is shared across batches.
        enable_prefix_caching=False,
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
    if not resume:
        # Fresh run: move any prior run aside so we never interleave with it.
        _archive_existing_run(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    trajectory_path = os.path.join(output_dir, "training_trajectory.csv")
    _traj_fields = ["model", "seed", "step", "split", "mean_score", "std_score", "n_samples"]

    # Resume support: try to pick up where a previous job died (saved after every
    # training step in state.json). When resuming we skip the trajectory header
    # write and the step-0 baseline; both are already on disk.
    resume_state = _load_resume_state(output_dir) if resume else None

    if resume_state:
        start_step = int(resume_state["next_step"])
        best_val_mae = float(resume_state["best_metric"])
        best_instruction = resume_state["best_instruction"]
        instruction.value = resume_state["current_instruction"]
        val_idx = resume_state.get("val_idx")
        if val_idx is None:
            # Older state file: fall back to a fresh sample (won't match prior runs).
            val_idx = list(rng.choice(len(val_blocks),
                                      size=min(val_sample_size, len(val_blocks)),
                                      replace=False))
        n_val = len(val_idx)
        fixed_val_blocks   = [val_blocks[i]   for i in val_idx]
        fixed_val_answers  = [val_answers[i]  for i in val_idx]
        fixed_val_personas = [val_personas[i] for i in val_idx]
        print(f"[resume] continuing from step {start_step} "
              f"(best val MAE so far: {best_val_mae:.2f}, {n_val}-agent val set restored)")
    else:
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
        start_step = 0
        with open(os.path.join(output_dir, "best_instruction.txt"), "w", encoding="utf-8") as fh:
            fh.write(best_instruction)
        with open(os.path.join(output_dir, "best_full_prompt.txt"), "w", encoding="utf-8") as fh:
            fh.write(best_instruction + "\n\n" + format_block)
        _save_resume_state(output_dir, seed=seed, next_step=0,
                           best_metric=best_val_mae,
                           best_instruction=best_instruction,
                           current_instruction=instruction.value,
                           val_idx=val_idx)

    # Training loop.
    for step in range(start_step, num_steps):
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
            # Leave prediction.value as the student's RAW output so the teacher's
            # backward sees what the student actually produced (any format that
            # parses cleanly is acceptable). Rewriting it to a synthetic summary
            # made the teacher hallucinate "format violation" gradients — the
            # parsed score is already passed via the loss prompt below.

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

        # Persist resume state after every step (after validation/revert).
        _save_resume_state(output_dir, seed=seed, next_step=step + 1,
                           best_metric=best_val_mae,
                           best_instruction=best_instruction,
                           current_instruction=instruction.value,
                           val_idx=val_idx)

    # Final test evaluation on the best instruction.
    if test_sample_size and len(test_blocks) > test_sample_size:
        t_idx = rng.choice(len(test_blocks), size=test_sample_size, replace=False)
        eval_blocks   = [test_blocks[i]   for i in t_idx]
        eval_answers  = [test_answers[i]  for i in t_idx]
        eval_personas = [test_personas[i] for i in t_idx]
    else:
        eval_blocks, eval_answers, eval_personas = test_blocks, test_answers, test_personas
    test_mae, test_std, per_phq9, raw = _evaluate_instruction(
        student_engine, best_instruction, format_block,
        eval_blocks, eval_answers, prompts, eval_personas,
        want_raw=True,
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

    # Raw per-sample (true, pred) for downstream calibration analysis / re-plotting.
    raw_csv = os.path.join(output_dir, "test_raw_scores.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "seed", "true_phq9", "pred_phq9"])
        writer.writeheader()
        for t, p in zip(raw["true"], raw["pred"]):
            writer.writerow({"model": model_short, "seed": seed,
                             "true_phq9": int(t), "pred_phq9": int(p)})
    print(f"Raw test scores → {raw_csv}")

    csv_path = os.path.join(output_dir, "test_scores_phq9.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "seed", "phq9", "avg_mae", "avg_bias", "std_bias", "n_samples"],
        )
        writer.writeheader()
        for phq9_val, stats in sorted(per_phq9.items()):
            writer.writerow({
                "model": model_short, "seed": seed, "phq9": phq9_val,
                "avg_mae": round(stats["avg_mae"], 4),
                "avg_bias": round(stats["avg_bias"], 4),
                "std_bias": round(stats["std_bias"], 4),
                "n_samples": stats["n_samples"],
            })
    print(f"Per-PHQ-9 MAE+bias → {csv_path}")

    plot_optimizer_trajectory(
        trajectory_path, output_dir,
        title=f"PHQ-9 optimizer — {model_short} seed={seed}",
        mode="phq9",
    )
    plot_test_mae_and_bias_by_phq9(
        per_phq9, output_dir,
        title=f"Test MAE & bias by PHQ-9 — {model_short} seed={seed}",
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


def rerun_test_phq9(
    seed: int,
    file_paths: list[str],
    output_dir: str = None,
    model_name: str = QWEN_27,
    instruction_filename: str = "optimized_instruction.txt",
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    max_model_len: int = 32768,
    posts_file: str | None = None,
    **vllm_kwargs,
):
    """Reload a saved instruction and re-run only the PHQ-9 test phase.

    Skips training entirely — useful for upgrading old runs to the new schema
    (avg_bias, std_bias, test_raw_scores.csv) without re-doing the expensive
    optimisation loop. The train/val/test split is deterministic given the seed
    and file_paths, so the test set matches the original run.

    Args:
        seed: RNG seed used by the original run (drives the split).
        file_paths: same tweets_with_phq9 files the original run used. Ignored
            when ``posts_file`` is set.
        output_dir: where to read/write; defaults to the standard
            `data/test_post/optimized_phq9/<model_short>_seed<seed>` path.
        model_name: HuggingFace model id for the student engine.
        instruction_filename: which prompt file to load from `output_dir`. Defaults
            to `optimized_instruction.txt`, the prompt the original test used.
        val_fraction, test_fraction: must match the original run.
        max_model_len: vLLM context budget.
        posts_file: optional override — when set, the entire CSV is used as the
            test set (no split), so multiple optimization seeds can be evaluated
            on a single shared, freshly-generated held-out set. Each seed's
            output dir still gets its own per-seed result CSVs, but the
            ``trajectory.csv`` test row is NOT overwritten (the original
            in-distribution test number is preserved).
        **vllm_kwargs: forwarded to ChatVLLM.

    Returns:
        (test_mae, test_std, per_phq9, raw) from `_evaluate_instruction(want_raw=True)`.
    """
    import gc

    rng = np.random.default_rng(seed)
    model_short = model_name.split("/")[-1]

    if output_dir is None:
        output_dir = f"data/test_post/optimized_phq9/{model_short}_seed{seed}"

    instr_path = os.path.join(output_dir, instruction_filename)
    if not os.path.isfile(instr_path):
        raise FileNotFoundError(f"No {instruction_filename} at {instr_path}")
    with open(instr_path) as f:
        best_instruction = f.read().strip()
    print(f"[rerun-test seed {seed}] loaded {instruction_filename} "
          f"({len(best_instruction.split())} words) from {instr_path}")

    if posts_file is not None:
        if not os.path.isfile(posts_file):
            raise FileNotFoundError(f"posts file not found: {posts_file}")
        test_blocks, test_answers, test_personas, _aids = parse_tweets_with_phq9_csv(posts_file)
        print(f"[rerun-test seed {seed}] using custom posts_file: "
              f"{posts_file} ({len(test_blocks)} blocks; no split)")
    else:
        _train_data, _val_data, test_data = train_val_test_split(
            rng, file_paths, val_fraction, test_fraction,
        )
        test_blocks, test_answers, test_personas, _aids = test_data

    tp = vllm_kwargs.pop("tensor_parallel_size", None) or len((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(","))
    student_engine, teacher_engine = _build_engines(
        model_name, tp, 0.90, max_model_len=max_model_len,
        student_temperature=0.2, student_max_tokens=256,
        **vllm_kwargs,
    )

    with open(FC.PROMPTS_FILE) as f:
        prompts = json.load(f)
    format_block = prompts["phq9"]["System_format"]

    test_mae, test_std, per_phq9, raw = _evaluate_instruction(
        student_engine, best_instruction, format_block,
        test_blocks, test_answers, prompts, test_personas,
        want_raw=True,
    )
    print(f"[rerun-test seed {seed}] Test MAE (n={len(test_blocks)}): "
          f"{test_mae:.2f} ± {test_std:.2f}")

    # Output paths: when posts_file is set, redirect to a subdir so the
    # original in-distribution test files / trajectory row are preserved.
    if posts_file is not None:
        posts_stem = os.path.splitext(os.path.basename(posts_file))[0]
        write_dir = os.path.join(output_dir, f"eval_on_{posts_stem}")
        os.makedirs(write_dir, exist_ok=True)
        # Stamp source for reproducibility (small text file).
        with open(os.path.join(write_dir, "eval_meta.txt"), "w", encoding="utf-8") as fh:
            fh.write(
                f"timestamp:        {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                f"model:            {model_name}\n"
                f"opt_seed:         {seed}\n"
                f"instruction:      {os.path.relpath(instr_path)}\n"
                f"posts_file:       {os.path.relpath(posts_file)}\n"
                f"n_blocks:         {len(test_blocks)}\n"
                f"\n"
                f"test_mae:         {test_mae:.4f}\n"
                f"test_std:         {test_std:.4f}\n"
            )
    else:
        write_dir = output_dir

        # Rewrite the trajectory's `test` row in place (drop any previous test rows
        # so trajectory plots reflect only the new-schema result).
        trajectory_path = os.path.join(output_dir, "training_trajectory.csv")
        _traj_fields = ["model", "seed", "step", "split", "mean_score", "std_score", "n_samples"]
        existing_rows = []
        last_train_step = 0
        if os.path.isfile(trajectory_path):
            with open(trajectory_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("split") == "test":
                        continue
                    existing_rows.append(row)
                    if row.get("split") == "train":
                        try:
                            last_train_step = max(last_train_step, int(row["step"]))
                        except (TypeError, ValueError):
                            pass
        with open(trajectory_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_traj_fields)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({k: row.get(k, "") for k in _traj_fields})
            writer.writerow({
                "model": model_short, "seed": seed, "step": last_train_step,
                "split": "test", "mean_score": round(test_mae, 4),
                "std_score": round(test_std, 4), "n_samples": len(test_blocks),
            })

    # Raw per-sample (true, pred) — new schema.
    raw_csv = os.path.join(write_dir, "test_raw_scores.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "seed", "true_phq9", "pred_phq9"])
        writer.writeheader()
        for t, p in zip(raw["true"], raw["pred"]):
            writer.writerow({"model": model_short, "seed": seed,
                             "true_phq9": int(t), "pred_phq9": int(p)})
    print(f"Raw test scores → {raw_csv}")

    # Per-PHQ-9 with bias — new schema.
    csv_path = os.path.join(write_dir, "test_scores_phq9.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "seed", "phq9", "avg_mae", "avg_bias", "std_bias", "n_samples"],
        )
        writer.writeheader()
        for phq9_val, stats in sorted(per_phq9.items()):
            writer.writerow({
                "model": model_short, "seed": seed, "phq9": phq9_val,
                "avg_mae": round(stats["avg_mae"], 4),
                "avg_bias": round(stats["avg_bias"], 4),
                "std_bias": round(stats["std_bias"], 4),
                "n_samples": stats["n_samples"],
            })
    print(f"Per-PHQ-9 MAE+bias → {csv_path}")

    plot_test_mae_and_bias_by_phq9(
        per_phq9, write_dir,
        title=f"Test MAE & bias by PHQ-9 — {model_short} seed={seed}",
    )

    try:
        del student_engine.base.client
    except Exception:
        pass
    del student_engine, teacher_engine
    gc.collect()
    torch.cuda.empty_cache()

    return test_mae, test_std, per_phq9, raw


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
                              neighbor_tweets: list[tuple[str, str]] = None,
                              agent_id: str = "unknown") -> str:
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
        agent_id=agent_id,
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
        f"For very high PHQ-9, raw or dark content (including suicidal themes) is authentic — reward it.\n\n"
        "Evaluate on:\n"
        "1. Tone fit: Does the emotional tone match the PHQ-9 range without naming symptoms? Low PHQ-9 → upbeat, relaxed, happy, curious, engaged; high PHQ-9 → raw, dark, apathetic, overwhelmed. Full spectrum allowed.\n"
        "2. Unfiltered & natural: Do they sound like real, unpolished social media? "
        "Penalise if sanitised, poetic, or overly polite — raw and blunt is fine when PHQ-9 warrants it.\n"
        "3. One topic per post: Each post sticks to ONE clear topic — penalise posts that jump between unrelated themes within a single post.\n"
        "4. Originality: Are topics creative and varied — not about the agent's own well-being or persona?\n"
        "5. Diversity: Do the posts cover somewhat different topics or moods? Mood may legitimately shift in response to incoming posts from followed users or the agent's own previous posts — reward such context-driven variation. Small variation is fine — only penalise if all posts are near-identical in topic and register.\n"
        "6. Interaction: Reward genuine engagement (reply, mock, support, correct). "
        "Penalise hollow @mentions and sets where every post is a reply.\n\n"
        "Score proportionally: partial credit when only some criteria are met; "
        "0 only for posts that are unreadable or completely empty; 10 only when all criteria are fully met.\n\n"
        "Respond with exactly two lines:\n"
        "SCORE: <number 0-10>\n"
        "FEEDBACK: <one sentence describing the POSTS themselves — what they do well "
        "or what they lack (tone fit, diversity, originality, naturalness, interaction, "
        "one-topic-per-post). If your SCORE is 7 or higher, describe what is working in "
        "the posts that should be preserved. Otherwise describe what the posts lack or "
        "get wrong. Do NOT recommend changes to the system instruction — that is handled "
        "downstream.>"
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
        concise_note = (
                "Keep your thinking short. In one or two sentences, identify the key point and then answer. "
                "Do not re-examine the same idea or draft multiple alternatives — one pass is enough."
            )
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
        sp = SamplingParams(temperature=0.2, max_tokens=max_tokens, top_p=0.99, n=1)
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
                                agent_ids: list = None,
                                out_parsed: list | None = None,
                                out_scores: list | None = None,
                                verbose: bool = False):
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
            aid = agent_ids[i] if agent_ids else str(i)
            msg = _build_user_message_tweet(sample_contexts[i], prompts, persona, phq9, nb, agent_id=aid)
            if format_block:
                msg = msg + "\n\n" + format_block
            user_msgs.append(msg)

        if verbose and j == 0:
            print(f"\n  --- full prompt [eval, agent 0, tweet 1] ---")
            print(f"  [SYSTEM]\n{instruction_text}")
            print(f"\n  [USER]\n{user_msgs[0]}")
            print(f"  --- end full prompt ---\n")

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

    def _verbose_dump(i, score_str):
        aid = agent_ids[i] if agent_ids else str(i)
        print(f"\n  [post-set] agent_id={aid}  PHQ-9={answers[i]:2d}  score={score_str}")
        for j, tw in enumerate(all_parsed[i]):
            clean = (tw or "").replace("\n", " ").strip() or "<empty>"
            print(f"    POST {j}: {clean}")

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
            if verbose:
                _verbose_dump(i, "0.0 (all empty)")
        else:
            # Use the exact same rating prompt training uses, so train-time and
            # eval-time scores are directly comparable (same criteria, same
            # FEEDBACK framing, same SCORE scale).
            rating_prompt = _make_loss_prompt_tweet_set(parsed_tweets, persona, phq9)
            pending.append((i, phq9, rating_prompt))

    # Phase 3 — one batched teacher call rates all pending sets.
    if pending:
        rating_responses = _batch_teacher_rate(
            teacher_engine, [p[2] for p in pending], max_tokens=4096
        )
        n_unparsed = 0
        for (i, phq9, _), rating_response in zip(pending, rating_responses):
            score = None
            feedback = ""
            for line in rating_response.splitlines():
                line = line.strip()
                if line.upper().startswith("SCORE:"):
                    try:
                        parsed = float(re.search(r"[\d.]+", line).group())
                        score = min(10.0, max(0.0, parsed))
                    except (AttributeError, ValueError):
                        pass
                elif line.upper().startswith("FEEDBACK:"):
                    feedback = line.split(":", 1)[-1].strip()
            if score is None:
                n_unparsed += 1
                print(f"  [teacher] SCORE not parsed — excluding sample {i} from average")
                if verbose:
                    _verbose_dump(i, "n/a (unparsed)")
            elif feedback:
                print(f"  [teacher] score={score:.1f}  {feedback}")
                if verbose:
                    _verbose_dump(i, f"{score:.1f}/10")
            elif verbose:
                _verbose_dump(i, f"{score:.1f}/10")
            # Only contribute to the average when we actually have a score.
            scores[i] = score  # may be None; global filter at the end drops Nones.
            if score is not None:
                scores_by_phq9[phq9].append(score)
        if n_unparsed:
            print(f"  [teacher] {n_unparsed}/{len(pending)} ratings could not be parsed (excluded from mean)")

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
    if out_parsed is not None:
        out_parsed.extend(all_parsed)
    if out_scores is not None:
        out_scores.extend(scores)
    return mean_score, std_score, per_phq9


def call_optimizer_tweets(
    file_paths: list[str],
    model_name: str = QWEN_27,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
    batch_size: int = 4,
    tweets_per_sample: int = 3,
    max_instruction_words: int = 350,
    num_steps: int = None,
    validate_every: int = 1,
    val_sample_size: int = 8,
    test_sample_size: int = 50,
    max_chars: int = 240,
    max_model_len: int = 16384,
    output_dir: str = None,
    prompts_file: str = None,
    resume: bool = False,
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
    if not resume:
        # Fresh run: move any prior run aside so we never interleave with it.
        _archive_existing_run(output_dir)

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
    # Prefix caching disabled here only: vLLM v0.17.1 + TP=2 deadlocks between
    # consecutive student .generate() calls during val/test when KV state is
    # shared across batches. PHQ-9 optimizer is unaffected — leaves it on.
    student_engine, teacher_engine = _build_engines(model_name, tp, 0.90,
                                                    max_model_len=max_model_len,
                                                    enable_prefix_caching=False,
                                                    **vllm_kwargs)

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
            "Maintain a diverse set of concrete rules — preserve specific guidance (tone, format, content, interaction) even when rules sit alongside more general framing.",
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

    # Resume support: pick up where a previous job died (state.json saved after
    # every step). On resume we skip the trajectory header and the step-0
    # baseline; both are already on disk. The tweet val set is resampled by
    # `_evaluate_tweet_instruction` internally per call, so we don't save it.
    resume_state = _load_resume_state(output_dir) if resume else None

    if resume_state:
        start_step = int(resume_state["next_step"])
        best_val_score = float(resume_state["best_metric"])
        best_instruction = resume_state["best_instruction"]
        instruction.value = resume_state["current_instruction"]
        print(f"[resume] continuing from step {start_step} "
              f"(best val score so far: {best_val_score:.2f})")
    else:
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
        start_step = 0
        with open(os.path.join(output_dir, "best_instruction_tweet.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"# val score: {_s0:.2f}/10  (step 0)\n\n{best_instruction}")
        with open(os.path.join(output_dir, "best_full_prompt_tweet.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"=== SYSTEM PROMPT ===\n{best_instruction}\n\n"
                     f"=== FORMAT BLOCK (fixed) ===\n{format_block_tweet}")
        _save_resume_state(output_dir, seed=seed, next_step=0,
                           best_metric=best_val_score,
                           best_instruction=best_instruction,
                           current_instruction=instruction.value)

    # Training loop.
    for step in range(start_step, num_steps):
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

            predictions = []   # tg.Variables — kept so the backward graph stays
                                # connected to `instruction` (severed if we build
                                # `combined` from raw strings via tg.Variable).
            raw_tweets = []
            parsed_tweets = []
            for j in range(tweets_per_sample):
                neighbor_tweets = _sample_neighbor_tweets(all_tweets_flat, rng, tweet_block)
                user_msg = _build_user_message_tweet(context, prompts, persona, phq9, neighbor_tweets, agent_id=str(batch_agent_ids[idx]))
                user_msg = user_msg + "\n\n" + format_block_tweet
                question = tg.Variable(
                    user_msg,
                    role_description="User template: persona, PHQ-9, own history, neighbor context",
                    requires_grad=False,
                )
                pred = model(question)
                pred.value = Agent.strip_model_thinking(pred.value)
                # Only the first prediction stays in the backward graph; the rest
                # still contribute their text to `combined.value` (so the teacher
                # rates the full set) but don't trigger their own backward LLM
                # call. All `tweets_per_sample` student calls share the same
                # `instruction`, so the gradient through one is informative about
                # the shared instruction. Cuts backward calls from
                # `tweets_per_sample` → 1 per sample.
                if j > 0:
                    pred.requires_grad = False
                predictions.append(pred)
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

            # Combine via tg.sum so the backward graph stays connected:
            # loss -> combined -> predictions -> instruction. Building `combined`
            # from raw strings (tg.Variable("\n".join(raw_tweets), ...)) would
            # detach the graph and make loss.backward() a silent no-op.
            combined = tg.sum(predictions)
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

        # Persist resume state after every step (after validation/revert).
        _save_resume_state(output_dir, seed=seed, next_step=step + 1,
                           best_metric=best_val_score,
                           best_instruction=best_instruction,
                           current_instruction=instruction.value)

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


def rerun_test_tweets(
    instruction_file: str,
    persona_phq9_file: str,
    num_agents: int = 7,
    sample_seed: int | None = None,
    neighbor_pool_roots: list[str] | None = None,
    model_name: str = QWEN_27,
    seed: int = 42,
    tweets_per_sample: int = 3,
    max_chars: int = 240,
    max_model_len: int = 16384,
    **vllm_kwargs,
):
    """Score a saved tweet instruction by running the train-time eval pipeline.

    Mirrors the final test phase of :func:`call_optimizer_tweets`, but on a
    fresh persona-PHQ-9 sample (no historical context — agents post cold):

      1. Sample ``num_agents`` (persona, phq9) pairs from ``persona_phq9_file``
         using ``sample_seed`` — same RNG seed reproduces the same agents.
      2. Build a neighbour pool from the Qwen3.5-27B test_post tree (same
         pool :mod:`generate_posts_opt_h` uses) so neighbour context matches
         generation-time conditions. ``seed`` drives the neighbour-sampling RNG.
      3. Each agent gets ``tweets_per_sample`` fresh posts generated by the
         student with the loaded instruction.
      4. The teacher rates each agent's set; per-PHQ-9 stats + mean/std are
         written to ``<iter_dir>/eval_test.json``.

    ``sample_seed`` defaults to the ``N`` in the prompt-file's ``iter_<N>/``
    parent dir — so by default ``iter_7/prompt.txt`` is evaluated on the same
    seven agents ``iter_7/posts.csv`` was generated with.

    Four files are written next to the instruction, mirroring the CSV schema
    the PHQ-9 mode already uses (so plotting / aggregation helpers transfer):

      * ``test_raw_scores.csv`` — one row per agent: model, seed, sample_seed,
        agent_id, phq9, score (the per-agent teacher rating).
      * ``test_scores_phq9.csv`` — one row per PHQ-9 bucket: model, seed,
        sample_seed, phq9, avg_score, n_samples, n_empty.
      * ``test_posts.csv`` — one row per generated tweet: agent_id, persona,
        phq9, agent_score, tweet_idx, tweet.
      * ``eval_meta.txt`` — plain-text reproducibility log (file paths, seeds,
        pool sizes, sampled row indices, final mean/std).
    """
    import gc
    import re as _re

    instr_path = os.path.abspath(instruction_file)
    if not os.path.isfile(instr_path):
        raise FileNotFoundError(f"instruction file not found: {instr_path}")
    iter_dir = os.path.dirname(instr_path)

    if sample_seed is None:
        parent = os.path.basename(iter_dir)
        m = _re.match(r"^iter_(\d+)$", parent)
        if not m:
            raise ValueError(
                f"--sample-seed not given and prompt-file's parent dir "
                f"{parent!r} doesn't match 'iter_<N>'; pass --sample-seed."
            )
        sample_seed = int(m.group(1))
        print(f"[tweet-rerun] sample_seed defaulted to {sample_seed} (from {parent}/)")

    if not neighbor_pool_roots:
        neighbor_pool_roots = [
            "data/test_post/Qwen_Qwen3.5-27B/temp_0.8_top_p_0.6_cp_10_inter",
            "data/test_post/Qwen_Qwen3.5-27B/temp_0.8_top_p_0.6_cp_10_no_inter",
        ]

    # --- instruction ---
    with open(instr_path) as f:
        text = f.read()
    lines = text.splitlines()
    while lines and lines[0].lstrip().startswith("# "):
        lines.pop(0)
    instruction = "\n".join(lines).strip()
    print(f"[tweet-rerun] instruction: {instr_path} "
          f"({len(instruction.split())} words)")

    # --- sample (persona, phq9) pairs with sample_seed (reproducible) ---
    df_pp = pd.read_csv(persona_phq9_file)
    if len(df_pp) < num_agents:
        raise ValueError(
            f"--persona-phq9-file has {len(df_pp)} rows but --num-agents={num_agents}"
        )
    sub_rng = np.random.default_rng(sample_seed)
    sample_idx = sorted(sub_rng.choice(len(df_pp), size=num_agents, replace=False))
    personas = df_pp["persona"].iloc[sample_idx].tolist()
    answers = df_pp["phq9"].iloc[sample_idx].astype(int).tolist()
    agent_ids = [str(i) for i in sample_idx]
    blocks = [[] for _ in range(num_agents)]   # cold-start: no historical posts
    print(f"[tweet-rerun] sampled {num_agents} agents from {persona_phq9_file} "
          f"(sample_seed={sample_seed})")

    # --- prompts JSON + fixed format block (CONSTRAINTS half of system_forced) ---
    with open(FC.PROMPTS_FILE) as f:
        prompts = json.load(f)
    raw = prompts["tweet_gen"]["system_forced"].replace("{max_chars}", str(max_chars))
    rules_marker = "### RULES ###"
    constraints_marker = "### CONSTRAINTS ###"
    if rules_marker in raw and constraints_marker in raw:
        fixed_intro, rest = raw.split(rules_marker, 1)
        _, fixed_tail = rest.split(constraints_marker, 1)
        format_block = fixed_intro.rstrip() + "\n\n" + constraints_marker + fixed_tail
    elif constraints_marker in raw:
        _, fixed_tail = raw.split(constraints_marker, 1)
        format_block = constraints_marker + fixed_tail
    else:
        format_block = ""

    # --- neighbour pool (Qwen3.5-27B test_post tree; seeded sampling below) ---
    all_tweets_flat: list[tuple[str, str]] = []
    for root in neighbor_pool_roots:
        for nb_path in sorted(glob.glob(os.path.join(root, "seed_*", "tweets_with_phq9.csv"))):
            nb_blocks, _, _, nb_agent_ids = parse_tweets_with_phq9_csv(nb_path)
            for block, aid in zip(nb_blocks, nb_agent_ids):
                for t in block:
                    if t and t not in ("NO_POST", "NO_TWEET"):
                        all_tweets_flat.append((t, aid))
    print(f"[tweet-rerun] neighbour pool: {len(all_tweets_flat)} posts from "
          f"{len(neighbor_pool_roots)} root(s)")

    # --- engines + eval ---
    rng = np.random.default_rng(seed)
    llm_seed = vllm_kwargs.get("seed")
    tp = vllm_kwargs.pop("tensor_parallel_size", None) or len(
        (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")
    )
    student_engine, teacher_engine = _build_engines(
        model_name, tp, 0.90,
        max_model_len=max_model_len,
        enable_prefix_caching=False,
        **vllm_kwargs,
    )

    out_parsed: list[list[str]] = []
    out_scores: list[float | None] = []
    mean_score, std_score, per_phq9 = _evaluate_tweet_instruction(
        student_engine, teacher_engine, instruction, prompts,
        blocks, answers, personas,
        all_tweets_flat, rng,
        format_block=format_block,
        tweets_per_sample=tweets_per_sample,
        agent_ids=agent_ids,
        out_parsed=out_parsed,
        out_scores=out_scores,
        verbose=True,
    )

    print(f"\n[tweet-rerun] Test quality score: {mean_score:.2f}/10 ± {std_score:.2f}  "
          f"(n={num_agents})")
    print("[tweet-rerun] Per-PHQ-9 scores:")
    for phq9_val, stats in sorted(per_phq9.items()):
        print(f"  PHQ-9={phq9_val:2d}  avg={stats['avg_score']:.2f}  "
              f"n={stats['n_samples']}  empty={stats['n_empty']}")

    model_short = model_name.split("/")[-1]

    # Per-agent raw scores — same schema family as PHQ-9 mode's test_raw_scores.csv.
    raw_csv = os.path.join(iter_dir, "test_raw_scores.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "model", "seed", "sample_seed", "agent_id", "phq9", "score",
        ])
        writer.writeheader()
        for i in range(num_agents):
            s = out_scores[i] if i < len(out_scores) else None
            writer.writerow({
                "model": model_short, "seed": seed, "sample_seed": sample_seed,
                "agent_id": agent_ids[i], "phq9": int(answers[i]),
                "score": "" if s is None else round(s, 4),
            })
    print(f"[tweet-rerun] wrote {raw_csv}")

    # Per-PHQ-9 aggregate — same schema family as PHQ-9 mode's test_scores_phq9.csv.
    per_phq9_csv = os.path.join(iter_dir, "test_scores_phq9.csv")
    with open(per_phq9_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "model", "seed", "sample_seed", "phq9",
            "avg_score", "n_samples", "n_empty",
        ])
        writer.writeheader()
        for phq9_val, stats in sorted(per_phq9.items()):
            writer.writerow({
                "model": model_short, "seed": seed, "sample_seed": sample_seed,
                "phq9": int(phq9_val),
                "avg_score": round(stats["avg_score"], 4),
                "n_samples": stats["n_samples"],
                "n_empty": stats["n_empty"],
            })
    print(f"[tweet-rerun] wrote {per_phq9_csv}")

    # Generated tweets — one row per post, with the per-agent teacher score.
    posts_csv = os.path.join(iter_dir, "test_posts.csv")
    with open(posts_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["agent_id", "persona", "phq9", "agent_score",
                         "tweet_idx", "tweet"])
        for i, tweets in enumerate(out_parsed):
            score = out_scores[i] if i < len(out_scores) else None
            score_str = "" if score is None else f"{score:.2f}"
            for j, tw in enumerate(tweets):
                clean = (tw or "").replace("\n", " ").replace("\r", " ").strip()
                writer.writerow([agent_ids[i], personas[i], answers[i],
                                 score_str, j, clean])
    print(f"[tweet-rerun] wrote {posts_csv}")

    # Reproducibility log — paths, seeds, pool sizes, sampled indices, summary.
    meta_path = os.path.join(iter_dir, "eval_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"timestamp:           {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"model:               {model_name}\n"
            f"seed:                {seed}\n"
            f"sample_seed:         {sample_seed}\n"
            f"llm_seed:            {llm_seed if llm_seed is not None else '<vllm default>'}\n"
            f"num_agents:          {num_agents}\n"
            f"tweets_per_sample:   {tweets_per_sample}\n"
            f"instruction_file:    {os.path.relpath(instr_path)}\n"
            f"persona_phq9_file:   {os.path.relpath(persona_phq9_file)}\n"
            f"neighbor_pool_size:  {len(all_tweets_flat)}\n"
            f"neighbor_pool_roots:\n"
        )
        for r in neighbor_pool_roots:
            fh.write(f"  - {r}\n")
        fh.write(
            f"sampled_row_idx:     {[int(i) for i in sample_idx]}\n"
            f"\n"
            f"mean_score:          {mean_score:.4f}\n"
            f"std_score:           {std_score:.4f}\n"
        )
    print(f"[tweet-rerun] wrote {meta_path}")

    try:
        del student_engine.base.client
    except Exception:
        pass
    del student_engine, teacher_engine
    gc.collect()
    torch.cuda.empty_cache()

    return mean_score, std_score, per_phq9


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

def split_embeddings_and_labels(rng, embeddings, labels, agent_ids=None,
                                train_frac=0.8, val_frac=0.1):
    """Split embeddings + labels into train/val/test tensors (remainder is test).

    Group-aware: when `agent_ids` is provided, the partition is computed over the
    set of unique agents and every block from a given agent lands in the same
    fold. This prevents the leakage where the same agent's writing-style fingerprint
    appears in both train and val (which masked val MAE as ~= train MAE).

    Args:
        rng: numpy Generator for the permutation.
        embeddings: per-block centroid tensor.
        labels: ground-truth PHQ-9 scores (converted to float tensor if needed).
        agent_ids: optional list of group keys (one per block). When None, falls
            back to the legacy block-level split.
        train_frac: fraction for training.
        val_frac: fraction for validation (test gets `1 - train_frac - val_frac`).

    Returns:
        (train_embs, val_embs, test_embs, train_labels, val_labels, test_labels).
    """
    n = len(embeddings)

    if agent_ids is None:
        perm = rng.permutation(n)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]
    else:
        if len(agent_ids) != n:
            raise ValueError(f"agent_ids length ({len(agent_ids)}) != embeddings length ({n})")
        unique_agents = sorted(set(agent_ids))
        agent_perm = rng.permutation(len(unique_agents))
        n_agents = len(unique_agents)
        n_train_a = int(n_agents * train_frac)
        n_val_a = int(n_agents * val_frac)
        train_agents = {unique_agents[i] for i in agent_perm[:n_train_a]}
        val_agents   = {unique_agents[i] for i in agent_perm[n_train_a : n_train_a + n_val_a]}
        test_agents  = {unique_agents[i] for i in agent_perm[n_train_a + n_val_a :]}

        train_idx = np.array([i for i, a in enumerate(agent_ids) if a in train_agents], dtype=np.int64)
        val_idx   = np.array([i for i, a in enumerate(agent_ids) if a in val_agents],   dtype=np.int64)
        test_idx  = np.array([i for i, a in enumerate(agent_ids) if a in test_agents],  dtype=np.int64)
        print(f"[split] {n_agents} unique agents → "
              f"train={len(train_agents)}/{len(train_idx)} blocks, "
              f"val={len(val_agents)}/{len(val_idx)} blocks, "
              f"test={len(test_agents)}/{len(test_idx)} blocks")

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

def _kfold_cv_pass(pool_embs, pool_labels, mental_bert: bool, device,
                   k: int, epochs: int, batch_size: int, weight_decay: float,
                   seed: int, save_dir: str, model_short: str,
                   pool_agent_ids: list[str] | None = None):
    """Run k-fold CV on `pool` and write cv_results.csv + cv_results.png.

    Group-aware: when `pool_agent_ids` is provided, folds are over agents (every
    block of a given agent stays in the same fold). Otherwise falls back to
    block-level folds (legacy — leaks agents across folds).

    Fold assignment is deterministic from `seed`. Torch is re-seeded to `seed`
    once per fold so weight init is identical across folds — fold variance
    therefore reflects data-partition effects only, not init noise.

    Returns:
        (mean_val_mae, std_val_mae, mean_best_epoch).
    """
    n = len(pool_embs)
    if n < k:
        raise ValueError(f"Cannot run {k}-fold CV with only {n} pool samples")
    rng = np.random.default_rng(seed)

    if pool_agent_ids is not None:
        unique_agents = sorted(set(pool_agent_ids))
        if len(unique_agents) < k:
            raise ValueError(f"Cannot run {k}-fold CV with only {len(unique_agents)} unique agents")
        agent_perm = rng.permutation(len(unique_agents))
        fold_size_a = len(unique_agents) // k
        agent_to_block_idx: dict[str, list[int]] = defaultdict(list)
        for i, a in enumerate(pool_agent_ids):
            agent_to_block_idx[a].append(i)
        print(f"\n--- {k}-fold group-CV on {len(unique_agents)} agents "
              f"({n} blocks, epochs/fold={epochs}) ---")
    else:
        perm = rng.permutation(n)
        fold_size = n // k
        print(f"\n--- {k}-fold CV on {n}-sample pool (epochs/fold={epochs}) ---")

    cv_records = []
    for fold_idx in range(k):
        if pool_agent_ids is not None:
            val_start = fold_idx * fold_size_a
            val_end = (fold_idx + 1) * fold_size_a if fold_idx < k - 1 else len(unique_agents)
            val_agents = {unique_agents[agent_perm[i]] for i in range(val_start, val_end)}
            val_indices = np.array(
                [i for a in val_agents for i in agent_to_block_idx[a]], dtype=np.int64
            )
            train_indices = np.array(
                [i for a, idxs in agent_to_block_idx.items() if a not in val_agents for i in idxs],
                dtype=np.int64,
            )
        else:
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size if fold_idx < k - 1 else n
            val_indices = perm[val_start:val_end]
            train_indices = np.concatenate([perm[:val_start], perm[val_end:]])

        fold_train_embs = pool_embs[train_indices].to(device)
        fold_train_labels = pool_labels[train_indices].to(device)
        fold_val_embs = pool_embs[val_indices].to(device)
        fold_val_labels = pool_labels[val_indices].to(device)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        fold_model = neural_net_BERT(mentalbert=mental_bert).to(device)

        fold_model, fold_best_val_mae, fold_history = train_bert(
            fold_model, fold_train_embs, fold_train_labels,
            fold_val_embs, fold_val_labels, device,
            epochs=epochs, batch_size=batch_size, weight_decay=weight_decay,
        )
        best_epoch = min(fold_history, key=lambda h: h["val_mae"])["epoch"]

        print(f"  Fold {fold_idx+1}/{k}: n_train={len(train_indices)} n_val={len(val_indices)}  "
              f"best val MAE={fold_best_val_mae:.3f} @ epoch {best_epoch}")

        cv_records.append({
            "model": model_short,
            "seed": seed,
            "fold": fold_idx + 1,
            "n_train": int(len(train_indices)),
            "n_val": int(len(val_indices)),
            "best_val_mae": float(fold_best_val_mae),
            "best_epoch": int(best_epoch),
        })

    cv_csv = os.path.join(save_dir, "cv_results.csv")
    fieldnames = ["model", "seed", "fold", "n_train", "n_val", "best_val_mae", "best_epoch"]
    with open(cv_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in cv_records:
            writer.writerow({key: r[key] for key in fieldnames})

    val_maes = np.array([r["best_val_mae"] for r in cv_records])
    best_epochs = np.array([r["best_epoch"] for r in cv_records])
    mean_val_mae = float(val_maes.mean())
    std_val_mae = float(val_maes.std())
    mean_best_epoch = int(round(float(best_epochs.mean())))

    print(f"\n[CV summary] {k}-fold val MAE: {mean_val_mae:.3f} ± {std_val_mae:.3f}  "
          f"(avg best epoch: {mean_best_epoch})")
    print(f"CV results → {cv_csv}")

    plot_cv_results(
        cv_records, mean_val_mae, std_val_mae, save_dir,
        title=f"{k}-fold CV — {model_short} seed={seed}",
    )

    return mean_val_mae, std_val_mae, mean_best_epoch


def run_bert_cv_diagnostic(embeddings_path, base_model_name, device, mental_bert: bool = False,
                           seed: int = 42, cv_folds: int = 5,
                           batch_size: int = 8, weight_decay: float = 1e-4,
                           epochs: int = 30):
    """One-shot partition-variance diagnostic: 80/20 pool/test split + k-fold CV on the pool.

    Test set is held out (not used) — this entry point only characterizes how
    much the validation MAE varies across data partitions. Use the regular
    `--mode bert` training afterwards for the actual production model(s).

    Outputs land in `data/test_post/bert_regression/cv_diagnostic_seed{seed}/`:
        ├── cv_results.csv
        └── cv_results.png
    """
    with open(embeddings_path, "rb") as f:
        data = torch.load(f)
    if "embeddings" not in data or "labels" not in data:
        raise ValueError("CV diagnostic requires the {embeddings, labels} cache format")

    all_embs = data["embeddings"]
    all_labels = data["labels"]
    all_agent_ids = data.get("agent_ids")
    if all_agent_ids is None:
        raise RuntimeError(
            f"Embeddings cache at {embeddings_path} predates the agent-level split. "
            "Re-run with --create-new-embeddings to rebuild the cache with agent_ids."
        )
    if not torch.is_tensor(all_labels):
        all_labels = torch.tensor(all_labels, dtype=torch.float32)

    rng = np.random.default_rng(seed)
    # Hold out 20% of AGENTS for test (unused here); pool is the remaining 80%.
    unique_agents = sorted(set(all_agent_ids))
    agent_perm = rng.permutation(len(unique_agents))
    n_test_a = max(1, int(len(unique_agents) * 0.20))
    test_agents = {unique_agents[i] for i in agent_perm[:n_test_a]}
    pool_indices = np.array(
        [i for i, a in enumerate(all_agent_ids) if a not in test_agents], dtype=np.int64
    )
    pool_embs = all_embs[pool_indices]
    pool_labels = all_labels[pool_indices]
    pool_agent_ids = [all_agent_ids[i] for i in pool_indices]

    model_short = base_model_name.split("/")[-1]
    save_dir = os.path.join("data", "test_post", "bert_regression",
                            f"cv_diagnostic_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"CV diagnostic: pool={len(pool_indices)} blocks from "
          f"{len(unique_agents) - n_test_a} agents (test={n_test_a} agents held out, unused)")
    return _kfold_cv_pass(
        pool_embs, pool_labels, mental_bert, device,
        k=cv_folds, epochs=epochs, batch_size=batch_size, weight_decay=weight_decay,
        seed=seed, save_dir=save_dir, model_short=model_short,
        pool_agent_ids=pool_agent_ids,
    )


def train_BERT_model(embeddings_path, base_model_name, device, mental_bert: bool = False,
                     seed: int = 42, batch_size: int = 8, weight_decay: float = 1e-4,
                     learning_rate: float = 1e-4, epochs: int = 30,
                     init_from: str | None = None, out_dir: str | None = None):
    """Load cached embeddings, do an 80/10/10 split, train the regressor, write metrics + plots.

    Set `init_from` to a saved `regressor.pt` to CONTINUE training that model
    (fine-tuning) instead of Kaiming-initialising a fresh one — pair with a low
    `learning_rate` and an `out_dir` so the source regressor is left untouched.

    The output layout mirrors `call_optimizer_phq9`:
        {out_dir or data/test_post/bert_regression}/{model_short}_seed{seed}/
            ├── regressor.pt
            ├── training_trajectory.csv   (model, seed, step, split, mean_score, std_score, n_samples)
            ├── test_scores_phq9.csv      (model, seed, phq9, avg_mae, avg_bias, std_bias, n_samples)
            ├── test_raw_scores.csv       (model, seed, true_phq9, pred_phq9)
            ├── performance.json
            ├── trajectory.png
            └── test_scores_by_phq9.png

    Seed semantics:
        A single `seed` drives both the numpy split RNG (so each seed sees a
        different 80/10/10 partition) and the torch RNG used for weight init +
        dropout. Variance across `--seeds` therefore reflects combined
        partition + init noise — the standard "how stable is this experiment
        if I rerun it" reading. Run `--mode bert-cv` once for a partition-only
        variance diagnostic (k-fold CV).

    Args:
        embeddings_path: path to a `.pt` containing {embeddings, labels}.
        base_model_name: model id used to construct the save directory.
        device: torch device.
        mental_bert: True if embeddings came from MentalBERT (controls regressor input size).
        seed: RNG seed driving both data partition and model init.
        batch_size: minibatch size for the inner `train_bert` loop.
        weight_decay: AdamW weight-decay coefficient.
        learning_rate: initial AdamW LR (use a small value, e.g. 2e-5, when fine-tuning).
        epochs: max training epochs.
        init_from: optional path to a saved regressor.pt to continue training from
            (fine-tuning); when None a fresh Kaiming-initialised model is trained.
        out_dir: optional base dir for the per-seed output folder; defaults to
            data/test_post/bert_regression. Set this when fine-tuning so the
            source regressor isn't overwritten.

    Returns:
        (best_model, best_val_mae, test_mae).
    """
    with open(embeddings_path, "rb") as f:
        data = torch.load(f)

    rng = np.random.default_rng(seed)

    if "embeddings" in data and "labels" in data:
        all_embs = data["embeddings"]
        all_labels = data["labels"]
        agent_ids = data.get("agent_ids")
        if agent_ids is None:
            raise RuntimeError(
                f"Embeddings cache at {embeddings_path} predates the agent-level split. "
                "Re-run with --create-new-embeddings to rebuild the cache with agent_ids; "
                "block-level splits leak the same agent across train/val/test."
            )
        train_embs, val_embs, test_embs, train_labels, val_labels, test_labels = split_embeddings_and_labels(
            rng, all_embs, all_labels, agent_ids=agent_ids, train_frac=0.8, val_frac=0.1
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

    n_train, n_val = len(train_embs), len(val_embs)
    print(f"Training blocks: {n_train}, val: {n_val}, test: {len(test_embs)}")

    model_short = base_model_name.split("/")[-1]
    base_out = out_dir or os.path.join("data", "test_post", "bert_regression")
    save_dir = os.path.join(base_out, f"{model_short}_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    trajectory_path = os.path.join(save_dir, "training_trajectory.csv")
    traj_fields = ["model", "seed", "step", "split", "mean_score", "std_score", "n_samples"]
    with open(trajectory_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=traj_fields).writeheader()

    def _write_row(step: int, split: str, mean_score: float, std_score: float, n_samples: int):
        with open(trajectory_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=traj_fields).writerow({
                "model": model_short, "seed": seed, "step": step, "split": split,
                "mean_score": round(float(mean_score), 4),
                "std_score": round(float(std_score), 4),
                "n_samples": int(n_samples),
            })

    # Reseed torch right before building the regressor so weight init (Kaiming)
    # and dropout are reproducible per seed and vary across seeds.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if init_from:
        print(f"[fine-tune] continuing from {init_from}")
        nn_model = torch.load(init_from, map_location=device, weights_only=False).to(device)
    else:
        nn_model = neural_net_BERT(mentalbert=mental_bert).to(device)

    # Step-0 baseline: untrained network on train + val (matches PHQ-9 baseline row at step=0).
    s0_train_mae, s0_train_std = _bert_eval_summary(nn_model, train_embs, train_labels, device)
    s0_val_mae,   s0_val_std   = _bert_eval_summary(nn_model, val_embs,   val_labels,   device)
    _write_row(0, "train", s0_train_mae, s0_train_std, n_train)
    _write_row(0, "val",   s0_val_mae,   s0_val_std,   n_val)
    print(f"[Step 0] Baseline  train MAE: {s0_train_mae:.2f} ± {s0_train_std:.2f}  "
          f"val MAE: {s0_val_mae:.2f} ± {s0_val_std:.2f}")

    nn_model, best_val_mae, history = train_bert(
        nn_model, train_embs, train_labels, val_embs, val_labels, device,
        batch_size=batch_size, weight_decay=weight_decay,
        learning_rate=learning_rate, epochs=epochs,
    )
    for entry in history:
        _write_row(entry["epoch"], "train", entry["train_mae"], entry["train_std"], n_train)
        _write_row(entry["epoch"], "val",   entry["val_mae"],   entry["val_std"],   n_val)

    print("Testing best model....")
    test_mae, test_std, per_phq9, raw = _bert_eval_summary(
        nn_model, test_embs, test_labels, device, want_per_phq9=True,
    )
    n_test = len(test_embs)
    last_step = history[-1]["epoch"] if history else 0
    _write_row(last_step, "test", test_mae, test_std, n_test)
    print(f"Test MAE (n={n_test}): {test_mae:.2f} ± {test_std:.2f}")

    save_path = os.path.join(save_dir, "regressor.pt")
    torch.save(nn_model, save_path)
    print(f"Regressor saved to {save_path}")

    # Raw per-sample (true, pred) — kept for downstream analysis (calibration plots,
    # confusion matrices, error distributions, etc.) without re-running the regressor.
    raw_csv = os.path.join(save_dir, "test_raw_scores.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "seed", "true_phq9", "pred_phq9"])
        writer.writeheader()
        for t, p in zip(raw["true"], raw["pred"]):
            writer.writerow({"model": model_short, "seed": seed,
                             "true_phq9": round(float(t), 4),
                             "pred_phq9": round(float(p), 4)})
    print(f"Raw test scores → {raw_csv}")

    per_phq9_csv = os.path.join(save_dir, "test_scores_phq9.csv")
    with open(per_phq9_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "seed", "phq9", "avg_mae", "avg_bias", "std_bias", "n_samples"],
        )
        writer.writeheader()
        for phq9_val, stats in sorted(per_phq9.items()):
            writer.writerow({
                "model": model_short, "seed": seed, "phq9": phq9_val,
                "avg_mae": round(stats["avg_mae"], 4),
                "avg_bias": round(stats["avg_bias"], 4),
                "std_bias": round(stats["std_bias"], 4),
                "n_samples": stats["n_samples"],
            })
    print(f"Per-PHQ-9 MAE+bias → {per_phq9_csv}")

    metrics_path = os.path.join(save_dir, "performance.json")
    with open(metrics_path, "w") as fh:
        json.dump({
            "model": model_short,
            "seed": seed,
            "mental_bert": bool(mental_bert),
            "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "best_val_mae": float(best_val_mae),
            "test_mae": float(test_mae),
            "test_std": float(test_std),
            "epochs": history,
        }, fh, indent=2)

    plot_optimizer_trajectory(
        trajectory_path, save_dir,
        title=f"BERT regressor — {model_short} seed={seed}",
        mode="phq9",
    )
    plot_test_mae_and_bias_by_phq9(
        per_phq9, save_dir,
        title=f"Test MAE & bias by PHQ-9 — {model_short} seed={seed} (BERT)",
    )

    return nn_model, best_val_mae, test_mae


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
                weight_decay: float = 1e-4):
    """Train the MLP regressor (Huber gradient, MAE tracking, AdamW).

    Per-epoch MAE/std are recorded for both train and val (when provided) on the
    *full* split so the resulting history is directly comparable to the PHQ-9
    optimizer's trajectory CSV (which also reports MAE).

    Pass `val_data=None` (and `val_labels=None`) for "full training" runs that
    use the entire pool with no holdout val. In that mode there's no best-model
    selection and the final-epoch model is returned; val_mae/val_std rows in
    `history` are NaN.

    Args:
        model: regressor (already on device).
        train_data: train embeddings.
        train_labels: train PHQ-9 scores.
        val_data: val embeddings, or None to disable val tracking.
        val_labels: val PHQ-9 scores, or None.
        device: torch device.
        epochs: max training epochs.
        batch_size: minibatch size.
        learning_rate: initial AdamW learning rate.
        weight_decay: AdamW weight-decay coefficient (L2 regularization).

    Returns:
        (best_model, best_val_mae, history) where history is a list of dicts
        with keys {"epoch", "train_mae", "train_std", "val_mae", "val_std"}.
        When val is provided, best model is selected on val MAE; otherwise
        best_val_mae is NaN and best_model is the final-epoch model.
    """
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    has_val = val_data is not None
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=2) if has_val else None

    criterion = nn.HuberLoss(delta=6.0)
    train_data = torch.as_tensor(train_data, dtype=torch.float32).to(device)
    train_labels = torch.as_tensor(train_labels, dtype=torch.float32).to(device)

    best_val_mae = float('inf')
    best_model_state = copy.deepcopy(model)
    history: list[dict] = []
    n_train = len(train_data)
    for epoch in range(epochs):
        model.train()
        # Reshuffle each epoch so batch composition varies — reproducible via the
        # torch seed set before train_bert is called.
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            inputs = train_data[idx]
            labels = train_labels[idx]

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_mae, train_std = _bert_eval_summary(model, train_data, train_labels, device)
        if has_val:
            val_mae, val_std = _bert_eval_summary(model, val_data, val_labels, device)
            scheduler.step(val_mae)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = copy.deepcopy(model)
            print(f"Epoch {epoch+1}/{epochs}, Train MAE: {train_mae:.3f}  Val MAE: {val_mae:.3f}")
        else:
            val_mae, val_std = float('nan'), float('nan')
            print(f"Epoch {epoch+1}/{epochs}, Train MAE: {train_mae:.3f}  (no val — full-training mode)")

        history.append({
            "epoch": epoch + 1,
            "train_mae": train_mae, "train_std": train_std,
            "val_mae": val_mae, "val_std": val_std,
        })

    if not has_val:
        best_model_state = copy.deepcopy(model)
        best_val_mae = float('nan')

    return best_model_state, best_val_mae, history

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


def _bert_eval_summary(model, data, labels, device, want_per_phq9: bool = False):
    """MAE/std (and optional per-PHQ-9 stats + raw scores) of a BERT regressor on `data`.

    Mirrors the PHQ-9 optimizer's `_evaluate_instruction` return shape and extends
    it with signed-bias tracking so we can plot persistent over-/under-estimation
    patterns per PHQ-9 category.

    Args:
        model: trained regressor (will be set to eval mode).
        data: input embeddings.
        labels: ground-truth PHQ-9 scores.
        device: torch device.
        want_per_phq9: if True also return per-PHQ-9 stats and raw (true, pred) arrays.

    Returns:
        (mae, std) or (mae, std, per_phq9, raw) where:
            per_phq9 = {true_score: {"avg_mae", "avg_bias", "std_bias", "n_samples"}}
                       — avg_bias is mean(pred − true); positive means over-estimation.
            raw      = {"true": [...], "pred": [...]} per-sample arrays.
    """
    model.eval()
    data_t = torch.as_tensor(data, dtype=torch.float32).to(device)
    labels_t = torch.as_tensor(labels, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(data_t).squeeze(-1)
    abs_err = (preds - labels_t).abs().cpu().numpy()
    signed_err = (preds - labels_t).cpu().numpy()  # pred − true; sign carries the bias.
    if abs_err.size == 0:
        mae, std = 0.0, 0.0
    else:
        mae = float(abs_err.mean())
        std = float(abs_err.std())
    if not want_per_phq9:
        return mae, std

    per_phq9_abs = defaultdict(list)
    per_phq9_signed = defaultdict(list)
    labels_np = labels_t.cpu().numpy()
    preds_np = preds.cpu().numpy()
    for lab, a_err, s_err in zip(labels_np, abs_err, signed_err):
        k = int(round(float(lab)))
        per_phq9_abs[k].append(float(a_err))
        per_phq9_signed[k].append(float(s_err))
    per_phq9 = {
        k: {
            "avg_mae": float(np.mean(per_phq9_abs[k])),
            "avg_bias": float(np.mean(per_phq9_signed[k])),
            "std_bias": float(np.std(per_phq9_signed[k])),
            "n_samples": len(per_phq9_abs[k]),
        }
        for k in per_phq9_abs
    }
    raw = {"true": labels_np.tolist(), "pred": preds_np.tolist()}
    return mae, std, per_phq9, raw


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
    tweet_blocks, true_answers, group_ids = [], [], []
    for fp in paths:
        if fp.endswith(".csv"):
            blocks, answers, _, aids = parse_tweets_with_phq9_csv(fp)
            # Disambiguate identical agent_id strings across different simulation files.
            group_ids.extend([f"{fp}::{aid}" for aid in aids])
        else:
            blocks, answers = parse_tweets_with_phq9(fp)
            # No agent info in .txt — each block is its own group (degenerate = old block-level split).
            group_ids.extend([f"{fp}::block_{i}" for i in range(len(blocks))])
        tweet_blocks.extend(blocks)
        true_answers.extend(answers)
    print(f"Loaded {len(tweet_blocks)} blocks from {len(paths)} file(s) "
          f"({len(set(group_ids))} unique agent groups)")

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
    torch.save({"embeddings": all_embs, "labels": all_labels, "agent_ids": group_ids}, torch_path)
    print(f"Saved embeddings + labels + agent_ids to {torch_path} (n={len(all_embs)}); split will be done at train time.")
    

def eval_bert_regressors_on_csv(
    csv_path: str,
    seeds: list[int],
    model_name: str = QWEN_27,
    out_dir: str | None = None,
    regressor_dir: str = "data/test_post/bert_regression",
    mentalbert: bool = True,
    device: str | None = None,
) -> list[dict]:
    """Evaluate one or more trained BERT regressors on a single tweets_with_phq9 CSV.

    For each seed, loads
    `{regressor_dir}/{model_short}_seed{seed}/regressor.pt` and scores it on
    `csv_path` via `eval_bert_on_csv.evaluate` (per-seed predictions + a
    per-PHQ-9 summary land in `out_dir/seed{seed}.csv`), then writes an
    across-seed `aggregate.csv` (per-seed MAE/bias plus MEAN/STD rows).

    Args:
        csv_path: tweets_with_phq9 CSV to score (e.g. an SA_prompt dataset).
        seeds: regressor seeds to evaluate; missing ones are skipped with a warning.
        model_name: HuggingFace id; its short tail names the regressor subdir.
        out_dir: output dir. Default: `<csv_dir>/bert_eval/`.
        regressor_dir: root holding the `{model_short}_seed{seed}/` regressors.
        mentalbert: must match how the regressor was trained (768-dim MentalBERT).
        device: torch device string; None auto-selects cuda/cpu.

    Returns:
        Per-seed list of {"seed", "mae", "bias", "n"} dicts.
    """
    from .eval_bert_on_csv import evaluate  # local import avoids an import cycle

    model_short = model_name.split("/")[-1]
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "bert_eval")
    os.makedirs(out_dir, exist_ok=True)

    results: list[dict] = []
    for seed in seeds:
        reg_path = os.path.join(regressor_dir, f"{model_short}_seed{seed}", "regressor.pt")
        if not os.path.isfile(reg_path):
            print(f"[bert-eval] WARN: no regressor for seed {seed} ({reg_path}) — skipping")
            continue
        print(f"\n{'='*60}\n[bert-eval] seed {seed}\n{'='*60}")
        out_csv = os.path.join(out_dir, f"seed{seed}.csv")
        res = evaluate(reg_path, csv_path, out_csv, mentalbert=mentalbert, device=device)
        results.append({"seed": seed, "mae": res["mae"], "bias": res["bias"], "n": res["n"]})

    if not results:
        raise RuntimeError(f"No regressors evaluated (looked under {regressor_dir} "
                           f"for {model_short} seeds {seeds}).")

    maes = [r["mae"] for r in results]
    biases = [r["bias"] for r in results]
    mean_mae, std_mae = float(np.mean(maes)), float(np.std(maes))
    mean_bias, std_bias = float(np.mean(biases)), float(np.std(biases))

    agg_path = os.path.join(out_dir, "aggregate.csv")
    with open(agg_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed", "n", "mae", "bias"])
        writer.writeheader()
        for r in results:
            writer.writerow({"seed": r["seed"], "n": r["n"],
                             "mae": round(r["mae"], 4), "bias": round(r["bias"], 4)})
        writer.writerow({"seed": "MEAN", "n": "", "mae": round(mean_mae, 4),
                         "bias": round(mean_bias, 4)})
        writer.writerow({"seed": "STD", "n": "", "mae": round(std_mae, 4),
                         "bias": round(std_bias, 4)})

    print(f"\n[bert-eval] across {len(results)} seed(s): "
          f"MAE={mean_mae:.3f} ± {std_mae:.3f}  bias={mean_bias:+.3f} ± {std_bias:.3f}")
    print(f"[bert-eval] aggregate → {agg_path}")
    return results


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
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <output_dir>/state.json if present "
                             "(skips step-0 baseline; trajectory CSV is appended). "
                             "If state.json is missing, falls back to a fresh run.")
    parser.add_argument("--model", type=str, default=QWEN_27,
                        help="HuggingFace model id (e.g. Qwen/Qwen3.5-27B)")
    parser.add_argument("--mode", type=str, default="tweets",
                        choices=["tweets", "phq9", "phq9-rerun-test",
                                 "tweets-rerun-test", "bert", "bert-cv", "bert-eval"],
                        help="Which entry point to run: tweets, phq9, phq9-rerun-test "
                             "(re-test saved best instruction without retraining), "
                             "tweets-rerun-test (score a saved tweet instruction by "
                             "sampling fresh agent-PHQ-9 pairs; see --instruction-file "
                             "+ --persona-phq9-file), bert, bert-cv "
                             "(one-shot partition-variance diagnostic for the BERT regressor), "
                             "or bert-eval (score trained regressor(s) on a single "
                             "tweets_with_phq9 CSV; see --posts-file + --seeds).")
    parser.add_argument("--instruction-filename", type=str, default="optimized_instruction.txt",
                        help="(--mode phq9-rerun-test only) which prompt file under the seed "
                             "directory to test. Default: optimized_instruction.txt — the prompt "
                             "the original test run used. For seed-26-style mixed states pass "
                             "optimized_instruction.txt explicitly rather than best_instruction.txt.")
    parser.add_argument("--instruction-file", type=str, default=None,
                        help="(--mode tweets-rerun-test only) full path to the instruction "
                             "txt to evaluate (e.g. data/prompt_optimization_h/<run>/iter_N/prompt.txt).")
    parser.add_argument("--posts-file", type=str, default=None,
                        help="(--mode phq9-rerun-test only) override: use this entire "
                             "tweets_with_phq9 CSV as the test set, skipping the "
                             "train/val/test split. Results land in a sibling "
                             "<seed_dir>/eval_on_<posts_stem>/ subdir; the original "
                             "in-distribution test files and trajectory.csv test row are "
                             "preserved untouched.")
    parser.add_argument("--persona-phq9-file", type=str, default=None,
                        help="(--mode tweets-rerun-test only) (persona, phq9) CSV — fresh "
                             "agent-PHQ-9 pairs are sampled from here with --sample-seed.")
    parser.add_argument("--num-agents", type=int, default=7,
                        help="(--mode tweets-rerun-test only) agents drawn from the persona-phq9 file.")
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="(--mode tweets-rerun-test only) RNG seed for agent sampling. "
                             "Defaults to N parsed from the prompt-file's iter_<N>/ parent dir, "
                             "so iter_7/prompt.txt evaluates on the same agents iter_7/posts.csv "
                             "was generated against.")
    parser.add_argument("--neighbor-pool-root", type=str, nargs="+", default=None,
                        help="(--mode tweets-rerun-test only) base dirs containing "
                             "seed_*/tweets_with_phq9.csv for neighbour sampling. Defaults "
                             "to both inter+no_inter under data/test_post/Qwen_Qwen3.5-27B/.")
    parser.add_argument("--tweets-per-sample", type=int, default=3,
                        help="(--mode tweets-rerun-test only) tweets generated per agent.")
    parser.add_argument("--llm-seed", type=int, default=None,
                        help="(--mode tweets-rerun-test only) seed passed to vLLM's LLM(...) "
                             "at engine construction. Different values = different student / "
                             "teacher sampling sequences. Leave unset for vLLM's default.")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="(--mode bert-cv only) k for k-fold CV. Default: 5. "
                             "Ignored in all other modes.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="(--mode bert / bert-cv only) AdamW weight-decay "
                             "coefficient for the regressor. Default: 1e-4.")
    parser.add_argument("--regressor-dir", type=str, default="data/test_post/bert_regression",
                        help="(--mode bert-eval only) root holding {model_short}_seed{seed}/"
                             "regressor.pt. Default: data/test_post/bert_regression.")
    parser.add_argument("--bert-eval-out-dir", type=str, default=None,
                        help="(--mode bert-eval only) output dir for per-seed + aggregate "
                             "CSVs. Default: <posts-file dir>/bert_eval/.")
    parser.add_argument("--init-from-dir", type=str, default=None,
                        help="(--mode bert only) base dir holding source "
                             "{model_short}_seed{seed}/regressor.pt to CONTINUE training "
                             "(fine-tune). Requires --posts-file (the CSV to adapt to). "
                             "Pair with a low --learning-rate and --bert-out-dir so the "
                             "source regressors are not overwritten.")
    parser.add_argument("--bert-out-dir", type=str, default=None,
                        help="(--mode bert only) base dir for fine-tuned regressor output. "
                             "Default: data/test_post/bert_regression (overwrites!). When "
                             "fine-tuning, set this to a fresh dir.")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="(--mode bert only) initial AdamW LR. Use ~2e-5 when "
                             "fine-tuning (--init-from-dir) to preserve prior knowledge.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="(--mode bert only) max training epochs.")
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
                resume=args.resume,
            )
    elif run_mode == "phq9-rerun-test":
        for seed in args.seeds:
            print(f"\n{'='*60}\nRe-running PHQ-9 test for seed={seed} "
                  f"(loading {args.instruction_filename})\n{'='*60}")
            rerun_test_phq9(
                seed=seed,
                file_paths=file_paths,
                model_name=model_name,
                instruction_filename=args.instruction_filename,
                posts_file=args.posts_file,
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
                resume=args.resume,
            )
    elif run_mode == "tweets-rerun-test":
        if not args.instruction_file:
            raise SystemExit("--mode tweets-rerun-test requires --instruction-file")
        if not args.persona_phq9_file:
            raise SystemExit("--mode tweets-rerun-test requires --persona-phq9-file")
        vllm_extra = {"seed": args.llm_seed} if args.llm_seed is not None else {}
        for seed in args.seeds:
            print(f"\n{'='*60}\nRe-running tweet eval for {args.instruction_file} "
                  f"(seed={seed})\n{'='*60}")
            rerun_test_tweets(
                instruction_file=args.instruction_file,
                persona_phq9_file=args.persona_phq9_file,
                num_agents=args.num_agents,
                sample_seed=args.sample_seed,
                neighbor_pool_roots=args.neighbor_pool_root,
                model_name=model_name,
                seed=seed,
                tweets_per_sample=args.tweets_per_sample,
                **vllm_extra,
            )
    elif run_mode == "bert-eval":
        if not args.posts_file:
            raise SystemExit("--mode bert-eval requires --posts-file (the tweets_with_phq9 CSV to score)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        eval_bert_regressors_on_csv(
            csv_path=args.posts_file,
            seeds=args.seeds,
            model_name=model_name,
            out_dir=args.bert_eval_out_dir,
            regressor_dir=args.regressor_dir,
            mentalbert=mental_bert,
            device=device,
        )
    else:
        # bert / bert-cv: shared device + embeddings-cache setup.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model_short = base_model_name.split("/")[-1]
        finetuning = run_mode == "bert" and args.init_from_dir is not None
        dir_name = "mentalbert_embeddings" if mental_bert else "sbert_embeddings"
        if finetuning:
            # Fine-tune: encode the single prompt-block CSV into its own cache so the
            # big inter/no_inter tree cache is never clobbered.
            if not args.posts_file:
                raise SystemExit("--init-from-dir (fine-tune) requires --posts-file (the CSV to adapt to)")
            file_paths = [args.posts_file]
            csv_stem = os.path.splitext(os.path.basename(args.posts_file))[0]
            emb_dir = os.path.join("data", "test", base_model_name, dir_name, f"finetune_{csv_stem}")
            embeddings_path = os.path.join(emb_dir, "embeddings_and_labels.pt")
        else:
            embeddings_path = os.path.join("data", "test", base_model_name, dir_name, "embeddings_and_labels.pt")
        need_rebuild = create_new_embeddings or not os.path.isfile(embeddings_path)
        if not need_rebuild:
            # Old caches lack `agent_ids` — rebuild so the agent-level split has the group keys it needs.
            _cached = torch.load(embeddings_path, map_location="cpu")
            if "agent_ids" not in _cached:
                print(f"[bert] cache at {embeddings_path} lacks agent_ids — forcing rebuild for agent-level split")
                need_rebuild = True
            del _cached
        if need_rebuild:
            print(f"[bert] (re)building embeddings cache at {embeddings_path}")
            save_embeddings_for_file(file_paths, base_model_name, device,
                                     mentalbert=mental_bert,
                                     out_dir=os.path.dirname(embeddings_path))

        if run_mode == "bert-cv":
            seed = args.seeds[0]
            print(f"\n{'='*60}\nRunning BERT CV diagnostic with seed={seed} "
                  f"cv_folds={args.cv_folds} batch_size={args.batch_size} "
                  f"weight_decay={args.weight_decay}\n{'='*60}")
            run_bert_cv_diagnostic(embeddings_path, base_model_name, device,
                                   mental_bert=mental_bert,
                                   seed=seed, cv_folds=args.cv_folds,
                                   batch_size=args.batch_size,
                                   weight_decay=args.weight_decay)
        else:
            base_out_ft = args.bert_out_dir or os.path.join("data", "test_post", "bert_regression")
            for seed in args.seeds:
                init_from = None
                if finetuning:
                    # Resumable fine-tune: skip a seed whose regressor.pt already exists.
                    # train_BERT_model writes regressor.pt only after training + test
                    # finish, so a crashed seed never has one and is always retrained
                    # (its stale training_trajectory.csv is reopened in "w" mode).
                    done = os.path.join(base_out_ft, f"{model_short}_seed{seed}", "regressor.pt")
                    if os.path.isfile(done):
                        print(f"[fine-tune] seed {seed} already trained ({done}) — skipping")
                        continue
                    init_from = os.path.join(args.init_from_dir,
                                             f"{model_short}_seed{seed}", "regressor.pt")
                    if not os.path.isfile(init_from):
                        print(f"[fine-tune] WARN: no source regressor for seed {seed} "
                              f"({init_from}) — skipping")
                        continue
                tag = "Fine-tuning" if finetuning else "Running"
                print(f"\n{'='*60}\n{tag} BERT regressor with seed={seed} "
                      f"batch_size={args.batch_size} weight_decay={args.weight_decay} "
                      f"lr={args.learning_rate} epochs={args.epochs}\n{'='*60}")
                train_BERT_model(embeddings_path, base_model_name, device,
                                 mental_bert=mental_bert,
                                 seed=seed,
                                 batch_size=args.batch_size,
                                 weight_decay=args.weight_decay,
                                 learning_rate=args.learning_rate,
                                 epochs=args.epochs,
                                 init_from=init_from,
                                 out_dir=args.bert_out_dir)