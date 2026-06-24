import csv
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from vllm import LLM, SamplingParams

from classes.agent import Agent
from utils.tools.format_config import FC


def _phq9_severity(score: int) -> str:
    """Map a PHQ-9 sumscore to the severity label prompt_optimizer uses."""
    if score >= 20:
        return "severe"
    if score >= 15:
        return "moderately severe"
    if score >= 10:
        return "moderate"
    if score >= 5:
        return "mild"
    return "minimal/none"


def load_instruction_file(path: str) -> str:
    """Load an optimizer-saved instruction file, stripping leading '# val score' headers.

    Only lines starting with `# ` (hash + space, i.e. metadata comments) are
    stripped — markdown-style headers like `### RULES ###` are preserved.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()
    while lines and lines[0].lstrip().startswith("# "):
        lines.pop(0)
    return "\n".join(lines).strip()


def derive_tweet_format_block(prompts: dict, max_chars: int = 240) -> str:
    """Mirror prompt_optimizer's split of tweet_gen.system_forced into instruction + fixed block."""
    raw = prompts["tweet_gen"]["system_forced"].replace("{max_chars}", str(max_chars))
    rules_marker = "### RULES ###"
    constraints_marker = "### CONSTRAINTS ###"
    if rules_marker in raw and constraints_marker in raw:
        fixed_intro, rest = raw.split(rules_marker, 1)
        _instruction, fixed_tail = rest.split(constraints_marker, 1)
        return fixed_intro.rstrip() + "\n\n" + constraints_marker + fixed_tail
    if constraints_marker in raw:
        _instruction, fixed_tail = raw.split(constraints_marker, 1)
        return constraints_marker + fixed_tail
    return ""


class TestLLMs:
    def __init__(self, seed: int = 42, num_agents: int = 5,
                 personas: list = None, agents=None, well_being: list = None,
                 interaction: bool = False,
                 tweet_instruction: str = None, tweet_format_block: str = None,
                 prompts: dict = None, thinking: bool = False,
                 phq9_assignments: list = None,
                 neighbor_pool: list = None, num_neighbors: int = 5,
                 neighbor_seed: int = None,
                 nondeterministic_sampling: bool = False,
                 gen_temp: float = 0.7, gen_top_p: float = 0.9):
        self.rng = np.random.default_rng(seed)
        # Honour an explicit phq9_assignments by keeping persona order aligned to
        # it (no permutation), so personas[i] still pairs with phq9_assignments[i]
        # after the assignment-driven well_being write below.
        if phq9_assignments is None:
            personas = self.rng.permutation(personas) if personas is not None else [None] * num_agents
            self.well_being = self.rng.permutation(well_being) if well_being is not None else [None] * num_agents
        else:
            personas = list(personas) if personas is not None else [None] * num_agents
            self.well_being = list(well_being) if well_being is not None else [None] * num_agents

        self.num_agents = num_agents
        self.interaction = interaction
        if agents is None:
            self.all_agents = [
                Agent(i, rng=np.random.default_rng(seed + i),
                      persona=personas[i], well_being=self.well_being[i])
                for i in range(int(num_agents))
            ]
        else:
            self.all_agents = agents

        # PHQ-9 sequence per agent. When phq9_assignments is given, each agent
        # gets a single fixed score (one block per persona); otherwise each agent
        # gets its own permutation of 0..27 as ground truth across the run.
        self.phq9_sequences = {}
        self.phq9_indices = {}
        if phq9_assignments is not None:
            if len(phq9_assignments) < len(self.all_agents):
                raise ValueError(
                    f"phq9_assignments has {len(phq9_assignments)} entries but "
                    f"{len(self.all_agents)} agents were created."
                )
            for agent, phq9 in zip(self.all_agents, phq9_assignments):
                self.phq9_sequences[agent.ID] = [int(phq9)]
                self.phq9_indices[agent.ID] = 0
                agent.update_well_being(int(phq9))
        else:
            scores = np.arange(28)
            for agent in self.all_agents:
                self.phq9_sequences[agent.ID] = self.rng.permutation(scores).tolist()
                self.phq9_indices[agent.ID] = 0
                agent.update_well_being(self.phq9_sequences[agent.ID][0])

        self.iterations = 0
        self.seed = seed

        # Optimizer-aligned tweet generation. When tweet_instruction is set, the
        # tweet system prompt + user message + sampling all match the student in
        # prompt_optimizer (so the generated dataset can be assessed downstream
        # under the same conditions the student was optimised for).
        self.tweet_instruction = tweet_instruction
        self.tweet_format_block = tweet_format_block
        self.prompts = prompts
        self.thinking = thinking
        # External neighbour pool (flat list of (agent_id, post) tuples) loaded
        # from a tweets_with_phq9 file — same source format Grok uses. Sampled
        # fresh on every inference (per round), capped at `num_neighbors`.
        # `neighbor_seed`, when set, drives a per-(agent, round) sub-RNG so the
        # same agent at the same round gets the same neighbours across runs,
        # independent of execution order or other RNG consumers. Used by
        # SA-prompt mode to compare prompts on identical neighbour contexts.
        self.neighbor_pool = list(neighbor_pool) if neighbor_pool else None
        self.num_neighbors = int(num_neighbors)
        self.neighbor_seed = neighbor_seed
        # When True, tweet-generation SamplingParams omit `seed=` so vLLM
        # samples fresh each invocation. Neighbour-pool RNGs above stay
        # seeded so the per-(agent, round) neighbour set remains reproducible.
        self.nondeterministic_sampling = bool(nondeterministic_sampling)
        # Tweet-generation sampling. Defaults to the optimizer student values
        # (0.7 / 0.9) so every existing caller is byte-identical; the decoding
        # sensitivity sweep overrides these per setting (via generate_test_data
        # --temp/--top_p) to measure output sensitivity to temperature/top_p.
        self.gen_temp = float(gen_temp)
        self.gen_top_p = float(gen_top_p)

    def _build_aligned_tweet_messages(self, agent):
        """Mirror prompt_optimizer._build_user_message_tweet + optimizer student sys prompt.

        Replicates the side effects of agent.step_llm_tweet (frac_distorted_neigh,
        _force_active) so downstream agent state stays consistent. Neighbours come
        from `self.neighbor_pool` when set (Grok-style external pool, re-sampled
        fresh each round, capped at `self.num_neighbors`); otherwise from the
        network via `agent.respond()` (legacy, capped at 5).
        """
        force_active = not self.interaction
        no_content = {"NO_POST", "NO_TWEET"}

        if self.neighbor_pool is not None:
            n = min(self.num_neighbors, len(self.neighbor_pool))
            if n > 0:
                if self.neighbor_seed is not None:
                    sub = np.random.default_rng(
                        np.random.SeedSequence([self.neighbor_seed,
                                                int(agent.ID),
                                                int(self.iterations)])
                    )
                    idx = sub.choice(len(self.neighbor_pool), size=n, replace=False)
                else:
                    idx = self.rng.choice(len(self.neighbor_pool), size=n, replace=False)
                neighbor_pairs = [self.neighbor_pool[i] for i in idx]
            else:
                neighbor_pairs = []
            agent.frac_distorted_neigh = 0
        else:
            # Network neighbours (matches step_llm_tweet).
            activated = agent.respond()
            neighbor_pairs = []
            distorted = 0
            for n in activated:
                if n.activation_state and n.last_tweet:
                    neighbor_pairs.append((n.ID, n.last_tweet))
                    if n.distorted_tweets and n.distorted_tweets[-1]:
                        distorted += 1
            agent.frac_distorted_neigh = distorted / len(activated) if activated else 0
            neighbor_pairs = list(self.rng.permutation(neighbor_pairs)[:5])
        agent._force_active = force_active

        # Own posts since the last PHQ-9 update == same-block context the optimizer uses.
        same_score = []
        if agent._tweets_since_phq9_update > 0:
            recent = agent.tweethistory[-agent._tweets_since_phq9_update:]
            same_score = [t for t in recent if t and t not in no_content]
        if same_score:
            prev_block = "### PREVIOUS POSTS ###\n" + "\n".join(f"- {t}" for t in same_score)
        else:
            prev_block = "### PREVIOUS POSTS ###\n(none yet)"
        if neighbor_pairs:
            prev_block += "\n\n### POSTS FROM OTHERS ###\n" + "\n".join(
                f"- @user_{nid}: {t}" for nid, t in neighbor_pairs
            )

        phq9_score = (agent.well_being or {}).get("phq9_sumscore", 0) or 0
        template = self.prompts["tweet_gen"]["user_template_forced"]
        user_msg = template.format(
            agent_id=agent.ID,
            persona=agent.persona or "unspecified",
            well_being=f"{_phq9_severity(phq9_score)} (PHQ-9: {phq9_score})",
            previous_tweet_block=prev_block,
        )
        # Reorder so the user message reads data → constraints/format → trigger
        # (the prompts JSON puts ### TASK ### before the format block, which leaves
        # the trigger floating in the middle once format_block is appended).
        if self.tweet_format_block:
            task_idx = user_msg.find("### TASK ###")
            if task_idx != -1:
                data_block = user_msg[:task_idx].rstrip()
                task_trigger = user_msg[task_idx:].strip()
                user_msg = (
                    data_block + "\n\n" + self.tweet_format_block.strip()
                    + "\n\n" + task_trigger
                )
            else:
                user_msg = user_msg + "\n\n" + self.tweet_format_block
        return [
            {"role": "system", "content": self.tweet_instruction},
            {"role": "user", "content": user_msg},
        ]

    def _prepare_prompts(self, tokenizer):
        """Build a templated prompt per agent.

        With tweet_instruction set, system prompt + user message + chat-template
        flag match the prompt_optimizer student exactly; otherwise the legacy
        agent-driven prompt builder is used.
        """
        prompts = []
        force_active = not self.interaction
        for agent in self.all_agents:
            if self.tweet_instruction is not None:
                messages = self._build_aligned_tweet_messages(agent)
            else:
                messages = agent.step_llm_tweet(
                    tokenizer, self.rng, round_idx=self.iterations,
                    force_active=force_active, tweet_block_phq9=True,
                )
            templated = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.thinking,
            )
            if self.iterations == 1 and len(prompts) == 0:
                print("=" * 70)
                print(f"[FULL PROMPT] agent {agent.ID}, round 1 (exact string vLLM receives)")
                print("=" * 70)
                print(templated)
                print("=" * 70)
            prompts.append(templated)
        return prompts, self.all_agents

    def _generate_outputs(self, llm, prompts, temp=1.0, top_p=1.0, phq9=False):
        """Run vLLM generation. Sampling matches the optimizer student when aligned."""
        if not prompts:
            return []
        sp_seed = None if self.nondeterministic_sampling else (self.seed + self.iterations)
        if phq9:
            params = SamplingParams(
                temperature=temp, top_p=top_p, max_tokens=1600, seed=None,
            )
        elif self.tweet_instruction is not None:
            # Optimizer student (tweet) sampling: defaults to _batch_student_generate
            # (temp 0.7, top_p 0.9); overridable via gen_temp/gen_top_p for the
            # decoding-parameter sensitivity sweep.
            params = SamplingParams(
                temperature=self.gen_temp, top_p=self.gen_top_p, max_tokens=512,
                seed=sp_seed,
            )
        else:
            params = SamplingParams(
                temperature=0.7, top_p=0.8,
                presence_penalty=0.4, repetition_penalty=1.05,
                max_tokens=8096, seed=sp_seed,
            )
        return llm.generate(prompts, params)

    def _phq9_questionnaire(self, tokenizer, pipe, mistakes, check_point, temp, top_p):
        """Run the PHQ-9 questionnaire for every agent and record per-score MAE."""
        prompts = []
        for agent in self.all_agents:
            prompt = agent.phq9_questionnaire_prompt(
                tokenizer, agent.tweethistory[-(check_point):], force_active=True,
            )
            templated = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            prompts.append(templated)

        out = self._generate_outputs(pipe, prompts, temp=temp, top_p=top_p, phq9=True)

        for agent, answer in zip(self.all_agents, out):
            questionnaire_answers = answer.outputs[0].text.strip()
            if agent.ID < 10:
                print(f"answer agent {agent.ID}: ", questionnaire_answers, "\n\n")
            sum_score = agent.parse_phq9_answers(questionnaire_answers)
            true_score, next_score = self._old_new_phq9(agent, new_phq9=True)
            mistakes[true_score].append(sum_score - true_score)
            agent.update_well_being(next_score, new_phq9=True)
        return mistakes

    def _old_new_phq9(self, agent):
        """Return the agent's current (true) PHQ-9 score and advance to the next."""
        sequence = self.phq9_sequences[agent.ID]
        idx = self.phq9_indices[agent.ID]
        true_score = sequence[idx]
        self.phq9_indices[agent.ID] = (idx + 1) % len(sequence)
        next_score = sequence[self.phq9_indices[agent.ID]]
        return true_score, next_score

    def _apply_outputs_and_update_state(self, agents_w_prompt, out, n_grams, update_score=False):
        """Take vLLM outputs, apply them as agent tweets, then commit all agents."""
        for agent, tweet in zip(agents_w_prompt, out):
            raw = tweet.outputs[0].text.strip()
            agent.send_tweet(max_chars=240, raw_tweet=raw)

        for agent in self.all_agents:
            _ = agent.commit(n_grams=n_grams, update_score=update_score)

    def update_round(self, mistake_dict, tokenizer, pipe, n_grams=[],
                     check_point=20, temp=1.0, top_p=1.0,
                     test_performance=True, time_info=False):
        """One round: generate tweets, commit them, optionally run the PHQ-9 step."""
        self.iterations += 1
        t0 = time.perf_counter()
        prompts, agents_w_prompt = self._prepare_prompts(tokenizer)
        t1 = time.perf_counter()

        out = self._generate_outputs(pipe, prompts)
        t2 = time.perf_counter()

        self._apply_outputs_and_update_state(agents_w_prompt, out, n_grams)
        t3 = time.perf_counter()
        t4 = t3

        if self.iterations % check_point == 0:
            # Each agent's last `check_point` posts form one (persona, PHQ-9) block.
            # Print them now, matching the Grok pipeline's per-block stdout format.
            for agent in self.all_agents:
                recent = agent.tweethistory[-check_point:]
                phq9 = (agent.well_being or {}).get("phq9_sumscore", 0)
                print(f"=== Agent {agent.ID} persona: {agent.persona} phq9: {phq9} ===")
                for j, post in enumerate(recent):
                    print(f"  POST {j}: {post}")

        if self.iterations % check_point == 0 and test_performance:
            mistake_dict = self._phq9_questionnaire(
                tokenizer, pipe, mistake_dict, check_point, temp=temp, top_p=top_p,
            )
        elif self.iterations % check_point == 0:
            for agent in self.all_agents:
                assert agent._tweets_since_phq9_update == check_point, \
                    "agent should have updated its PHQ-9 score every check_point iterations"
                _, new = self._old_new_phq9(agent)
                agent.update_well_being(new, new_phq9=True)
        t4 = time.perf_counter()

        if time_info:
            print(f"Time to prepare prompts: {t1 - t0:.4f} seconds")
            print(f"Time to generate outputs: {t2 - t1:.4f} seconds")
            print(f"Time to apply outputs and update state: {t3 - t2:.4f} seconds")
            if self.iterations % check_point == 0:
                print(f"Time for PHQ-9 update step: {t4 - t3:.4f} seconds")
        return mistake_dict

    def assess_performance(self, mistake_dict):
        """Aggregate per-score errors into mean bias and MAE (overall + per-PHQ-9)."""
        total_bias = []
        bias_per_phq9 = {}
        total_mae = []
        mae_per_phq9 = {}

        for old_score, changes in mistake_dict.items():
            bias_per_phq9[old_score] = np.mean(changes) if changes else 0
            total_bias.extend(changes)
            mae_per_phq9[old_score] = np.mean(np.abs(np.array(changes))) if changes else 0
            total_mae.extend(np.abs(np.array(changes)))

        avg_change = sum(total_bias) / len(total_bias) if total_bias else 0
        total_mae = sum(total_mae) / len(total_mae) if total_mae else 0
        return avg_change, bias_per_phq9, mae_per_phq9, total_mae

    def log_results_to_csv(self, file_path, model_name, temp, top_p, check_point,
                           avg_change, total_mae, bias_per_phq9, mae_per_phq9):
        """Append this run's row to results.csv, replacing any prior row with the same config."""
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "seed": self.seed,
            "num_agents": self.num_agents,
            "temp": temp,
            "top_p": top_p,
            "check_point": check_point,
            "avg_phq9_change": avg_change,
            "total_mae": total_mae,
            "iterations": self.iterations,
        }
        for score, val in bias_per_phq9.items():
            data[f"bias_phq9_{score}"] = val
        for score, val in mae_per_phq9.items():
            data[f"mae_phq9_{score}"] = val

        new_row = pd.DataFrame([data])
        if os.path.isfile(file_path):
            existing = pd.read_csv(file_path)
            same_config = (
                (existing.get("model_name") == model_name) &
                (existing.get("seed") == self.seed) &
                (existing.get("check_point") == check_point) &
                (existing.get("temp") == temp) &
                (existing.get("top_p") == top_p)
            )
            existing = existing.loc[~same_config]
            df_out = pd.concat([existing, new_row], ignore_index=True)
        else:
            df_out = new_row

        df_out.to_csv(file_path, index=False)
        print(f"Results logged to {file_path}")

    def save_checkpoint(self, model_name: str, temp: float, top_p: float,
                        check_point: int, interaction: bool,
                        mistake_dict: dict) -> str:
        """Serialise state via reading_in.write_out_tester. Returns the file path."""
        from utils.tools.reading_in import write_out_tester
        path = write_out_tester(
            self, model_name=model_name, temp=temp, top_p=top_p,
            check_point=check_point, interaction=interaction,
            mistake_dict=mistake_dict,
        )
        print(f"[TestLLMs] Checkpoint saved to {path}")
        return path

    @classmethod
    def load_checkpoint(cls, file_path: str):
        """Restore a TestLLMs instance from a checkpoint file. Returns (tester, mistake_dict)."""
        from utils.tools.reading_in import load_tester_checkpoint
        return load_tester_checkpoint(file_path)

    def run_simulation(self, tokenizer, pipe, n_rounds=100, n_grams=[],
                       check_point=20, temp=1.0, top_p=1.0,
                       model_name="llama_3_8b_instruct",
                       test_performance=True, time_info=False,
                       mistake_dict=None, checkpoint_every=0):
        """Run n_rounds of update_round, optionally checkpointing and assessing perf."""
        if self.iterations == 0:
            mistake_dict = {i: [] for i in range(28)}

        for round_idx in range(n_rounds - self.iterations):
            mistake_dict = self.update_round(
                mistake_dict, tokenizer, pipe,
                n_grams=n_grams, check_point=check_point,
                temp=temp, top_p=top_p,
                test_performance=test_performance, time_info=time_info,
            )
            if checkpoint_every > 0 and (round_idx + 1) % checkpoint_every == 0:
                self.save_checkpoint(
                    model_name=model_name, temp=temp, top_p=top_p,
                    check_point=check_point, interaction=self.interaction,
                    mistake_dict=mistake_dict,
                )

        if test_performance:
            avg_change, bias_per_phq9, mae_per_phq9, total_mae = self.assess_performance(mistake_dict)
            self.log_results_to_csv(
                f"data/test{FC.DIR_SUFFIX}/results.csv", model_name,
                temp=temp, top_p=top_p, check_point=check_point,
                avg_change=avg_change, total_mae=total_mae,
                bias_per_phq9=bias_per_phq9, mae_per_phq9=mae_per_phq9,
            )
            return avg_change, bias_per_phq9, mae_per_phq9, total_mae
        return None, None, None, None

    def export_tweets_with_phq9(self, file_path, check_point, temp, top_p,
                                model_name, interaction=False):
        """Write the per-agent (step, phq9, tweet) CSV consumed downstream.

        Accepts either a .csv or .txt path for backwards compatibility — a
        trailing .txt is rewritten to .csv before writing.
        """
        if file_path.endswith(".txt"):
            file_path = file_path[:-4] + ".csv"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["agent_id", "persona", "age", "step", "phq9", "tweet", "interaction"])
            for agent in self.all_agents:
                phq_series = list(agent.all_phq9_sumscores)
                tweets = list(agent.tweethistory)
                for idx, value in enumerate(phq_series):
                    tweet = tweets[idx] if idx < len(tweets) else ""
                    tweet = tweet.replace("\n", " ").replace("\r", " ").strip()
                    writer.writerow([agent.ID, agent.persona, getattr(agent, "age", None),
                                     idx, value, tweet, interaction])
