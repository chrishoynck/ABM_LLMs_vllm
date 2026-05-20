import numpy as np
import torch
from classes.agent import Agent
from utils.format_config import FC
from vllm import LLM, SamplingParams
import os
import pandas as pd
import numpy as np
import torch
import csv
from datetime import datetime
from openai import OpenAI, AsyncOpenAI
import asyncio
import time


class TestLLMs:
    def __init__(self, seed: int = 42, num_agents: int = 5,
                 personas: list = None, agents=None, well_being: list = None,
                 deepseek: bool = False, interaction: bool = False):
        self.rng = np.random.default_rng(seed)
        personas = self.rng.permutation(personas) if personas is not None else [None]*num_agents
        self.well_being = self.rng.permutation(well_being) if well_being is not None else [None]*num_agents

        self.num_agents = num_agents
        self.interaction = interaction
        if agents is None:
            self.all_agents = [Agent(i, rng=np.random.default_rng(seed + i), persona=personas[i], 
                                well_being=self.well_being[i]) for i in range(int(num_agents))]
            
        else: 
            self.all_agents = agents

        self.phq9_sequences = {}
        self.phq9_indices = {}
        scores = np.arange(28)
        for agent in self.all_agents:
                self.phq9_sequences[agent.ID] = self.rng.permutation(scores).tolist()
                self.phq9_indices[agent.ID] = 0
                agent.update_well_being(self.phq9_sequences[agent.ID][0]) 

        self.iterations = 0
        self.seed = seed
        self.deepseek = deepseek
        # For testing: assign each agent its own permutation of PHQ-9 scores (0..27)
        # that acts as the "true" target sequence across questionnaires.
        
        self.client = None
        self.api_model = None
        if self.deepseek:
            self.setup_api_client()


    def setup_api_client(self):
        """
        Set up the remote LLM client (xAI Grok, OpenAI-compatible API).

        Configure via environment variables:
            XAI_API_KEY      - required: your xAI API key (https://console.x.ai)
            LLM_API_BASE_URL - base URL (default: https://api.x.ai/v1)
            LLM_API_MODEL    - model name (default: grok-3)
        """
        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key found. Set XAI_API_KEY (from https://console.x.ai) "
                "to use the remote Grok API path."
            )
        base_url = os.environ.get("LLM_API_BASE_URL", "https://api.x.ai/v1")
        self.api_model = os.environ.get("LLM_API_MODEL", "grok-3")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        print(f"[API] Grok client ready: model={self.api_model} base_url={base_url}")

    def _prepare_prompts(self, tokenizer) -> list:
        """
        Prepare the prompts for the agents.
        """
        prompts = []
        # If there is no interaction (independent agents), use forced prompts;
        # otherwise (interaction=True), use non-forced prompts so neighbors are considered.
        force_active = not self.interaction
        for agent in self.all_agents:
            raw_messages = agent.step_llm_tweet(
                tokenizer,
                self.rng,
                round_idx=self.iterations,
                force_active=force_active,
                tweet_block_phq9=True,
            )
            if self.deepseek:
                prompts.append(raw_messages)
            else:
                templated = tokenizer.apply_chat_template(
                    raw_messages, tokenize=False, add_generation_prompt=True, chat_template_kwargs={"enable_thinking": True}
                )
                prompts.append(templated)

        return prompts, self.all_agents
    
    # VLLM
    def _generate_outputs(self, llm, prompts, temp=1.0, top_p=1.0, phq9=False):
        """
        Run vLLM generation. 
        Note: batch_size argument is largely ignored here because vLLM 
        handles batching internally (Continuous Batching).
        """
        if not prompts:
            return []
        
        if phq9: 
            sampling_params_clinical = SamplingParams(
                    temperature=temp, 
                    top_p=top_p,
                    max_tokens=1600,
                    seed=None 
                )
            outputs = llm.generate(prompts, sampling_params_clinical)
            return outputs

        # Define Sampling Parameters
        if not self.deepseek:
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                presence_penalty=0.4,
                repetition_penalty=1.05,
                max_tokens=8096,
                seed= self.seed + self.iterations  # vLLM handles seeding here
            )

            # VLLM does batching automatically. 
            outputs = llm.generate(prompts, sampling_params)

        return outputs
    
    async def _generate_outputs_api(self, prompts, temp=1.0, top_p=1.0,
                                    phq9=False, max_concurrency=8):
        """
        Generate outputs via the remote LLM API (Grok, OpenAI-compatible).

        Parameters
        ----------
        prompts : list[list[dict]]
            One OpenAI-style message list per agent (built by _prepare_prompts
            / phq9_questionnaire_prompt when self.deepseek is True).
        phq9 : bool
            True for the PHQ-9 questionnaire step, False for tweet generation.
        max_concurrency : int
            Cap on simultaneous in-flight requests (keeps within rate limits).

        Returns
        -------
        list[str]
            Generated text per prompt, aligned with `prompts`. Empty string
            on permanent failure so a long run degrades instead of crashing.
        """
        if not prompts:
            return []

        if phq9:
            gen_kwargs = dict(temperature=temp, top_p=top_p, max_tokens=1600)
        else:
            # Mirror the local vLLM tweet sampling: discourage repetition.
            gen_kwargs = dict(temperature=0.7, top_p=0.8, max_tokens=800,
                              presence_penalty=0.4, frequency_penalty=0.3)

        sem = asyncio.Semaphore(max_concurrency)

        async def _one(idx, messages):
            async with sem:
                for attempt in range(6):
                    try:
                        resp = await self.client.chat.completions.create(
                            model=self.api_model,
                            messages=messages,
                            seed=self.seed + self.iterations,
                            **gen_kwargs,
                        )
                        return (resp.choices[0].message.content or "").strip()
                    except Exception as exc:
                        # Honour a server-provided retry delay if present,
                        # otherwise back off exponentially (rate limits etc.).
                        wait = min(60, 2 ** attempt)
                        headers = getattr(getattr(exc, "response", None), "headers", None)
                        if headers:
                            try:
                                wait = float(headers.get("retry-after", wait))
                            except (TypeError, ValueError):
                                pass
                        print(f"[API] prompt {idx} failed (attempt {attempt + 1}/6): "
                              f"{exc} - retrying in {wait:.0f}s")
                        await asyncio.sleep(wait)
                print(f"[API] prompt {idx} permanently failed; returning empty string")
                return ""

        return list(await asyncio.gather(
            *(_one(i, m) for i, m in enumerate(prompts))
        ))
    
    def _phq9_questionnaire(self, tokenizer, pipe, mistakes, check_point, temp, top_p):
        """
        Have all agents complete the PHQ-9 questionnaire via LLM and update their well-being scores.
        Args:
            tokenizer: The tokenizer for the LLM.
            pipe: The LLM pipeline for generating responses.
            check_point: The number of recent tweets to consider for the questionnaire.
        Returns:
            mistakes: A dictionary mapping old PHQ-9 scores to lists of changes in scores.
        """
        # prepare prompts for all agents
        prompts = []
        for agent in self.all_agents:
            prompt = agent.phq9_questionnaire_prompt(tokenizer, agent.tweethistory[-(check_point):], force_active=True)
            if self.deepseek:
                prompts.append(prompt)
            else:
                templated = tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True, chat_template_kwargs={"enable_thinking": True}
                )
                prompts.append(templated)

        # inference with LLM
        if self.deepseek:
            out = asyncio.run(
                self._generate_outputs_api(prompts, temp=temp, top_p=top_p, phq9=True)
            )
        else:
            out = self._generate_outputs(pipe, prompts, temp=temp, top_p=top_p, phq9=True)
        # update well-being scores based on responses
        for agent, answer in zip(self.all_agents, out):
            questionnaire_answers = answer if self.deepseek else answer.outputs[0].text.strip()
            if agent.ID < 10:
                print(f"answer agent {agent.ID}: ", questionnaire_answers, "\n\n")
            sum_score = agent.parse_phq9_answers(questionnaire_answers)

            true_score, next_score = self._old_new_phq9(agent, new_phq9=True)

            # # Determine this agent's current true PHQ-9 score from its own permutation sequence.
            # sequence_phq9 = self.phq9_sequences[agent.ID]
            # idx_sequence = self.phq9_indices[agent.ID]
            # true_score = sequence_phq9[idx_sequence]

            # Record the LLM's error (mae, bias)
            change = sum_score - true_score
            mistakes[true_score].append(change)

            # increase index (wrap around if we ever exceed the permutation length)
            # self.phq9_indices[agent.ID] = (idx_sequence + 1) % len(sequence_phq9)
            # idx_sequence = self.phq9_indices[agent.ID]
            # next_score = sequence_phq9[idx_sequence]

            # update well-being score
            agent.update_well_being(next_score, new_phq9=True)
        return mistakes

    def _old_new_phq9(self, agent):
        """
        Get the old and new PHQ-9 scores for an agent.
        """
        sequence_phq9 = self.phq9_sequences[agent.ID]
        idx_sequence = self.phq9_indices[agent.ID]
        true_score = sequence_phq9[idx_sequence]

        # update
        self.phq9_indices[agent.ID] = (idx_sequence + 1) % len(sequence_phq9)
        idx_sequence = self.phq9_indices[agent.ID]
        
        
        next_score = sequence_phq9[idx_sequence]


        return true_score, next_score

    def _apply_outputs_and_update_state(self, agents_w_prompt, out, n_grams, update_score=False):
        """
        Use LLM outputs to update agents' tweets and activation states,
        then compute distorted tweet statistics for this round.
        """
        # agents send out their tweets
        for agent, tweet in zip(agents_w_prompt, out):
            
            if not self.deepseek:
                raw = tweet.outputs[0].text.strip()
            else:
                raw = tweet
            agent.send_tweet(
                max_chars=240,  #CHANGED THIS FROM 240 TO 1000
                raw_tweet=raw,
            )

            # if agent.ID < 10:
            #     current_phq9 = agent.well_being.get("phq9_sumscore") if agent.well_being else -1
            #     print(f"tweet agent {agent.ID}: ", tweet.outputs[0].text.strip(), "phq9: ", current_phq9, "\n\n")

        for agent in self.all_agents:
            _ = agent.commit(n_grams=n_grams, update_score=update_score)


    def update_round(self, 
                    mistake_dict, 
                    tokenizer, 
                    pipe, 
                    n_grams=[], 
                    check_point= 20, 
                    temp=1.0, 
                    top_p=1.0, 
                    test_performance=True,
                    time_info=False):
        """
        Update the network for one round by responding to news intensities and adjusting the network accordingly.
        """
        self.iterations +=1
        t0 = time.perf_counter()
        prompts, agents_w_prompt = self._prepare_prompts(tokenizer)
        t1 = time.perf_counter()
        # generate outputs in parallel

        if not self.deepseek:
            out = self._generate_outputs(pipe, prompts)
        else:
            out = asyncio.run(
                self._generate_outputs_api(prompts, temp=temp, top_p=top_p)
            )
        t2 = time.perf_counter()

        # agents send out their tweets + state update + stats
        self._apply_outputs_and_update_state(
            agents_w_prompt, out, n_grams
        )
        t3 = time.perf_counter()

        t4 = t3
        if self.iterations % check_point == 0 and test_performance:
            mistake_dict = self._phq9_questionnaire(tokenizer, pipe, mistake_dict, check_point, temp=temp, top_p=top_p)
        elif self.iterations % check_point == 0:
            for agent in self.all_agents:
                assert agent._tweets_since_phq9_update == check_point, "agent should have updated its PHQ-9 score every check_point iterations"
                old, new = self._old_new_phq9(agent)
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
        """
        Assess the performance of the agents based on the mistake dictionary.
        """
        total_bias = []
        bias_per_phq9 = {}

        # Mean absolute error (MAE) in PHQ-9 points, not accuracy.
        total_mae = []
        mae_per_phq9 = {}

        for old_score, changes in mistake_dict.items():
            bias_per_phq9[old_score] = np.mean(changes) if changes else 0
            total_bias.extend(changes)
            mae_per_phq9[old_score] = np.mean(np.abs(np.array(changes))) if changes else 0
            total_mae.extend(np.abs(np.array(changes)))
        
        if total_bias:
            avg_change = sum(total_bias) / len(total_bias)
        else:
            avg_change = 0
        
        if total_mae:
            total_mae = sum(total_mae) / len(total_mae)
        else:
            total_mae = 0
        
        # Return mean bias, per-score MAE, and overall MAE (misnamed "accuracy" previously).
        return avg_change, bias_per_phq9, mae_per_phq9, total_mae

    def log_results_to_csv(self, file_path, model_name, temp, top_p, check_point, 
                           avg_change, total_mae, bias_per_phq9, mae_per_phq9):
        """
        Logs the simulation parameters and results into a CSV file.

        For a given combination of model, seed and settings, only the most recent
        run is kept (the previous row is replaced). Runs with different seeds
        still appear as separate lines.
        """
        # Prepare the data row
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
            "iterations": self.iterations}

        for score, val in bias_per_phq9.items():
            data[f"bias_phq9_{score}"] = val

        for score, val in mae_per_phq9.items():
            data[f"mae_phq9_{score}"] = val

        new_row = pd.DataFrame([data])
        
        if os.path.isfile(file_path):
            # Load existing results and drop any previous row with the same
            # (model_name, seed, check_point, temp, top_p) so that only the
            # latest run for this configuration is kept.
            existing = pd.read_csv(file_path)
            same_config = (
                (existing.get("model_name") == model_name) &
                (existing.get("seed") == self.seed) &
                (existing.get("check_point") == check_point) &
                (existing.get("temp") == temp) &
                (existing.get("top_p") == top_p)
            )
            # drop previous row with the same configuration
            existing = existing.loc[~same_config]
            df_out = pd.concat([existing, new_row], ignore_index=True)
        else:
            df_out = new_row

        df_out.to_csv(file_path, index=False)
        
        print(f"Results logged to {file_path}")
    
    # ------------------------------------------------------------------
    #  Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, model_name: str, temp: float, top_p: float,
                        check_point: int, interaction: bool,
                        mistake_dict: dict) -> str:
        """
        Serialise the current state to a JSON checkpoint via
        :func:`reading_in.write_out_tester`.

        Returns
        -------
        str  – path the file was written to
        """
        from utils.reading_in import write_out_tester
        path = write_out_tester(
            self, model_name=model_name, temp=temp, top_p=top_p,
            check_point=check_point, interaction=interaction,
            mistake_dict=mistake_dict,
        )
        print(f"[TestLLMs] Checkpoint saved to {path}")
        return path

    @classmethod
    def load_checkpoint(cls, file_path: str):
        """
        Restore a :class:`TestLLMs` instance from a checkpoint file.

        Returns
        -------
        (tester, mistake_dict)
        """
        from utils.reading_in import load_tester_checkpoint
        return load_tester_checkpoint(file_path)

    # ------------------------------------------------------------------

    def run_simulation(self, 
                       tokenizer, 
                       pipe, 
                       n_rounds=100, 
                       n_grams=[], 
                       check_point=20, 
                       temp=1.0, 
                       top_p=1.0, 
                       model_name="llama_3_8b_instruct", 
                       test_performance=True,
                       time_info=False,
                       mistake_dict=None,
                       checkpoint_every=0):
        """
        Run the full simulation for a specified number of rounds.

        Parameters
        ----------
        checkpoint_every : int
            Save a checkpoint every this many rounds (0 = disabled)
        """

        if self.iterations == 0:
            mistake_dict = {i: [] for i in range(0, 28)}  # PHQ-9 scores range from 0 to 27
        else:
            mistake_dict = mistake_dict


        for round_idx in range(n_rounds - self.iterations):
            mistake_dict = self.update_round(mistake_dict, 
                              tokenizer,
                              pipe,
                              n_grams=n_grams,
                              check_point=check_point,
                              temp=temp,
                              top_p=top_p,
                              test_performance=test_performance,
                              time_info=time_info)

            if checkpoint_every > 0 and (round_idx + 1) % checkpoint_every == 0:
                self.save_checkpoint(
                    model_name=model_name, temp=temp, top_p=top_p,
                    check_point=check_point, interaction=self.interaction,
                    mistake_dict=mistake_dict,
                )
        
        if test_performance:
            avg_change, bias_per_phq9, mae_per_phq9, total_mae = self.assess_performance(mistake_dict)
            self.log_results_to_csv(f"data/test{FC.DIR_SUFFIX}/results.csv", model_name, 
                                    temp=temp, 
                                    top_p=top_p, 
                                    check_point=check_point, 
                                    avg_change=avg_change, 
                                    total_mae=total_mae, 
                                    bias_per_phq9=bias_per_phq9, 
                                    mae_per_phq9=mae_per_phq9)

            return avg_change, bias_per_phq9, mae_per_phq9, total_mae
        return None, None, None, None

    def export_tweets_with_phq9_txt(self, file_path, check_point, temp, top_p, model_name, interaction=False):
        """
        Write a plain-text log for this TestLLMs run, with tweet history and
        PHQ-9 values per agent, and explicit markers when PHQ-9 changes.

        Format (per file):
        - Header with run settings (model, seed, temp, top_p, check_point, num_agents)
        - Then, for each agent:
            === Agent <id> ===
            initial_phq9: <value or None>
            step <i>: phq9=<val>  tweet="<text>" [CHANGED from <old>]
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        lines: list[str] = []
        # Run-level header
        lines.append(f"model_name: {model_name}")
        lines.append(f"seed: {self.seed}")
        lines.append(f"num_agents: {self.num_agents}")
        lines.append(f"temp: {temp}")
        lines.append(f"top_p: {top_p}")
        lines.append(f"check_point: {check_point}")
        lines.append(f"interaction: {interaction}")
        lines.append("")  # blank line

        for agent in self.all_agents:
            lines.append(f"=== Agent {agent.ID} persona: {agent.persona} ===")
            # (both start empty, both get one entry per commit()).
            phq_series = list(agent.all_phq9_sumscores)
            tweets = list(agent.tweethistory)
            lines.append(f"initial_phq9: {phq_series[0] if phq_series else None}")

            # prev = None
            for idx, value in enumerate(phq_series):
                tweet = tweets[idx] if idx < len(tweets) else ""
                # Collapse newlines so each step stays on one line in the file
                tweet = tweet.replace("\n", " ").replace("\r", " ").strip()
                line = f"step {idx}: phq9={value}  tweet=\"{tweet}\""
                # if prev is not None and value != prev:
                #     line += f"  (CHANGED from {prev})"
                lines.append(line)
                # prev = value

            lines.append("")  # blank line between agents

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        # Write to CSV
        csv_path = file_path.replace(".txt", ".csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["agent_id", "persona", "age", "step", "phq9", "tweet", "interaction"])

            for agent in self.all_agents:
                phq_series = list(agent.all_phq9_sumscores)
                tweets = list(agent.tweethistory)

                for idx, value in enumerate(phq_series):
                    tweet = tweets[idx] if idx < len(tweets) else ""
                    tweet = tweet.replace("\n", " ").replace("\r", " ").strip()
                    writer.writerow([agent.ID, agent.persona, getattr(agent, "age", None), idx, value, tweet, interaction])