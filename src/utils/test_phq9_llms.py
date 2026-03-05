import numpy as np
import torch
from classes.agent import Agent
from vllm import LLM, SamplingParams
import os
import pandas as pd
import numpy as np
import torch
import csv
from datetime import datetime
from openai import OpenAI, AsyncOpenAI
import asyncio


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
        if self.deepseek:
            self.setup_deepseek()
        

    def setup_deepseek(self):
        """
        Setup the DeepSeek client.
        """
        api_key = "sk-b6503fb67b2b44829ee430515b13b1e7"
        base_url = "https://api.deepseek.com"
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

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
                    raw_messages, tokenize=False, add_generation_prompt=True
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
                    max_tokens=300,   
                    seed=None 
                )
            outputs = llm.generate(prompts, sampling_params_clinical)
            return outputs

        # Define Sampling Parameters
        if not self.deepseek:
            sampling_params = SamplingParams(
                temperature=1.0,
                top_p=0.9,
                presence_penalty=0.4,
                repetition_penalty=1.05,
                max_tokens=256,
                seed= self.seed + self.iterations  # vLLM handles seeding here
            )

            # VLLM does batching automatically. 
            outputs = llm.generate(prompts, sampling_params)
        return outputs
    
    # async def _generate_outputs_deepseek(self, prompts, temp=1.0, top_p=1.0):
    #     async def generate_outje(prompt):
    #         response = await self.client.chat.completions.create(
    #             model="deepseek-chat",
    #             messages=prompt,
    #             temperature=temp,
    #             top_p=top_p,
    #             seed=self.seed + self.iterations)
    #         return response.choices[0].message.content
    #     todos = [generate_outje(prompt) for prompt in prompts]
    #     outputs = await asyncio.gather(*todos)
    #     return outputs
    
    def _phq9_questionnaire(self, tokenizer, pipe,mistakes, check_point, temp, top_p):
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
            templated = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            prompts.append(templated)
        
        # inference with LLM
        out = self._generate_outputs(pipe, prompts, temp=temp, top_p=top_p, phq9=True)
        # update well-being scores based on responses
        for agent, answer in zip(self.all_agents, out):
            if agent.ID == 0:
                print("answer agent 0: ", answer.outputs[0].text.strip(), "\n\n")
            questionnaire_answers = answer.outputs[0].text.strip()
            sum_score = agent.parse_phq9_answers(questionnaire_answers)

            # Determine this agent's current true PHQ-9 score from its own permutation sequence.
            sequence_phq9 = self.phq9_sequences[agent.ID]
            idx_sequence = self.phq9_indices[agent.ID]
            true_score = sequence_phq9[idx_sequence]

            # Record the LLM's error (mae, bias)
            change = sum_score - true_score
            mistakes[true_score].append(change)

            # increase index (wrap around if we ever exceed the permutation length)
            self.phq9_indices[agent.ID] = (idx_sequence + 1) % len(sequence_phq9)
            idx_sequence = self.phq9_indices[agent.ID]
            next_score = sequence_phq9[idx_sequence]

            # update well-being score
            agent.update_well_being(next_score)
        return mistakes

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
                max_chars=240,
                raw_tweet=raw,
            )

        for agent in self.all_agents:
            _ = agent.commit(n_grams=n_grams, update_score=update_score)


    def update_round(self, mistake_dict, tokenizer, pipe, n_grams=[], check_point= 20, temp=1.0, top_p=1.0):
        """
        Update the network for one round by responding to news intensities and adjusting the network accordingly.
        """
        self.iterations +=1
        prompts, agents_w_prompt = self._prepare_prompts(tokenizer)
        # generate outputs in parallel

        if not self.deepseek:
            out = self._generate_outputs(pipe, prompts)
        else:
            out = None
            # out = asyncio.run(self._generate_outputs_deepseek(prompts))

        # agents send out their tweets + state update + stats
        self._apply_outputs_and_update_state(
            agents_w_prompt, out, n_grams
        )

        if self.iterations % check_point  == 0:
            mistake_dict = self._phq9_questionnaire(tokenizer, pipe, mistake_dict, check_point, temp=temp, top_p=top_p)
  
        
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
    
    def run_simulation(self, tokenizer, pipe, n_rounds=100, n_grams=[], check_point=20, temp=1.0, top_p=1.0, model_name="llama_3_8b_instruct"):
        """
        Run the full simulation for a specified number of rounds.
        """
        self.iterations = 0
        mistake_dict = {i: [] for i in range(0, 28)}  # PHQ-9 scores range from 0 to 27

        for round_idx in range(n_rounds):
            self.update_round(mistake_dict, 
                              tokenizer, 
                              pipe, 
                              n_grams=n_grams, 
                              check_point=check_point, 
                              temp=temp, 
                              top_p=top_p)
        
        avg_change, bias_per_phq9, mae_per_phq9, total_mae = self.assess_performance(mistake_dict)
        self.log_results_to_csv("data/test/results.csv", model_name, 
                                temp=temp, 
                                top_p=top_p, 
                                check_point=check_point, 
                                avg_change=avg_change, 
                                total_mae=total_mae, 
                                bias_per_phq9=bias_per_phq9, 
                                mae_per_phq9=mae_per_phq9)

        return avg_change, bias_per_phq9, mae_per_phq9, total_mae

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
            writer.writerow(["agent_id", "persona", "step", "phq9", "tweet", "interaction"])

            for agent in self.all_agents:
                phq_series = list(agent.all_phq9_sumscores)
                tweets = list(agent.tweethistory)

                for idx, value in enumerate(phq_series):
                    tweet = tweets[idx] if idx < len(tweets) else ""
                    tweet = tweet.replace("\n", " ").replace("\r", " ").strip()
                    writer.writerow([agent.ID, agent.persona, idx, value, tweet, interaction])