import numpy as np
import torch
from classes.agent import Agent
from vllm import LLM, SamplingParams
import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime


class TestLLMs:
    def __init__(self, seed: int = 42, num_agents: int = 5,
                 personas: list = None, well_being: list = None):
        self.rng = np.random.default_rng(seed)
        personas = self.rng.permutation(personas) if personas is not None else [None]*num_agents
        self.well_being = self.rng.permutation(well_being) if well_being is not None else [None]*num_agents
        self.all_agents = [Agent(i, rng=np.random.default_rng(seed + i), persona=personas[i], 
                              well_being=self.well_being[i]) for i in range(int(num_agents))]
        self.iterations = 0
        self.seed = seed
        self.num_agents = num_agents

    def _prepare_prompts(self, tokenizer) -> list:
        prompts = []
        for agent in self.all_agents:
            prompt = agent.step_llm_tweet(tokenizer, self.rng, round_idx=self.iterations, force_active=True)
            prompts.append(prompt)
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
                    max_tokens=100,   # Keep it short; you only need the scores
                    seed=None 
                )
            outputs = llm.generate(prompts, sampling_params_clinical)
            return outputs
    
        # Define Sampling Parameters
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            presence_penalty=0.6,
            max_tokens=256,
            seed= None #self.seed + self.iterations  # vLLM handles seeding here
        )

        # VLLM does batching automatically. 
        outputs = llm.generate(prompts, sampling_params)
        
        return outputs

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
            prompt = agent.phq9_questionnaire_prompt(tokenizer, agent.tweethistory[-(check_point):])
            prompts.append(prompt)
        
        # inference with LLM
        out = self._generate_outputs(pipe, prompts, temp=temp, top_p=top_p)
        # update well-being scores based on responses
        for agent, answer in zip(self.all_agents, out):
            questionnaire_answers = answer.outputs[0].text.strip()
            sum_score = agent.parse_phq9_answers(questionnaire_answers)

            old_sum_score = agent.well_being.get("phq9_sumscore", None)
            if old_sum_score is not None:
                change = sum_score - old_sum_score
                mistakes[old_sum_score].append(change)

                # do sumscore with one higher
                sum_score = max(0, min(27, old_sum_score + 1))  # Ensure score is within valid range
            agent.update_well_being(sum_score)

        return mistakes

    def _apply_outputs_and_update_state(self, agents_w_prompt, out, n_grams, update_score):
        """
        Use LLM outputs to update agents' tweets and activation states,
        then compute distorted tweet statistics for this round.
        """
        # agents send out their tweets
        for agent, tweet in zip(agents_w_prompt, out):
            raw = tweet.outputs[0].text.strip()
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
        self.iterations += 1
        batch_size = 8
        prompts, agents_w_prompt = self._prepare_prompts(tokenizer)
        update_score = False

        if self.iterations % check_point== 0:
            mistake_dict = self._phq9_questionnaire(tokenizer, pipe, mistake_dict, check_point, temp=temp, top_p=top_p)
            update_score = True
        
        # generate outputs in parallel
        out = self._generate_outputs(pipe, prompts)

        # agents send out their tweets + state update + stats
        self._apply_outputs_and_update_state(
            agents_w_prompt, out, n_grams, update_score=update_score
        )

           
    def assess_performance(self, mistake_dict):
        """
        Assess the performance of the agents based on the mistake dictionary.
        """
        total_bias = []
        bias_per_phq9 = {}

        total_acc = []
        acc_per_phq9 = {}

        for old_score, changes in mistake_dict.items():
            bias_per_phq9[old_score] = np.mean(changes) if changes else 0
            total_bias.extend(changes)
            acc_per_phq9[old_score] = np.mean(np.abs(np.array(changes))) if changes else 0
            total_acc.extend(np.abs(np.array(changes)))
        
        if total_bias:
            avg_change = sum(total_bias) / len(total_bias)
        else:
            avg_change = 0
        
        if total_acc:
            total_acc = sum(total_acc) / len(total_acc)
        else:
            total_acc = 0
        
        return avg_change, bias_per_phq9, acc_per_phq9, total_acc

    def log_results_to_csv(self, file_path, model_name, temp, top_p, check_point, 
                           avg_change, total_acc, bias_per_phq9, acc_per_phq9):
        """
        Logs the simulation parameters and results into a CSV file.
        Appends to the file if it already exists.
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
            "total_accuracy": total_acc,
            "iterations": self. 
        }

        for score, val in bias_per_phq9.items():
            data[f"bias_phq9_{score}"] = val

        for score, val in acc_per_phq9.items():
            data[f"acc_phq9_{score}"] = val

        df = pd.DataFrame([data])

        # If file doesn't exists, make header
        file_exists = os.path.isfile(file_path)
        df.to_csv(file_path, mode='a', index=False, header=not file_exists)
        
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
        
        avg_change, bias_per_phq9, acc_per_phq9, total_acc = self.assess_performance(mistake_dict)
        self.log_results_to_csv("data/test/results.csv", model_name, 
                                temp=temp, 
                                top_p=top_p, 
                                check_point=check_point, 
                                avg_change=avg_change, 
                                total_acc=total_acc, 
                                bias_per_phq9=bias_per_phq9, 
                                acc_per_phq9=acc_per_phq9)

        return avg_change, bias_per_phq9, acc_per_phq9, total_acc
        
        