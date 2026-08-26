# Prompt Optimization with TextGrad

This README explains how to run prompt optimization for the two LLM tasks in the project:

1. **Post generation** — getting an agent to write posts that match a persona + PHQ-9 profile.
2. **PHQ-9 assessment** — getting an agent to infer a PHQ-9 score from a user's post history.

Both are implemented in [src/utils/prompt_optimizer.py](src/utils/prompt_optimizer.py) and use [TextGrad](https://github.com/zou-group/textgrad) in a student–teacher loop.

---

## Method

(longer version is in Thesis appendix, **Appendix: Prompt Optimization**.)

We use TextGrad as a **student–teacher** setup:
- The **student** LLM runs the actual task (write posts / fill in PHQ-9).
- The **teacher** LLM grades the student's output and produces *textual gradients* — i.e. natural-language feedback on what the system prompt should do differently.
- TextGrad backprops that feedback into the prompt and rewrites it.

We only optimize the **behavioral / stylistic** part of the prompt. Hard constraints (character limits, JSON format, the fixed PHQ-9 format block) are kept frozen so outputs stay parseable.

Each rewrite step is bounded: stay under ~180 words, no hallucinated concepts, preserve the fixed formatting and the allowance for harsh/negative emotional ranges.

### Starting prompts (from [data/prompts_post_minimal.json](data/prompts_post_minimal.json))

Only the system instruction is optimized — the format block and output template stay frozen.

- **PHQ-9 (`phq9.system_instruction`):** *"Fill in the PHQ-9 questionnaire based on the post history. Infer symptoms from emotional tone"*
- **Post generation (`tweet_gen.system_forced`, rules block only):** *"Base your TONE and MOOD on your well-being (PHQ-9)… Low PHQ-9 -> positive/relaxed; High PHQ-9 -> apathetic, irritable, overwhelmed, raw. Pick a new, specific TOPIC … Be ORIGINAL. Interact with others when natural."* (full text in the JSON)

### Input / output paths

- **Input** (per-agent post histories + true PHQ-9 scores): pulled from `data/test_post/Qwen_Qwen3.5-27B/temp_0.8_top_p_0.6_cp_10_{inter,no_inter}/seed_*/tweets_with_phq9.csv` — set in `__main__`.
- **Outputs (PHQ-9):** `data/test_post/optimized_phq9/{model_short}_seed{seed}/` — contains `best_instruction.txt`, `best_full_prompt.txt`, `latest_instruction_phq9.txt`, `training_trajectory.csv`, `optimized_instruction.txt`, `test_scores_phq9.csv`.
- **Outputs (tweets):** `data/test_post/optimized_tweets/{model_short}_seed{seed}/` — same idea (`best_instruction_tweet.txt`, `training_trajectory.csv`, etc.).

### Post generation loop (per step)
- **Sample** a batch of (persona, PHQ-9 score) profiles. Ground the student with the agent's own historical posts + a few neighbouring posts.
- **Generate** N=3 posts per profile with the current prompt.
- **Evaluate** the *set* of posts (not single posts, lets the teacher judge within-user variation and mood consistency) on: Tone/Mood, Naturalness, Originality, Unfiltered, Diversity, Interaction.
- **Update** the prompt from the textual feedback.

### PHQ-9 assessment loop (per step)
- **Sample** a batch of simulated users (persona + post history + true PHQ-9).
- **Predict** PHQ-9 with the student model.
- **Evaluate** the gap between predicted and true score direction *and* magnitude so the teacher can critique the reasoning, not just the number.
- **Update** the prompt from the textual feedback.

Validation runs every 2 steps; final prompt is scored on a held-out test set, and the whole thing is repeated across multiple seeds because lightweight models have high variance.

---

## File outline — [src/utils/prompt_optimizer.py](src/utils/prompt_optimizer.py)

One big file, but it splits into a few clear blocks. Skim this before diving in.

**Engines & shared plumbing (top of file)**
- `_StudentEngine` / `_TeacherEngine` — thin `ChatVLLM` wrappers. Student runs with reasoning *off* (fast, deterministic-ish output); teacher runs with thinking *on* so it can produce sensible gradients/critiques.
- `_build_engines` loads the vLLM model into memory and returns both engine wrappers.
- `_extract_score_feedback`, `_teacher_call_kind` — small helpers to parse `SCORE:` / `FEEDBACK:` out of teacher dumps and to tell whether a teacher call is a backward / optimizer / loss step.
- `_OPTIMIZER_SYSTEM_PROMPT` overrides TextGrad's default optimizer prompt (the default trips up thinking models).

**Data loading**
- `parse_tweets_with_phq9` (`.txt`) and `parse_tweets_with_phq9_csv` — read per-agent tweet blocks + true PHQ-9 scores out of the dataset files.
- `train_val_test_split` splits agents into train/val/test for the optimizer.
- `_generate_file_path` globs `seed_*` subfolders to gather input files.

**PHQ-9 optimizer (assessment task)**
- `call_optimizer_phq9`  **the main entry point for `--mode phq9`.** Owns the full TextGrad loop: sample batch -> student predicts PHQ-9 -> teacher computes textual loss-> prompt update -> periodic val eval -> final test eval.
- `_build_user_message`, `_evaluate_instruction`, `_make_loss_prompt` build student inputs, score a prompt on a val/test set, and frame the teacher's loss as a critique of (predicted vs. true) gap.

**Post-generation optimizer (tweets task)**
- `call_optimizer_tweets` **the main entry point for `--mode tweets`.** Same TextGrad loop, but the student creates a set of N posts per profile and the teacher rates the *set*.
- `_build_user_message_tweet`, `_sample_neighbor_tweets`, `_phq9_severity` — grounding helpers (persona + history + neighbour posts + severity bucket).
- `_batch_student_generate`, `_batch_teacher_rate` batched vLLM calls so a step doesn't take forever.
- `_make_loss_prompt_tweet_set` packs the six evaluation criteria (Tone, Naturalness, Originality, Unfiltered, Diversity, Interaction) into a single teacher prompt.
- `_evaluate_tweet_instruction`, `parse_tweet_answers` score a prompt on val/test and parse the JSON output the student returns.

**BERT regressor (separate, `--mode bert`)**
- `setup_BERT_model`, `save_embeddings_for_file`, `train_BERT_model`, `neural_net_BERT`, `train_bert`, `evaluate_bert` — SBERT/MentalBERT embeddings + a small MLP regressor trained with Huber loss. Unrelated to the prompt optimization itself; lives here because it shares the dataset parsers.

**`__main__`**
- Parses CLI args, then dispatches on `--mode` to `call_optimizer_phq9`, `call_optimizer_tweets`, or the BERT training path.

If you only care about the prompts, the two functions to read are `call_optimizer_phq9` and `call_optimizer_tweets`: everything else is supporting functions.

---

## How to run

### Environment

The vLLM environment used by both jobs can be installed from [requirements_vllm.txt](requirements_vllm.txt) using [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv .venv_vllm
source .venv_vllm/bin/activate
uv pip install -r requirements_vllm.txt
```

### Example job script (post generation)

A working SLURM script — this is [jobs/run_prompt_optimizer.job](jobs/run_prompt_optimizer.job):

```bash
#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=2
#SBATCH --job-name=Run_model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load CUDA/12.8.0
cd ~/thesis/ABM_based_LLMS/
source .venv_vllm/bin/activate

# Check whether the GPU is available
srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"
PYTHONPATH=src srun python -m utils.prompt_optimizer --seeds 22 23 --num-steps 8 --batch-size 7 --val-sample-size 40 --test-sample-size 120
```

Submit it with:

```bash
sbatch jobs/run_prompt_optimizer.job
```

### PHQ-9 variant — [jobs/run_prompt_optimizer_phq9.job](jobs/run_prompt_optimizer_phq9.job)

Same template, but add `--mode phq9` and bump the partition to `gpu_h100` / walltime to 8h:

```bash
PYTHONPATH=src srun python -m utils.prompt_optimizer \
    --seeds 23 24 25 26 --num-steps 8 \
    --batch-size 10 --val-sample-size 40 --test-sample-size 100 \
    --mode phq9
```

Submit it with:

```bash
sbatch jobs/run_prompt_optimizer_phq9.job
```

---

## Useful flags

| Flag | What it does |
|---|---|
| `--mode {tweets,phq9,bert}` | Which optimizer to run. `tweets` = post generation, `phq9` = assessment. |
| `--seeds` | One or more seeds. Each seed runs the full optimization independently. |
| `--num-steps` | Optimizer steps per seed. |
| `--batch-size` | Profiles sampled per step. |
| `--val-sample-size` / `--test-sample-size` | Held-out set sizes for the periodic val / final test scoring. |
| `--model` | Student/teacher HF model id (defaults to Qwen 27B). |

That's it — pick a mode, pick seeds, submit the job.
