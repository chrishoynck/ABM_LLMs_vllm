import textgrad as tg
from textgrad.engine.vllm import ChatVLLM
import json
import numpy as np
import os
import re
import csv
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from src.utils.metrics import *
from src.utils.format_config import FC
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

            tweet = (row.get("tweet") or "").strip()

            if agent_id != current_agent or phq9 != current_phq9:
                # Flush previous block for previous agent/score
                if current_tweets and current_phq9 is not None:
                    if len(current_tweets) > 1:
                        tweet_blocks.append(current_tweets)
                        true_answers.append(current_phq9)

                current_agent = agent_id
                current_phq9 = phq9
                current_tweets = [tweet] if tweet else []
            else:
                if tweet:
                    current_tweets.append(tweet)

        # Flush last block
        if current_tweets and current_phq9 is not None:
            if len(current_tweets) > 1:
                tweet_blocks.append(current_tweets)
                true_answers.append(current_phq9)

    return tweet_blocks, true_answers


FORMAT_SPLIT_MARKER = "### OPTIONS ###"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"


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


def call_optimizer(
    file_path: str,
    model_name: str = LLAMA_70B,
    batch_size: int = 4,
    max_instruction_words: int = 300,
    num_steps: int = None,
    val_fraction: float = 0.15,
    validate_every: int = 5,
    seed: int = 42,
    **vllm_kwargs,
):
    """
    Optimise the PHQ-9 instruction (task framing + reasoning guidance)
    using TextGrad with batched gradient accumulation.

    The PHQ-9 questions, scoring options, and answer format are kept fixed
    and passed in the user message so they are never modified.

    Parameters:
        file_path (str): Path to a tweets_with_phq9.txt produced by TestLLMs.
        model_name (str): HuggingFace model ID for both the forward and backward (teacher)
            engine.  Defaults to Llama-3.3-70B-Instruct.
        batch_size (int): Number of samples per gradient step.
        max_instruction_words (int): Soft word-count cap enforced via TGD constraints.
        num_steps (int or None): Optimisation steps.  None -> one epoch over the training set.
        val_fraction (float): Fraction of data reserved for validation.
        validate_every (int): Run validation every N steps.
        seed (int): RNG seed for data shuffling.
        **vllm_kwargs: Extra kwargs forwarded to ChatVLLM (e.g. dtype).
    """
    if file_path.endswith(".csv"):
        csv_path = file_path
        txt_path = file_path.replace(".csv", ".txt")
    else:
        txt_path = file_path
        csv_path = file_path.replace(".txt", ".csv") if file_path.endswith(".txt") else file_path + ".csv"

    if os.path.isfile(csv_path):
        tweet_blocks, true_answers = parse_tweets_with_phq9_csv(csv_path)
        print(f"Parsed {len(tweet_blocks)} tweet blocks from {csv_path}")
    else:
        tweet_blocks, true_answers = parse_tweets_with_phq9(txt_path)
        print(f"Parsed {len(tweet_blocks)} tweet blocks from {txt_path} (CSV not found, used TXT parser)")

    # Train / validation split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tweet_blocks))

    # ADDRESS THIS:
    n_val = max(1, int(len(tweet_blocks) * val_fraction/5)) # just for now, reduce the size of the validation set to 1/5 of the data for validation

    val_blocks = [tweet_blocks[i] for i in perm[:n_val]]
    val_answers = [true_answers[i] for i in perm[:n_val]]
    train_blocks = [tweet_blocks[i] for i in perm[n_val:]]
    train_answers = [true_answers[i] for i in perm[n_val:]]
    print(f"Train: {len(train_blocks)},  Validation: {len(val_blocks)}")

    # Engine (single model for forward + backward)
    tp = vllm_kwargs.pop("tensor_parallel_size", None) or torch.cuda.device_count()
    engine = ChatVLLM(
        model_string=model_name,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        **vllm_kwargs,
    )
    tg.set_backward_engine(engine, override=True)

    # --- split prompt into instruction (grad) + format (fixed) --------------
    with open(FC.PROMPTS_FILE, "r") as f:
        prompts = json.load(f)

    # full_system = prompts["phq9"]["system"]
    # instruction_text, format_block = _split_system_prompt(full_system)

    instruction_text = prompts["phq9"]["system_instruction"]
    format_block = prompts["phq9"]["System_format"]

    instruction = tg.Variable(
        instruction_text,
        role_description=(
            "System-prompt INSTRUCTION for PHQ-9 assessment from tweets. "
            "Contains task framing and step-by-step reasoning guidance only. "
            "The PHQ-9 questions, scoring options, and answer format are provided separately and must NOT be duplicated here."
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

    # Optimisation loop
    for step in range(num_steps):
        batch_idx = rng.choice(
            len(train_blocks),
            size=min(batch_size, len(train_blocks)),
            replace=False,
        )
        batch_blocks = [train_blocks[i] for i in batch_idx]
        batch_answers = [train_answers[i] for i in batch_idx]

        optimizer.zero_grad()
        model = tg.BlackboxLLM(engine, system_prompt=instruction)

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

        # --- periodic validation --------------------------------------------
        if (step + 1) % validate_every == 0 or step == num_steps - 1:
            print(f"Validating on {len(val_blocks)} validation blocks \n\n")
            val_mae = _evaluate_instruction(
                engine, instruction.value, format_block,
                val_blocks, val_answers, prompts,
            )
            print(f"  -> Validation MAE: {val_mae:.2f}  (best: {best_val_mae:.2f})")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_instruction = instruction.value
                print("  -> New best instruction saved!")

    # --- save results -------------------------------------------------------
    output_dir = os.path.dirname(file_path)

    instr_path = os.path.join(output_dir, "optimized_instruction.txt")
    with open(instr_path, "w", encoding="utf-8") as f:
        f.write(best_instruction)
    print(f"\nBest instruction (val MAE={best_val_mae:.2f}) → {instr_path}")

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

def split_embeddings_and_labels(embeddings, labels, train_frac=0.8, val_frac=0.1, seed=None):
    """
    Split embeddings and labels into train/val/test after loading.
    Uses the same 80/10/10 proportions as create_dataset.
    Parameters:
        embeddings (torch.Tensor): The embeddings to split.
        labels (torch.Tensor): The labels to split.
        train_frac (float): The fraction of the data to use for training.
        val_frac (float): The fraction of the data to use for validation.
        seed (int): The seed to use for the random number generator.
    Returns:
        train_embs (torch.Tensor): The embeddings for the training data.
        val_embs (torch.Tensor): The embeddings for the validation data.
        test_embs (torch.Tensor): The embeddings for the test data.
        train_labels (torch.Tensor): The labels for the training data.
        val_labels (torch.Tensor): The labels for the validation data.
        test_labels (torch.Tensor): The labels for the test data.
    """
    n = len(embeddings)
    rng = np.random.default_rng(seed)
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

    if "embeddings" in data and "labels" in data:
        # New format: single array; split after loading
        all_embs = data["embeddings"]
        all_labels = data["labels"]
        train_embs, val_embs, test_embs, train_labels, val_labels, test_labels = split_embeddings_and_labels(
            all_embs, all_labels, train_frac=0.8, val_frac=0.1, seed=split_seed
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
    
if __name__ == "__main__":
    create_new_embeddings = False
    mental_bert = True
    file_path = "data/test/meta-llama_Llama-3.1-8B-Instruct/temp_0.8_top_p_0.6_cp_10/seed_71/tweets_with_phq9.txt"
    base_model_name = "meta-llama_Llama-3.1-8B-Instruct"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    run_prompt_optimizer = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if run_prompt_optimizer:
        call_optimizer(file_path, model_name=model_name, batch_size=4)
    else:
        if create_new_embeddings:
            save_embeddings_for_file(file_path, base_model_name, device, mentalbert=mental_bert)
        
        if mental_bert:
            dir_name = "mentalbert_embeddings"
        else:
            dir_name = "sbert_embeddings"

        embeddings_path = os.path.join("data", "test", base_model_name, dir_name, "embeddings_and_labels.pt")
        train_BERT_model(embeddings_path, base_model_name, device, mental_bert=mental_bert)