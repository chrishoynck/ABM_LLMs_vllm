import textgrad as tg
from textgrad.engine.vllm import ChatVLLM
import json
import numpy as np
import os
import re
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from src.utils.metrics import *
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

        # New agent header → flush current block and reset
        if line.startswith("=== Agent"):
            if current_tweets:
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
            # New PHQ-9 period → save the previous block (if any)
            if current_tweets:
                tweet_blocks.append(current_tweets)
                true_answers.append(current_phq9)
            current_phq9 = phq9
            current_tweets = [tweet]
        else:
            current_tweets.append(tweet)

    # Flush the last block
    if current_tweets:
        tweet_blocks.append(current_tweets)
        true_answers.append(current_phq9)

    return tweet_blocks, true_answers


def call_optimizer(file_path: str, model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """
    Parse tweet data from file_path, optimise the PHQ-9 system
    instruction using textgrad, and write the resulting prompt to
    the same directory as the input file.

    Parameters:
        file_path (str): Path to a tweets_with_phq9.txt file.
        model_name (str): model ID used for both forward and backward engine.
    """
    tweet_blocks, true_answers = parse_tweets_with_phq9(file_path)
    print(f"Parsed {len(tweet_blocks)} tweet blocks from {file_path}")

    engine = ChatVLLM(model_string=model_name)
    tg.set_backward_engine(engine, override=True)

    with open('data/prompts.json', 'r') as f:
        prompts = json.load(f)

    system_prompt_string = prompts["phq9"]["system"]
    instruction = tg.Variable(
        system_prompt_string,
        role_description=(
            "full system prompt for PHQ-9 assessment: contains the task framing, "
            "reasoning instructions, the 9 PHQ-9 questions, scoring options, "
            "and the required answer format"
        ),
        requires_grad=True,
    )

    optimizer = tg.TGD(parameters=[instruction])
    permutation_blocks = np.random.permutation(len(tweet_blocks))
    permuted_tweet_blocks = [tweet_blocks[i] for i in permutation_blocks]
    permuted_true_answers = [true_answers[i] for i in permutation_blocks]

    for i, (tweet_block, true_answer) in enumerate(zip(permuted_tweet_blocks, permuted_true_answers)):

        for j, tweet in enumerate(tweet_block):
           print(f"Tweet {j+1}: {tweet}")
        print("--------------------------------")
        user_string = prompts["phq9"]["user_template_forced"].format(
            tweets_block="\n".join(tweet_block)
        )

        question = tg.Variable(
            user_string,
            role_description="user message with the patient's tweet history to assess",
            requires_grad=False,
        )

        prediction_val, instruction = prompt_optimizer(
            true_answer, engine, optimizer, instruction, question, i
        )
        predicted_score = parse_phq9_answers(prediction_val)
        print(f"\n\n[{i+1}/{len(tweet_blocks)}] true_phq9={true_answer}  predicted={predicted_score}")

    # Write the optimised prompt next to the input file
    output_dir = os.path.dirname(file_path)
    output_path = os.path.join(output_dir, "optimized_prompt.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(instruction.value)
    print(f"Optimised prompt written to {output_path}")

    return instruction

def prompt_optimizer(true_answer, 
                    engine, 
                    optimizer, 
                    instruction,
                    question, i=0):

    model = tg.BlackboxLLM(engine, system_prompt=instruction)
    prediction = model(question)
    total_score = parse_phq9_answers(prediction.value)

    error = total_score - true_answer
    if error > 0:
        direction = f"overestimated by {error} points"
    elif error < 0:
        direction = f"underestimated by {abs(error)} points"
    else:
        direction = "matched exactly"

    loss_fn = tg.TextLoss(f"""You are a clinical supervisor reviewing a PHQ-9 assessment produced by a student LLM.

        The true PHQ-9 sumscore is {true_answer}. The student predicted {total_score} ({direction}).

        Evaluate the student's response and provide targeted feedback to improve the SYSTEM PROMPT that guides the student. Focus on:

        1. **Per-item analysis**: For each PHQ-9 dimension (Q1-Q9), assess whether the student's score seems justified by the tweet content. Which specific items were likely over- or under-scored?
        2. **Reasoning gaps**: What patterns in the tweets did the student likely miss or misinterpret? For example: did it confuse cultural expression with distress? Did it ignore subtle cues of sleep disruption, appetite change, or concentration issues?
        3. **Calibration**: Is the student systematically biased (e.g., always scoring low because tweets don't explicitly mention symptoms, or scoring high because of emotional language that isn't truly pathological)?
        4. **Instruction improvement**: What specific changes to the system prompt's reasoning instructions would help the student better calibrate its PHQ-9 inference from tweet histories?

        Be specific and actionable. Focus on what the system prompt should tell the LLM to do differently.""")

    loss = loss_fn(prediction)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f"\n\nUPDATED INSTRUCTION: {instruction.value}")

    return prediction.value, instruction

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

# call_optimizer(
#     "data/test/meta-llama_Llama-3.1-8B-Instruct/temp_0.8_top_p_0.6_cp_10/seed_61/tweets_with_phq9.txt"
# )

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

def drop_high_phq9(tweet_blocks, true_answers):
    """
    Drop the tweets with a PHQ-9 score greater than 15.
    """
    filter_tweets = [true_answer <= 21 for true_answer in true_answers]
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
        window_centroid = torch.cat([embeddings.mean(dim=0), embeddings.max(dim=0)[0]], dim=0)
        centroids.append(window_centroid)
    centroids = torch.stack(centroids).to(device)
    return centroids

def train_BERT_model(embeddings_path, base_model_name, device, mental_bert: bool = False):
    """
    Train the BERT model for the PHQ-9 assessment.
    """
    
    
    # load BERT model
    # bert_model = generate_sbert_model(mentalbert=True)
    # bert_model.to(device)

    with open(embeddings_path, "rb") as f:
        data = torch.load(f)
        train_embs = data["train_embs"]
        val_embs = data["val_embs"]
        test_embs = data["test_embs"]
        train_labels = data["train_labels"]
        val_labels = data["val_labels"]
        test_labels = data["test_labels"]

    # create embeddings
    train_embs, train_labels = drop_high_phq9(train_embs.to(device),train_labels.to(device)) 
    val_embs, val_labels = drop_high_phq9(val_embs.to(device),val_labels.to(device))
    test_embs, test_labels = drop_high_phq9(test_embs.to(device),test_labels.to(device))
   
    print(f"Training blocks: {len(train_embs)}")

    nn_model = neural_net_BERT(mentalbert=mental_bert).to(device)
    nn_model, best_loss, epoch_history = train_bert(nn_model, train_embs, train_labels, val_embs, val_labels, device)
    print("Testing model....")
    test_loss = evaluate_bert(nn_model, test_embs, test_labels, device)
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
            input_size = 768*2
        else:
            input_size = 384*2
        super(neural_net_BERT, self).__init__()
        self.model = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 1), # 1 output for the PHQ-9 score
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)
    
def train_bert(model, train_data, train_labels, val_data, val_labels, device, epochs=50, batch_size=128, learning_rate=0.001):
    """
    Train the neural network for the PHQ-9 assessment.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)

    criterion = nn.MSELoss()
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
            best_model_state = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_state)
    return model, best_loss, epoch_history

def evaluate_bert(model, test_data, test_labels, device):
    """
    Evaluate the BERT model for the PHQ-9 assessment.
    """
    model.eval()
    criterion = nn.MSELoss()

    test_data = torch.as_tensor(test_data, dtype=torch.float32).to(device)
    test_labels = torch.as_tensor(test_labels, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(test_data).squeeze(-1)
        loss = criterion(outputs, test_labels)
    return loss.item()

def save_embeddings_for_file(file_path: str, base_model_name: str, device, mentalbert: bool = False, out_dir = None):
    """
    - Parses tweets_with_phq9.txt into train/val/test blocks
    - Encodes each block with SBERT
    - Saves embeddings + labels to disk
    """

    tweet_blocks, true_answers = create_dataset(file_path)
    train_blocks, val_blocks, test_blocks = tweet_blocks
    train_labels, val_labels, test_labels = true_answers

    sbert_model = generate_sbert_model(mentalbert=mentalbert).to(device)

    print("Encoding train blocks...")
    train_embs = setup_BERT_model(train_blocks, sbert_model, device)  # (N_train, 384)
    print("Encoding val blocks...")
    val_embs = setup_BERT_model(val_blocks, sbert_model, device)      # (N_val, 384)
    print("Encoding test blocks...")
    test_embs = setup_BERT_model(test_blocks, sbert_model, device)    # (N_test, 384)

    if mentalbert:
        dir_name = "mentalbert_embeddings"
    else:
        dir_name = "sbert_embeddings"

    if out_dir is None:
        out_dir = os.path.join("data", "test", base_model_name, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    torch_path = os.path.join(out_dir, "embeddings_and_labels.pt")
    torch.save({
    "train_embs": train_embs,
    "val_embs": val_embs,
    "test_embs": test_embs,
    "train_labels": torch.tensor(train_labels),
    "val_labels": torch.tensor(val_labels),
    "test_labels": torch.tensor(test_labels),
        }, torch_path)

    print(f"Saved embeddings + labels to {torch_path}")
    


if __name__ == "__main__":
    create_new_embeddings = False
    mental_bert = True
    file_path = "data/test/meta-llama_Llama-3.1-8B-Instruct/temp_0.8_top_p_0.6_cp_10/seed_71/tweets_with_phq9.txt"
    base_model_name = "meta-llama_Llama-3.1-8B-Instruct"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if create_new_embeddings:
        save_embeddings_for_file(file_path, base_model_name, device, mentalbert=mental_bert)
    
    if mental_bert:
        dir_name = "mentalbert_embeddings"
    else:
        dir_name = "sbert_embeddings"

    embeddings_path = os.path.join("data", "test", base_model_name, dir_name, "embeddings_and_labels.pt")

    train_BERT_model(embeddings_path, base_model_name, device, mental_bert=mental_bert)