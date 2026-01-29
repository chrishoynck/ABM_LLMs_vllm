import re, csv, json, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
from sentence_transformers import models
from sentence_transformers import SentenceTransformer as sbert

def print_histories(network, file_dir, file_name, save=False):
    """
    Parses and prints the tweet history for every agent in a readable format.
    
    Args:
        network: The network object containing agents.
        path (str): Directory path to save the output file.
    """
    output_lines = []
    output_lines.append(f"{'='*25} AGENT TWEET HISTORIES {'='*25}")

    for agent in network.all_agents:
        # Extract meaningful tweets
        valid_tweets = []
        for round_idx, entry in enumerate(agent.tweethistory):
            if "TWEET:" in entry:
                # Split on the first occurrence of "TWEET:" to handle colons in the tweet text safely
                clean_text = entry.split("TWEET:", 1)[1].strip()
                valid_tweets.append((round_idx, clean_text))
        
        # Only add agents who actually tweeted
        if valid_tweets:
            header = f"\n🔹 Agent {agent.ID} (Tweeted {len(valid_tweets)} times, phq9: {agent.well_being.get('phq9_sumscore')})"
            output_lines.append(header)
            for r_idx, text in valid_tweets:
                output_lines.append(f"   [Round {r_idx}]: \"{text}\"")
        else:
             output_lines.append(f"\n🔸 Agent {agent.ID} (Silent throughout simulation),  phq9: {agent.well_being.get('phq9_sumscore')}")

    output_lines.append(f"\n{'='*60}")
    
    # Join all lines into a single string
    final_output = "\n".join(output_lines)
    
    # Print to console
    print(final_output)

    if save:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        filename = f"tweet_histories_{file_name}"
        export_file = os.path.join(file_dir, filename)
        with open(export_file, "w", encoding="utf-8") as f:
            f.write(final_output)
        print(f"\n[Info] Tweet history saved to: {export_file}")
    
def degree_weighted_mean(network): 
    """
    Calculate the degree-weighted mean of agents phq9_sumscore at a given round.
    
    Args:
        network: The network object containing agents.
        round (int): The round number to evaluate.
    Returns:
        float: The degree-weighted mean phq9_sumscore.
    """
    
    total_degree = len(network.connections) 
    phq9_per_round = []

    min_rounds = min(len(agent.all_phq9_sumscores) for agent in network.all_agents)


    for roundje in range(min(min_rounds, network.iterations)):
        total_weighted_score = 0.0
        for agent in network.all_agents:
            degree = len(agent.agent_connections)
            phq9_score = agent.all_phq9_sumscores[roundje]
            total_weighted_score += phq9_score * degree

        if total_degree == 0:
            phq9_per_round.append(0.0)  # Avoid division by zero
        else:
            phq9_per_round.append(total_weighted_score / total_degree)
    return np.array(phq9_per_round)

def load_ngrams_tsv(filepath: str, skip_header=True) -> set:
    """
    Load distorted-language n-grams from a TSV file with columns:
      categories | markers | variants
    
    - markers column: base n-gram
    - variants column: JSON list of variant forms (optional)
    
    Returns a set of lowercased n-grams (base + all variants).
    """
    ngrams = set()
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        if skip_header:
            next(reader, None)  # skip header row
        for row in reader:
            if len(row) < 2:
                continue
            # column 1: base marker
            base = row[1].strip().lower()
            if base:
                ngrams.add(base)
            # column 2: variants (may be empty or JSON list)
            if len(row) > 2 and row[2].strip():
                variants_str = row[2].strip()
                try:
                    variants = json.loads(variants_str)
                    if isinstance(variants, list):
                        for v in variants:
                            clean = v.strip().lower()
                            if clean:
                                ngrams.add(clean)
                except json.JSONDecodeError:
                    # if not valid JSON, treat as plain text (single variant)
                    clean = variants_str.lower()
                    if clean:
                        ngrams.add(clean)
    return ngrams

def contains_ngram(text: str, ngrams: set) -> bool:
    """Check if any n-gram from ngrams appears in text (case-insensitive, word boundaries)."""
    text_low = text.lower()
    for ng in ngrams:
        # use word-boundary regex so "cat" doesn't match inside "catch"
        if re.search(r'\b' + re.escape(ng) + r'\b', text_low):
            return True
    return False

def analyze_distorted_language(network, ngrams_file: str, ngrams = None, n: int = 5, skip_header= True):
    """
    For each agent in the network, count distorted-language n-grams in:
      - the first N tweets
      - the last N tweets
    Prints a summary and returns results as a dict.
    
    Args:
        network: The network object (must have .all_agents attribute).
        ngrams_file (str): Path to the TSV file with distorted-language n-grams.
        n (int): Number of tweets from the start/end to analyze.
    
    Returns:
        dict: {agent_id: {"first_n": count, "last_n": count, "total_tweets": int}}
    """
    if ngrams is None:
        ngrams = load_ngrams_tsv(ngrams_file, skip_header=skip_header)
    # print(ngrams)
    print(f"Loaded {len(ngrams)} distorted-language n-grams from {ngrams_file}")
    highest_frac = 0

    results = {}
    for agent in network.all_agents:
        history = getattr(agent, "tweethistory", [])
 
        # now only consider actual tweets
        history = [t for t in history if t!= "NO_TWEET"]
        first_tweets = history[:n]
        last_tweets = history[-n:] if len(history) >= n else history

        first_tweets = [t for t in first_tweets if t != "NO_TWEET"]
        last_tweets = [t for t in last_tweets if t != "NO_TWEET"]
        
        first_count = sum(1 for tweet in first_tweets if contains_ngram(tweet, ngrams))
        last_count = sum(1 for tweet in last_tweets if contains_ngram(tweet, ngrams))
        
        results[agent.ID] = {
            "first_n": first_count,
            "last_n": last_count,
            "Length_last_tweets": len(last_tweets),
            "Length_first_tweets": len(first_tweets),
            "total_tweets": len(history),
            "frac_distorted_first": first_count / len(first_tweets) if len(first_tweets)>0 else 0,
            "frac_distorted_last": last_count / len(last_tweets) if len(last_tweets)>0 else 0,
        }
        highest_frac = max(highest_frac, results[agent.ID]["frac_distorted_last"])
        highest_frac = max(highest_frac, results[agent.ID]["frac_distorted_first"])
    return results, highest_frac

#=========================SBERT functions=========================

def generate_sbert_model(model_name="all-MiniLM-L6-v2", mentalbert=False):
    """Generates and returns a SBERT model for embedding sentences.
    
    Args:
        model_name (str): Name of the pre-trained SBERT model to load.
        mentalbert (bool): Whether to use a custom MentalBERT architecture.
            (use "mental/mental-bert-base-uncased" as model_name in that case)
    
    Returns:
        SentenceTransformer: The loaded SBERT model.
    """

    if mentalbert:
        word_embedding_model = models.Transformer("mental/mental-bert-base-uncased")
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        
        model = sbert(modules=[word_embedding_model, pooling_model])
    else:
        model = sbert(model_name)
    return model

def create_embedding(model, texts):
    """Generates embeddings for a list of texts using the provided SBERT model.
    
    Args:
        model (SentenceTransformer): The SBERT model to use for embedding.
        texts (list of str): List of texts to embed.
    Returns:
        Tensor: The generated embeddings.
    """
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings


def network_list_w_slices(networks_per_setting: dict):
    """Flattens networks_per_setting dict into a single list and records slices.
    
    Args:
        networks_per_setting (dict): {setting_name: [list_of_networks]}
    Returns:
        all_networks (list): Flattened list of all networks.
        setting_slices (dict): {setting_name: (start_index, end_index)}
    """
    all_networks = []
    setting_slices = {}
    start_index = 0
    
    # Here, we flatten all networks into a single list and keep track of slices per setting (group runs)
    for setting, networks in networks_per_setting.items():
        all_networks.extend(networks)
        end_index = start_index + len(networks)
        setting_slices[setting] = (start_index, end_index)
        start_index = end_index
    return all_networks, setting_slices
    

def mean_sbert_per_networks(model, all_networks, num_steps=30, shift=5):    
    """Computes SBERT embeddings using Mean Pooling over time windows for a list of networks.
    Args:
        model: Loaded SentenceTransformer model
        all_networks: List of network objects.
        num_steps: Size of the sliding window (number of tweets)
        shift: Stride of the sliding window
    Returns:
        global_sbert_means: List of (Time, 384) arrays for each network
        global_sbert_vars: List of (Time, 384) arrays for each network
    """

    global_sbert_means = []
    global_sbert_vars = []

    embedding_dim = model.get_sentence_embedding_dimension()
    
    print(f"Starting embedding for {len(all_networks)} networks...")


    # Here, we embed tweets in windows and mean-pool them, for every network. 
    for i, network in enumerate(all_networks):

        # Calculate number of windows based on total iterations
        net = network["network"]
        max_iters = net.iterations
        num_windows = max(1, (max_iters - num_steps) // shift + 1)
        
        # collect one trajectory per agent, then average them to get the "Network Trajectory"
        mean_net_window_vectors = [] 
        var_net_window_vectors = []

        for w in range(num_windows):
            start_t = w * shift
            end_t = start_t + num_steps
            
            # Collect all valid tweets from all agents in this specific time window
            tweets_in_window = []
            for agent in net.all_agents:
                # Safety check for history length
                hist = agent.tweethistory
                if len(hist) > start_t:
                    # Slice robustly
                    slice_end = min(len(hist), end_t)
                    window_segment = hist[start_t : slice_end]
                    
                    # Filter out NO_TWEET
                    valid_tweets = [t for t in window_segment if t != "NO_TWEET"]
                    tweets_in_window.extend(valid_tweets)
            
            # Embed and Average (Mean Pooling)
            if not tweets_in_window:
                # Handle empty window (rare, but possible if no one tweets)
                # Use a zero vector as a placeholder
                window_centroid = np.zeros(embedding_dim)
                window_variance = np.zeros(embedding_dim)

            else:
                # Embed all tweets in this window individually
                embeddings = model.encode(tweets_in_window, batch_size=64, show_progress_bar=True)
                
                # Calculate the Mean Vector (Centroid) for this window
                window_centroid = np.mean(embeddings, axis=0)
                window_variance = np.var(embeddings, axis=0)
            
            mean_net_window_vectors.append(window_centroid)
            var_net_window_vectors.append(window_variance)
        
        # Stack to create (Time, 384) matrix for this specific network run
        global_sbert_means.append(np.stack(mean_net_window_vectors))
        global_sbert_vars.append(np.stack(var_net_window_vectors))


    return global_sbert_means, global_sbert_vars


def sbert_for_runs(networks_per_setting: dict, num_steps=30, shift=5, mentalbert=True):
    """
    Computes SBERT embeddings using Mean Pooling over time windows.
    
    Args:
        networks_per_setting: Dictionary of {setting_name: [list_of_networks]}
        model: Loaded SentenceTransformer model
        num_steps: Size of the sliding window (number of tweets)
        shift: Stride of the sliding window
        
    Returns:
        mean_sbert_per_setting: {setting: (Time, 384) array} -> Average trajectory
        all_mats_per_setting:   {setting: List of (Time, 384) arrays} -> Individual runs
    """
    # Flatten networks and get slices
    all_networks, setting_slices = network_list_w_slices(networks_per_setting)

    # Compute SBERT embeddings for all networks
    model = generate_sbert_model(mentalbert=mentalbert)
    global_sbert_means, global_sbert_vars = mean_sbert_per_networks(model, all_networks, num_steps=num_steps, shift=shift)

    # Group by setting and compute mean trajectories
    mean_sbert_per_setting = {}
    var_sbert_per_setting = {}

    all_mats_per_setting = {}
    
    for setting, (start, end) in setting_slices.items():
        matrices_over_runs = global_sbert_means[start:end]
        matrices_vars_over_runs = global_sbert_vars[start:end]
        
        if not matrices_over_runs:
            continue

        # Trim to minimum length (in case some runs stopped early)
        min_len = min(m.shape[0] for m in matrices_over_runs)
        cut_short_means = [m[:min_len] for m in matrices_over_runs]
        cut_short_vars = [m[:min_len] for m in matrices_vars_over_runs]
        
        # Stack: (Num_Runs, time_window, Embedding_Dim)
        embedding_per_setting = np.stack(cut_short_means, axis=0)
        embedding_vars_per_setting = np.stack(cut_short_vars, axis=0)

        # Calculate Mean Trajectory over the runs
        mean_sbert_per_setting[setting] = np.mean(embedding_per_setting, axis=0)
        var_sbert_per_setting[setting] = np.mean(embedding_vars_per_setting, axis=0)
        
        if setting not in all_mats_per_setting:
            all_mats_per_setting[setting] = []

        all_mats_per_setting[setting].extend(cut_short_means)

    return mean_sbert_per_setting, var_sbert_per_setting, all_mats_per_setting


#=========================TF-IDF functions=========================

def retrieve_windowed_data(networks_data, num_steps= 30, shift=5, n_grams= None):
    '''Retrieve TF-IDF data from the network's agents' tweet histories.
    Args:
        network_data: Tuple containing the network objects.
        num_steps (int): Number of steps used in TF-IDF retrieval.
        shift (int): Shift between windows.
    Returns:
        all_tweets_extracted (List(str)): All tweets extracted for TF-IDF fitting.
        docs_per_network (List(List(str))): List of documents (one per time window) per network.
    '''
    all_tweets_extracted = []
    docs_per_network = []
  
    for i,  network_data in enumerate(networks_data):
        network = network_data["network"]
        # calculate number of windows
        num_windows = max(1, (network.iterations - num_steps) // shift + 1)
        window_list = [[] for _ in range(num_windows)]
        for agent in network.all_agents:

            # iterate over windows
            for w in range(num_windows):
                # extend with tweets in this window
                window_list[w].extend(agent.tweethistory[(w* shift):(w*shift + num_steps)])

            # extend with remaining tweets
            if (num_windows - 1) * shift + num_steps < network.iterations:
                if len(window_list) < num_windows + 1:
                    window_list.append([])
                window_list[-1].extend(agent.tweethistory[((num_windows-1) * shift + num_steps):])
            
            # also keep a vocab of all tweets
            all_tweets_extracted.extend(agent.tweethistory)
            if n_grams is not None and i == 0:
                all_tweets_extracted.extend(n_grams)
    
        filtered_window_texts = []
        w = 0
        while w < len(window_list):
            window_list[w] = [t for t in window_list[w] if t != "NO_TWEET"]
            window_list[w] = " ".join(window_list[w])
            # remove empty tweet lists
            if window_list[w] == "":
                print("WARNING: Empty tweet list for window ", w)
                window_list.pop(w)
            else:
                filtered_window_texts.append(window_list[w])
                w+=1

        docs_per_network.append(filtered_window_texts)
    
    all_tweets_extracted = [t for t in all_tweets_extracted if t!= "NO_TWEET"]

    if len(all_tweets_extracted) == 0:
        raise ValueError("No valid tweets found in the network for TF-IDF computation.")
    return all_tweets_extracted, docs_per_network

# TF-IDF computation
def compute_tf_idf(all_tweets):
    '''Compute TF-IDF for a list of tweets.
    Args:
        all_tweets (List(str)): List of tweet texts.
    Returns:
        vocab (np.array): Vocabulary array. 
        vectorizer: Fitted TfidfVectorizer object.
    '''

    # remove english common words
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

    # fit the model
    vectorizer.fit(all_tweets)

    # get vocabulary
    vocab = np.array(vectorizer.get_feature_names_out())
    return  vocab, vectorizer

def retrieve_tf_idf(networks, num_steps= 30, shift=5, n_grams= None):
    all_tweets_extracted, docs_per_network = retrieve_windowed_data(networks, num_steps=num_steps, shift=shift, n_grams=n_grams)

    # test with n-grams added to vocab
    print("using n-gram as vocab in TF-IDF: ")
    all_tweets_extracted = n_grams
    vocab, vectorizer = compute_tf_idf(all_tweets_extracted)

    global_tf_idf = [vectorizer.transform(doc).toarray() for doc in docs_per_network]

    # global_tf_idf = global_tf_idf.toarray()
    return global_tf_idf, vocab, vectorizer


def tf_idf_for_runs(networks_per_setting: dict, num_steps=30, shift=5, n_grams=None):
    '''Compute TF-IDF matrices for multiple network runs.
    Args:
        networks: List of network objects.
        num_steps (int): Number of steps used in TF-IDF retrieval.
        shift (int): Shift between windows.
    Returns:
        meanvar_tf_idf_per_setting: {setting: mean_tf_idf_matrix}
        all_mats_per_setting: {setting: List of tf_idf_matrices}
    '''
    # flatten networks and get slices
    all_networks, setting_slices = network_list_w_slices(networks_per_setting)

    tf_idf_matrices, vocab, vectorizer = retrieve_tf_idf(
        all_networks, num_steps=num_steps, shift=shift, n_grams=n_grams)
    
    mean_tf_idf_per_setting = {}
    all_mats_per_setting = {}
    for setting, (start, end) in setting_slices.items():

        # returns tf-idf matrices for runs in this setting
        matrixjes_over_runs = tf_idf_matrices[start:end]

        # makes sure all matrices have the same length (number of time windows)
        min_length = min(m.shape[0] for m in matrixjes_over_runs)
        trimmed_matrices = [m[:min_length] for m in matrixjes_over_runs]
        print("mim_length for setting ", setting, ": ", min_length)

        # compute mean tf-idf over runs
        stacked_matrices = np.stack(trimmed_matrices, axis=0)
        mean_tf_idf = np.mean(stacked_matrices, axis=0)

        mean_tf_idf_per_setting[setting] = mean_tf_idf
        if setting not in all_mats_per_setting:
            all_mats_per_setting[setting] = []
        all_mats_per_setting[setting].extend(trimmed_matrices)

    return mean_tf_idf_per_setting, all_mats_per_setting


#=========================PCA functions=========================
    
def reduce_dimensionality(tf_idf_matrices, n_components=2):
    '''Reduce dimensionality of TF-IDF matrix using PCA.
    Args:
        tf_idf_matrices (np.array): TF-IDF matrix.
        n_components (int): Number of PCA components.
    Returns:
        reduced_runs (np.array): PCA-reduced data.
    '''
    assert n_components <= tf_idf_matrices[0].shape[1], "n_components must be <= number of features"
    tf_idf_stacked = np.vstack(tf_idf_matrices)
    pca = PCA(n_components=n_components)
    pca.fit(tf_idf_stacked)
    reduced_runs = [pca.transform(tf_idf_matrix) for tf_idf_matrix in tf_idf_matrices]
    return reduced_runs

def pca_on_means(embedding_per_setting, n_components=2):
    '''Apply PCA on mean embedding matrices for multiple settings.
    Args:
        mean_tf_idf_per_setting (dict): {setting: mean_tf_idf_matrix}
        n_components (int): Number of PCA components.
    Returns:
        reduced_means (dict): {setting: PCA-reduced mean embedding matrix}
        pca: Fitted PCA object.
    '''
    settings = list(embedding_per_setting.keys())
    mean_matrices = [embedding_per_setting[setting] for setting in settings]
    embedding_stacked = np.vstack(mean_matrices)
    pca = PCA(n_components=n_components)
    pca.fit(embedding_stacked)

    mean_traj = {
        s: pca.transform(embedding_per_setting[s])      # (T, n_components)
        for s in settings
    }
    return mean_traj, pca

def traj_variance_in_pca_space(runs_embedding_per_setting, pca):
    """
    runs_embedding_per_setting: dict[setting] -> list[np.ndarray] each (T, V)
    pca: fitted PCA object
    Returns:
        std_traj: dict[setting] -> (T, D)
        var_traj:  dict[setting] -> (T, D)
    """
    std_traj = {}
    var_traj = {}

    for setting, run_mats in runs_embedding_per_setting.items():
        # run_mats: list of (T, V), all with same T by construction

        # project each run into PCA space
        run_trajs = [pca.transform(M) for M in run_mats]   # each (T, D)

        stacked = np.stack(run_trajs, axis=0)             # (R, T, D)
        var_traj[setting]  = stacked.var(axis=0)          # (T, D)
        std_traj[setting] = stacked.std(axis=0)           # (T, D)

    return std_traj, var_traj


# =============================Tweet frequency statistics===============================
def calculate_tweet_frequency_stats(agent_histories, window_size=5):
    """
    Calculate the mean and variance of tweet frequency over time using a sliding window.

    Args:
        agent_histories (list of list of str): List of tweet histories for each agent.
        window_size (int): The size of the sliding window.

    Returns:
        dict: A dictionary containing 'mean' and 'variance' lists over time.
    """
    if len(agent_histories) == 0:
        return {'mean': [], 'variance': []}

    num_steps = len(agent_histories[0])
    mean_freqs = []
    var_freqs = []

    for t in range(num_steps):
        # Determine the window range
        start = max(0, t - window_size + 1)
        end = t + 1
        
        freqs_at_t = []
        for history in agent_histories:
            window = history[start:end]
            if not window:
                freqs_at_t.append(0.0)
                continue
            
            tweets_count = sum(1 for tweet in window if tweet != "NO_TWEET")
            freq = tweets_count / len(window)
            freqs_at_t.append(freq)
        
        mean_freqs.append(np.mean(freqs_at_t))
        var_freqs.append(np.var(freqs_at_t))

    return {'mean': mean_freqs, 'variance': var_freqs}


def obtain_tweet_histories(networks):
    """
    Obtain tweet histories from a list of networks.

    Args:
        networks (list): List of network objects.
    Returns:
        list of list of str: List of tweet histories for each agent across all networks.
    """
    all_histories = []
    for network in networks:
        for agent in network.all_agents:
            history = getattr(agent, "tweethistory", [])
            all_histories.append(history)
    return all_histories


# =========================Critical Slowing Down analysis=========================

def calculate_agent_cd(sequence, window_size):
    """
    Calculates rolling variance and lag-1 autocorrelation using NumPy.
    """
    seq = np.array(sequence)
    n = len(seq)
    
    # Initialize arrays with NaNs (to represent the 'warm-up' period)
    variances = np.full(n, np.nan)
    autocorrs = np.full(n, np.nan)
    
    for i in range(window_size, n + 1):
        # Extract the current window
        window = seq[i - window_size : i]
        
        # Variance calculation
        variances[i-1] = np.var(window, ddof=1)
        
        # Lag-1 Autocorrelation
        if len(window) > 1:
            # Current values vs lagged values
            x = window[1:]
            y = window[:-1]
            
            corr_matrix = np.corrcoef(x, y)
            autocorrs[i-1] = corr_matrix[0, 1]
            
    return variances, autocorrs

# Example:
def all_agent__tweet_cd(network, window_size):
    """
    Calculate rolling variance and lag-1 autocorrelation for all agents in the network.
    
    Args:
        network: The network object containing agents.
        window_size (int): The size of the rolling window.
    
    Returns:
        dict: {agent_id: {'variance': list, 'autocorrelation': list}}
    """
    cd_results = {}
    for agent in network.all_agents:
        history = getattr(agent, "tweethistory", [])
        # Convert tweet history to binary sequence (1 if tweeted, 0 if NO_TWEET)
        binary_sequence = [1 if tweet != "NO_TWEET" else 0 for tweet in history]
        
        variances, autocorrs = calculate_agent_cd(binary_sequence, window_size)
        
        cd_results[agent.ID] = {
            'variance': variances.tolist(),
            'autocorrelation': autocorrs.tolist()
        }
    return cd_results

def all_agent_phq9_cd(network, window_size):
    """
    Calculate rolling variance and lag-1 autocorrelation for agents' PHQ-9 scores.
    
    Args:
        network: The network object containing agents.
        window_size (int): The size of the rolling window.
    Returns:
        dict: {agent_id: {'variance': list, 'autocorrelation': list}}
    """
    cd_results = {}
    for agent in network.all_agents:
        phq9_scores = agent.all_phq9_sumscores
        
        variances, autocorrs = calculate_agent_cd(phq9_scores, window_size)
        
        cd_results[agent.ID] = {
            'variance': variances.tolist(),
            'autocorrelation': autocorrs.tolist()
        }
    return cd_results