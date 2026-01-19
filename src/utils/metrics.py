import re, csv, json, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np

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
    for roundje in range(network.iterations):
        total_weighted_score = 0.0
        for agent in network.all_agents:
            degree = len(agent.agent_connections)
            phq9_score = agent.all_phq9_sumscores[roundje]
            total_weighted_score += phq9_score * degree

        if total_degree == 0:
            phq9_per_round.append(0.0)  # Avoid division by zero

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
    '''Retrieve TF-IDF data from the network's agents' tweet histories.
    Args:
        networks: The network objects.
        num_steps (int): Number of steps used in TF-IDF retrieval.
        shift (int): Shift between windows.
    Returns:
        global_tf_idf (np.array): TF-IDF matrix for all windows.
        vocab (np.array): Vocabulary array.
        vectorizer: Fitted TfidfVectorizer object.
    '''
    tf_idf_all = []
    docs_per_network = []
  
    for i,  network in enumerate(networks):
        # calculate number of windows
        num_windows = max(1, (network.iterations - num_steps) // shift + 1)
        tf_idf_lists = [[] for _ in range(num_windows)]
        for agent in network.all_agents:

            # iterate over windows
            for w in range(num_windows):
                # extend with tweets in this window
                tf_idf_lists[w].extend(agent.tweethistory[(w* shift):(w*shift + num_steps)])

            # extend with remaining tweets
            if (num_windows - 1) * shift + num_steps < network.iterations:
                if len(tf_idf_lists) < num_windows + 1:
                    tf_idf_lists.append([])
                tf_idf_lists[-1].extend(agent.tweethistory[((num_windows-1) * shift + num_steps):])
            
            # also keep a vocab of all tweets
            tf_idf_all.extend(agent.tweethistory)
            if n_grams is not None and i == 0:
                tf_idf_all.extend(n_grams)
    
        cleaned_tf_idf = []
        w = 0
        while w < len(tf_idf_lists):
            tf_idf_lists[w] = [t for t in tf_idf_lists[w] if t != "NO_TWEET"]
            tf_idf_lists[w] = " ".join(tf_idf_lists[w])
            # remove empty tweet lists
            if tf_idf_lists[w] == "":
                print("WARNING: Empty tweet list for window ", w)
                tf_idf_lists.pop(w)
            else:
                cleaned_tf_idf.append(tf_idf_lists[w])
                w+=1

        docs_per_network.append(cleaned_tf_idf)
    
    tf_idf_all = [t for t in tf_idf_all if t!= "NO_TWEET"]

    if len(tf_idf_all) == 0:
        raise ValueError("No valid tweets found in the network for TF-IDF computation.")
    vocab, vectorizer = compute_tf_idf(tf_idf_all)

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
        tf_idf_matrices (List(np.array)): List of TF-IDF matrices for each run.
        vocab (np.array): Vocabulary array.
        vectorizer: Fitted TfidfVectorizer object.
    '''
    all_networks = []
    setting_slices = {}
    start_index = 0
    for setting, networks in networks_per_setting.items():
        all_networks.extend(networks)
        end_index = start_index + len(networks)

        # record slice for this setting
        setting_slices[setting] = (start_index, end_index)
        start_index = end_index

    tf_idf_matrices, vocab, vectorizer = retrieve_tf_idf(
        all_networks, num_steps=num_steps, shift=shift, n_grams=n_grams)
    
    meanvar_tf_idf_per_setting = {}
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

        meanvar_tf_idf_per_setting[setting] = mean_tf_idf
        if setting not in all_mats_per_setting:
            all_mats_per_setting[setting] = []
        all_mats_per_setting[setting].extend(trimmed_matrices)

    return meanvar_tf_idf_per_setting, all_mats_per_setting

    

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

def pca_on_means(mean_tf_idf_per_setting, n_components=2):
    '''Apply PCA on mean TF-IDF matrices for multiple settings.
    Args:
        mean_tf_idf_per_setting (dict): {setting: mean_tf_idf_matrix}
        n_components (int): Number of PCA components.
    Returns:
        reduced_means (dict): {setting: PCA-reduced mean TF-IDF matrix}
        pca: Fitted PCA object.
    '''
    settings = list(mean_tf_idf_per_setting.keys())
    mean_matrices = [mean_tf_idf_per_setting[setting] for setting in settings]
    tf_idf_stacked = np.vstack(mean_matrices)
    pca = PCA(n_components=n_components)
    pca.fit(tf_idf_stacked)

    mean_traj = {
        s: pca.transform(mean_tf_idf_per_setting[s])      # (T, n_components)
        for s in settings
    }
    return mean_traj, pca

def traj_variance_in_pca_space(runs_tf_idf_per_setting, pca):
    """
    runs_tf_idf_per_setting: dict[setting] -> list[np.ndarray] each (T, V)
    pca: fitted PCA object
    Returns:
        std_traj: dict[setting] -> (T, D)
        var_traj:  dict[setting] -> (T, D)
    """
    std_traj = {}
    var_traj = {}

    for setting, run_mats in runs_tf_idf_per_setting.items():
        # run_mats: list of (T, V), all with same T by construction

        # project each run into PCA space
        run_trajs = [pca.transform(M) for M in run_mats]   # each (T, D)

        stacked = np.stack(run_trajs, axis=0)             # (R, T, D)
        var_traj[setting]  = stacked.var(axis=0)          # (T, D)
        std_traj[setting] = stacked.std(axis=0)           # (T, D)

    return std_traj, var_traj

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
