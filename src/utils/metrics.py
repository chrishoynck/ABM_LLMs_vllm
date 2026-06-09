import re, csv, json, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
import networkx as nx
from sentence_transformers import models
from sentence_transformers import SentenceTransformer as sbert
import umap
from .tools.format_config import FC

def print_histories(network, file_dir, file_name, save=False):
    """
    Parses and prints the tweet history for every agent in a readable format.
    
    Args:
        network: The network object containing agents.
        path (str): Directory path to save the output file.
    """
    output_lines = []
    output_lines.append(f"{'='*25} AGENT {FC.Label.upper()} HISTORIES {'='*25}")

    for agent in network.all_agents:
        # Extract meaningful tweets
        valid_tweets = []
        for round_idx, entry in enumerate(agent.tweethistory):
            if FC.CONTENT_PREFIX in entry:
                clean_text = entry.split(FC.CONTENT_PREFIX, 1)[1].strip()
                valid_tweets.append((round_idx, clean_text))
        
        # Only add agents who actually tweeted
        if valid_tweets:
            header = f"\n🔹 Agent {agent.ID} ({FC.Label}ed {len(valid_tweets)} times, phq9: {agent.well_being.get('phq9_sumscore')}, persona: {agent.persona})"
            output_lines.append(header)
            for r_idx, text in valid_tweets:
                output_lines.append(f"   [Round {r_idx}]: \"{text}\"")
        else:
             output_lines.append(f"\n🔸 Agent {agent.ID} (Silent throughout simulation),  phq9: {agent.well_being.get('phq9_sumscore')}, persona: {agent.persona}")

    output_lines.append(f"\n{'='*60}")
    
    # Join all lines into a single string
    final_output = "\n".join(output_lines)
    
    # Print to console
    if not save:
        print(final_output)

    if save:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        filename = file_name.replace("json", "txt")
        filename = f"tweet_histories_{filename}"
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
        history = [t for t in history if t!= FC.NO_CONTENT]
        first_tweets = history[:n]
        last_tweets = history[-n:] if len(history) >= n else history

        first_tweets = [t for t in first_tweets if t != FC.NO_CONTENT]
        last_tweets = [t for t in last_tweets if t != FC.NO_CONTENT]
        
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


# =========================CDS parsing from neighbor_history=========================

def neighbor_cds_records(agent, ngrams):
    """Re-derive CDS / neighbour-PHQ-9 stats per round from ``agent.neighbor_history``.

    Each ``neighbor_history`` entry stores the agent's own committed tweet/PHQ-9
    plus the full tweets/IDs/PHQ-9 of every activated neighbour that round (see
    ``Agent.commit``). This recomputes, per round, the CDS signals that used to
    live only as the on-the-fly ``frac_distorted_neigh`` / ``network.cds_info``:

        - ``frac_neigh_cds``  : fraction of activated neighbours whose tweet
          contains a distorted-language n-gram (== the old frac_distorted_neigh).
        - ``distorted``       : whether the agent's own tweet contains CDS — the
          per-(agent, round) event behind "probability of sending CDS".
        - ``mean_neigh_phq9`` : mean PHQ-9 of activated neighbours, for studying
          the influence of neighbour well-being on the agent's language use.

    Recomputing from the raw saved tweets (rather than trusting a stored flag)
    lets you swap in a different CDS detector / n-gram set after the fact.

    Args:
        agent: an Agent with a populated ``neighbor_history``.
        ngrams (set): distorted-language n-grams, e.g. from ``load_ngrams_tsv``.

    Returns:
        list[dict]: one dict per round, in chronological order.
    """
    out = []
    for rec in agent.neighbor_history:
        neighs = rec.get("neighbors", [])
        n_neigh = len(neighs)
        n_cds = sum(1 for nb in neighs if contains_ngram(nb.get("tweet") or "", ngrams))
        phq9s = [nb["phq9"] for nb in neighs if nb.get("phq9") is not None]
        out.append({
            "round": rec.get("round"),
            "phq9": rec.get("phq9"),
            "activated": bool(rec.get("activated", False)),
            "distorted": bool(rec.get("distorted", False)),
            "tweet": rec.get("tweet"),
            "frac_neigh_cds": (n_cds / n_neigh) if n_neigh else 0.0,
            "n_neighbors": n_neigh,
            "mean_neigh_phq9": (float(np.mean(phq9s)) if phq9s else None),
        })
    return out


def cds_info_from_neighbor_history(network, ngrams):
    """Reproduce ``network.cds_info`` from per-agent ``neighbor_history``.

    Returns a list of ``(frac_neigh_cds, activated, distorted)`` tuples in the
    same round-major / ``all_agents`` order that
    ``_Network._apply_outputs_and_update_state`` appends to ``network.cds_info``
    — so it can be diffed against the live ``cds_info`` as a correctness check,
    or fed directly to ``visualization.distorted_info``.
    """
    if not network.all_agents:
        return []
    n_rounds = min(len(a.neighbor_history) for a in network.all_agents)
    parsed = {a.ID: neighbor_cds_records(a, ngrams) for a in network.all_agents}
    out = []
    for r in range(n_rounds):
        for agent in network.all_agents:
            rec = parsed[agent.ID][r]
            out.append((rec["frac_neigh_cds"], rec["activated"], rec["distorted"]))
    return out


#=========================SBERT functions=========================

def generate_sbert_model(model_name="all-MiniLM-L6-v2", mentalbert=False, device=None):
    """Generates and returns a SBERT model for embedding sentences.

    Args:
        model_name (str): Name of the pre-trained SBERT model to load.
        mentalbert (bool): Whether to use a custom MentalBERT architecture.
            (use "mental/mental-bert-base-uncased" as model_name in that case)
        device: torch device or string to place the model on (e.g. "cpu", "cuda").
            Defaults to CUDA if available when None.

    Returns:
        SentenceTransformer: The loaded SBERT model.
    """

    if mentalbert:
        word_embedding_model = models.Transformer("mental/mental-bert-base-uncased")
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        model = sbert(modules=[word_embedding_model, pooling_model], device=device)
    else:
        model = sbert(model_name, device=device)
    return model

def build_network_graph(network):
    """
    Build a NetworkX graph from a network object and return it alongside
    a mapping from agent ID to index in network.all_agents.

    Returns:
        graph:      nx.DiGraph or nx.Graph (matches network.directed)
        id_to_idx:  dict mapping agent.ID -> index in network.all_agents
    """
    graph = nx.DiGraph() if network.directed else nx.Graph()
    for agent in network.all_agents:
        graph.add_node(agent.ID)
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    id_to_idx = {agent.ID: i for i, agent in enumerate(network.all_agents)}
    return graph, id_to_idx

def save_agent_embeddings(agent_embs, filepath):
    """Save agent_embs (list[list[ndarray | None]]) to a .npz file.

    None entries are stored as rows of NaN so the array stays rectangular.
    A boolean mask is saved alongside to reconstruct Nones on load.
    """
    if not agent_embs or not agent_embs[0]:
        print(f"[Warning] Empty agent_embs, nothing to save.")
        return
    n_agents = len(agent_embs)
    T = len(agent_embs[0])
    # infer embedding dim from first non-None entry
    dim = next(e.shape[0] for row in agent_embs for e in row if e is not None)

    data = np.full((n_agents, T, dim), np.nan)
    mask = np.zeros((n_agents, T), dtype=bool)  # True where valid
    for i, row in enumerate(agent_embs):
        for t, emb in enumerate(row):
            if emb is not None:
                data[i, t, :] = emb
                mask[i, t] = True

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    np.savez_compressed(filepath, embeddings=data, mask=mask)
    print(f"[Info] Saved agent embeddings to {filepath}  "
          f"(agents={n_agents}, T={T}, dim={dim})")


def load_agent_embeddings(filepath):
    """Load agent_embs from a .npz file written by save_agent_embeddings.

    Returns:
        agent_embs: list[list[ndarray | None]]  same format as build_agent_embeddings.
    """
    npz = np.load(filepath)
    data = npz["embeddings"]   # (n_agents, T, dim)
    mask = npz["mask"]         # (n_agents, T)
    n_agents, T, _ = data.shape
    agent_embs = []
    for i in range(n_agents):
        row = [data[i, t] if mask[i, t] else None for t in range(T)]
        agent_embs.append(row)
    print(f"[Info] Loaded agent embeddings from {filepath}  "
          f"(agents={n_agents}, T={T})")
    return agent_embs


def save_tweet_embeddings(tweet_to_emb, filepath):
    """Save a tweet→embedding dict to a .npz file."""
    if not tweet_to_emb:
        print("[Warning] Empty tweet_to_emb, nothing to save.")
        return
    tweets = list(tweet_to_emb.keys())
    embs = np.stack([tweet_to_emb[t] for t in tweets])
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    np.savez_compressed(filepath, tweets=np.array(tweets, dtype=object), embeddings=embs)
    print(f"[Info] Saved {len(tweets)} tweet embeddings to {filepath}")


def load_tweet_embeddings(filepath):
    """Load a tweet→embedding dict from a .npz file.

    Returns:
        tweet_to_emb: dict[str, ndarray]
    """
    npz = np.load(filepath, allow_pickle=True)
    tweets = npz["tweets"]
    embs = npz["embeddings"]
    tweet_to_emb = {str(t): embs[i] for i, t in enumerate(tweets)}
    print(f"[Info] Loaded {len(tweet_to_emb)} tweet embeddings from {filepath}")
    return tweet_to_emb


def build_tweet_embedding_cache(all_networks, mentalbert=True, cache_path=None):
    """Collect all unique tweets across networks, encode once, return tweet→emb dict.

    If cache_path exists, loads from disk. Otherwise encodes and saves.

    Args:
        all_networks: List of network dicts (each with "network" key).
        mentalbert (bool): Model choice.
        cache_path (str | None): Optional .npz path for caching.

    Returns:
        tweet_to_emb: dict[str, ndarray]
        embedding_dim: int
    """
    if cache_path is not None:
        if not cache_path.endswith(".npz"):
            cache_path += ".npz"
        if os.path.exists(cache_path):
            print(f"[Cache hit] Loading tweet embeddings from {cache_path}")
            tweet_to_emb = load_tweet_embeddings(cache_path)
            dim = next(iter(tweet_to_emb.values())).shape[0]
            return tweet_to_emb, dim

    # Collect all unique tweets across every network
    unique_tweets = set()
    for net_dict in all_networks:
        net = net_dict["network"]
        for agent in net.all_agents:
            for tweet in agent.tweethistory:
                if tweet and tweet != FC.NO_CONTENT:
                    unique_tweets.add(tweet)
    unique_tweets = list(unique_tweets)

    if not unique_tweets:
        return {}, 0

    print(f"Embedding {len(unique_tweets)} unique tweets across {len(all_networks)} networks...")
    model = generate_sbert_model(mentalbert=mentalbert)
    raw_embs = model.encode(unique_tweets, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    tweet_to_emb = dict(zip(unique_tweets, raw_embs))

    if cache_path is not None:
        save_tweet_embeddings(tweet_to_emb, cache_path)

    return tweet_to_emb, raw_embs.shape[1]


def build_agent_embeddings(network, mentalbert=True, cache_path=None):
    """
    Embed all unique tweets from a network and return a per-agent, per-timestep lookup.

    If cache_path is given and the file exists, embeddings are loaded from disk
    instead of recomputed.  If cache_path is given but the file does not exist,
    embeddings are computed and then saved to that path.

    Args:
        network: The network object containing agents.
        mentalbert (bool): Whether to use MentalBERT or default SBERT.
        cache_path (str | None): Optional .npz file path for saving / loading.

    Returns:
        agent_embs: list[list[ndarray | None]]  shape [n_agents][T]
            agent_embs[i][t] is the embedding of agent i's tweet at timestep t,
            or None if the agent posted NO_CONTENT at that step.
    """
    # ── try loading from cache ──
    if cache_path is not None:
        # ensure the extension is .npz
        if not cache_path.endswith(".npz"):
            cache_path += ".npz"
        if os.path.exists(cache_path):
            print(f"[Cache hit] Loading embeddings from {cache_path}")
            return load_agent_embeddings(cache_path)

    # ── compute embeddings ──
    T = min(len(agent.tweethistory) for agent in network.all_agents)

    unique_tweets = list({
        agent.tweethistory[t]
        for agent in network.all_agents
        for t in range(T)
        if agent.tweethistory[t] and agent.tweethistory[t] != FC.NO_CONTENT
    })
    if not unique_tweets:
        return [[] for _ in network.all_agents]

    print(f"Embedding {len(unique_tweets)} unique tweets...")
    model = generate_sbert_model(mentalbert=mentalbert)
    raw_embs = model.encode(unique_tweets, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    tweet_to_emb = dict(zip(unique_tweets, raw_embs))

    agent_embs = []
    for agent in network.all_agents:
        row = [tweet_to_emb.get(agent.tweethistory[t])
               if (agent.tweethistory[t] and agent.tweethistory[t] != FC.NO_CONTENT) else None
               for t in range(T)]
        agent_embs.append(row)

    # ── save to cache ──
    if cache_path is not None:
        save_agent_embeddings(agent_embs, cache_path)

    return agent_embs

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
    

def mean_sbert_per_networks(model, all_networks, num_steps=30, shift=5,
                            tweet_to_emb=None, embedding_dim=None):
    """Computes SBERT embeddings using Mean Pooling over time windows for a list of networks.

    If tweet_to_emb is provided, embeddings are looked up from the dict
    instead of calling model.encode (much faster on repeated/cached runs).

    Args:
        model: Loaded SentenceTransformer model (can be None when tweet_to_emb is given).
        all_networks: List of network objects (each with "network" key).
        num_steps: Size of the sliding window (number of tweets)
        shift: Stride of the sliding window
        tweet_to_emb: Optional dict[str, ndarray] — pre-computed tweet embeddings.
        embedding_dim: Required when tweet_to_emb is given and model is None.
    Returns:
        global_sbert_means: List of (Time, dim) arrays for each network
        global_sbert_vars: List of (Time, dim) arrays for each network
        global_phq9_means: List of (Time,) arrays: average PHQ-9 across agents at end of each window
    """

    global_sbert_means = []
    global_sbert_vars = []
    global_phq9_means = []

    if embedding_dim is None:
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
        phq9_per_window = []

        for w in range(num_windows):
            start_t = w * shift
            end_t = start_t + num_steps
            # Round index at end of window (for PHQ-9 at that time)
            round_idx = min(end_t - 1, max_iters - 1)

            # Average PHQ-9 across agents at this time
            scores = []
            for agent in net.all_agents:
                hist = getattr(agent, "all_phq9_sumscores", [])
                if round_idx < len(hist) and hist[round_idx] is not None:
                    scores.append(float(hist[round_idx]))
            if scores:
                phq9_per_window.append(np.mean(scores))
            else:
                phq9_per_window.append(np.nan)

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
                    valid_tweets = [t for t in window_segment if t != FC.NO_CONTENT]
                    tweets_in_window.extend(valid_tweets)

            # Embed and Average (Mean Pooling)
            if not tweets_in_window:
                window_centroid = np.zeros(embedding_dim)
                window_variance = np.zeros(embedding_dim)
            else:
                if tweet_to_emb is not None:
                    # Look up pre-computed embeddings
                    emb_list = [tweet_to_emb[t] for t in tweets_in_window if t in tweet_to_emb]
                    if emb_list:
                        embeddings = np.stack(emb_list)
                    else:
                        embeddings = np.zeros((1, embedding_dim))
                else:
                    embeddings = model.encode(tweets_in_window, batch_size=64, show_progress_bar=True)

                window_centroid = np.mean(embeddings, axis=0)
                window_variance = np.var(embeddings, axis=0)

            mean_net_window_vectors.append(window_centroid)
            var_net_window_vectors.append(window_variance)

        # Stack to create (Time, dim) matrix for this specific network run
        global_sbert_means.append(np.stack(mean_net_window_vectors))
        global_sbert_vars.append(np.stack(var_net_window_vectors))
        global_phq9_means.append(np.array(phq9_per_window))


    return global_sbert_means, global_sbert_vars, global_phq9_means


def sbert_for_runs(networks_per_setting: dict, num_steps=30, shift=5, mentalbert=True,
                   cache_path=None):
    """
    Computes SBERT embeddings using Mean Pooling over time windows.

    If cache_path is given, tweet-level embeddings are loaded from / saved to
    that .npz file, avoiding re-encoding on subsequent runs.

    Args:
        networks_per_setting: Dictionary of {setting_name: [list_of_networks]}
        num_steps: Size of the sliding window (number of tweets)
        shift: Stride of the sliding window
        mentalbert (bool): If True, use MentalBERT; else use default SBERT.
        cache_path (str | None): Optional .npz path for tweet embedding cache.

    Returns:
        mean_sbert_per_setting: {setting: (Time, dim) array} -> Average trajectory
        var_sbert_per_setting:  {setting: (Time, dim) array} -> Variance
        all_mats_per_setting:  {setting: List of (Time, dim) arrays} -> Individual runs
        mean_phq9_per_setting: {setting: (Time,) array} -> Mean PHQ-9 at end of each window (for coloring)
    """
    # Flatten networks and get slices
    all_networks, setting_slices = network_list_w_slices(networks_per_setting)

    # Build or load the tweet→embedding cache
    tweet_to_emb, embedding_dim = build_tweet_embedding_cache(
        all_networks, mentalbert=mentalbert, cache_path=cache_path
    )

    # Compute windowed means using cached lookups (no model needed)
    global_sbert_means, global_sbert_vars, global_phq9_means = mean_sbert_per_networks(
        model=None, all_networks=all_networks, num_steps=num_steps, shift=shift,
        tweet_to_emb=tweet_to_emb, embedding_dim=embedding_dim
    )

    # Group by setting and compute mean trajectories
    mean_sbert_per_setting = {}
    var_sbert_per_setting = {}
    mean_phq9_per_setting = {}
    all_mats_per_setting = {}
    
    for setting, (start, end) in setting_slices.items():
        matrices_over_runs = global_sbert_means[start:end]
        matrices_vars_over_runs = global_sbert_vars[start:end]
        phq9_over_runs = global_phq9_means[start:end]
        
        if not matrices_over_runs:
            continue

        # Trim to minimum length (in case some runs stopped early)
        min_len = min(m.shape[0] for m in matrices_over_runs)
        cut_short_means = [m[:min_len] for m in matrices_over_runs]
        cut_short_vars = [m[:min_len] for m in matrices_vars_over_runs]
        cut_short_phq9 = [p[:min_len] for p in phq9_over_runs]
        
        # Stack: (Num_Runs, time_window, Embedding_Dim)
        embedding_per_setting = np.stack(cut_short_means, axis=0)
        embedding_vars_per_setting = np.stack(cut_short_vars, axis=0)

        # Calculate Mean Trajectory over the runs
        mean_sbert_per_setting[setting] = np.mean(embedding_per_setting, axis=0)
        var_sbert_per_setting[setting] = np.mean(embedding_vars_per_setting, axis=0)
        # Mean PHQ-9 over runs (nanmean in case some windows have no scores)
        mean_phq9_per_setting[setting] = np.nanmean(np.stack(cut_short_phq9, axis=0), axis=0)
        
        if setting not in all_mats_per_setting:
            all_mats_per_setting[setting] = []

        all_mats_per_setting[setting].extend(cut_short_means)

    # Scalar per time: mean within-run variance (average over embedding dim)
    mean_within_var_per_setting = {
        s: np.mean(var_sbert_per_setting[s], axis=1) for s in var_sbert_per_setting
    }
    return mean_sbert_per_setting, var_sbert_per_setting, all_mats_per_setting, mean_phq9_per_setting, mean_within_var_per_setting


#=========================TF-IDF functions=========================

def retrieve_windowed_data(networks_data, num_steps= 30, shift=5, n_grams= None):
    '''Retrieve TF-IDF data from the network's agents' tweet histories.
    Args:
        network_data: Tuple containing the network objects.
        num_steps (int): Number of steps used in TF-IDF retrieval.
        shift (int): Shift between windows.
    Returns:
        all_tweets_extracted (List(str)): All tweets extracted for TF-IDF fitting.
        docs_per_network (List(List(str))): List of documents (one joined string per time window) per network.
        docs_per_network_tweets (List(List(List(str)))): List of (network -> window -> list of tweet strings) for within-window variance.
    '''
    all_tweets_extracted = []
    docs_per_network = []
    docs_per_network_tweets = []

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
        filtered_window_tweets = []  # list of lists of tweet strings per window (for within-window variance)
        w = 0
        while w < len(window_list):
            window_list[w] = [t for t in window_list[w] if t != FC.NO_CONTENT]
            tweets_this_window = list(window_list[w])
            joined = " ".join(window_list[w])
            if joined == "":
                print("WARNING: Empty tweet list for window ", w)
                window_list.pop(w)
            else:
                filtered_window_texts.append(joined)
                filtered_window_tweets.append(tweets_this_window)
                w += 1

        docs_per_network.append(filtered_window_texts)
        docs_per_network_tweets.append(filtered_window_tweets)
    
    all_tweets_extracted = [t for t in all_tweets_extracted if t!= FC.NO_CONTENT]

    if len(all_tweets_extracted) == 0:
        raise ValueError("No valid tweets found in the network for TF-IDF computation.")
    return all_tweets_extracted, docs_per_network, docs_per_network_tweets

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

def retrieve_tf_idf(networks, num_steps=30, shift=5, n_grams=None):
    all_tweets_extracted, docs_per_network, docs_per_network_tweets = retrieve_windowed_data(
        networks, num_steps=num_steps, shift=shift, n_grams=n_grams
    )

    print("using n-gram as vocab in TF-IDF: ")
    all_tweets_extracted = n_grams
    vocab, vectorizer = compute_tf_idf(all_tweets_extracted)

    # One vector per window (joined doc) - for mean trajectory
    global_tf_idf = [vectorizer.transform(doc).toarray() for doc in docs_per_network]

    # Within-window variance: per window, transform each tweet -> (n_tweets, V), then var(axis=0) -> (V,)
    global_tf_idf_vars = []
    for net_tweets in docs_per_network_tweets:
        var_list = []
        for window_tweets in net_tweets:
            if len(window_tweets) == 0:
                V = len(vectorizer.get_feature_names_out())
                var_list.append(np.zeros(V))
            elif len(window_tweets) == 1:
                V = len(vectorizer.get_feature_names_out())
                var_list.append(np.zeros(V))
            else:
                mat = vectorizer.transform(window_tweets).toarray()  # (n_tweets, V)
                var_list.append(np.var(mat, axis=0))
        global_tf_idf_vars.append(np.stack(var_list))
    return global_tf_idf, global_tf_idf_vars, vocab, vectorizer


def tf_idf_for_runs(networks_per_setting: dict, num_steps=30, shift=5, n_grams=None):
    '''Compute TF-IDF matrices for multiple network runs, with mean within-run variance.
    Args:
        networks_per_setting: Dict of {setting: [list of network dicts]}.
        num_steps (int): Number of steps used in TF-IDF retrieval.
        shift (int): Shift between windows.
    Returns:
        mean_tf_idf_per_setting: {setting: (T, V) mean TF-IDF matrix}
        all_mats_per_setting: {setting: List of (T, V) matrices}
        mean_within_var_per_setting: {setting: (T,) mean within-window variance (scalar per time)}
    '''
    all_networks, setting_slices = network_list_w_slices(networks_per_setting)

    tf_idf_matrices, tf_idf_vars, vocab, vectorizer = retrieve_tf_idf(
        all_networks, num_steps=num_steps, shift=shift, n_grams=n_grams
    )

    mean_tf_idf_per_setting = {}
    mean_within_var_per_setting = {}
    all_mats_per_setting = {}
    for setting, (start, end) in setting_slices.items():
        matrices_over_runs = tf_idf_matrices[start:end]
        vars_over_runs = tf_idf_vars[start:end]

        min_length = min(m.shape[0] for m in matrices_over_runs)
        trimmed_matrices = [m[:min_length] for m in matrices_over_runs]
        trimmed_vars = [v[:min_length] for v in vars_over_runs]
        print("min_length for setting ", setting, ": ", min_length)

        stacked_matrices = np.stack(trimmed_matrices, axis=0)
        mean_tf_idf_per_setting[setting] = np.mean(stacked_matrices, axis=0)

        # Mean within-run variance: average over runs, then scalar per time (mean over vocab dim)
        stacked_vars = np.stack(trimmed_vars, axis=0)  # (R, T, V)
        mean_var_tv = np.mean(stacked_vars, axis=0)   # (T, V)
        mean_within_var_per_setting[setting] = np.mean(mean_var_tv, axis=1)  # (T,)

        if setting not in all_mats_per_setting:
            all_mats_per_setting[setting] = []
        all_mats_per_setting[setting].extend(trimmed_matrices)

    return mean_tf_idf_per_setting, all_mats_per_setting, mean_within_var_per_setting


#=========================PCA functions=========================
    
def reduce_dimensionality(embedding_matrices, n_components=2):
    '''Reduce dimensionality of TF-IDF matrix using PCA.
    Args:
        embedding_matrices (np.array): TF-IDF matrix.
        n_components (int): Number of PCA components.
    Returns:
        reduced_runs (np.array): PCA-reduced data.
    '''
    assert n_components <= embedding_matrices[0].shape[1], "n_components must be <= number of features"
    embedding_stacked = np.vstack(embedding_matrices)
    pca = PCA(n_components=n_components)
    pca.fit(embedding_stacked)
    reduced_runs = [pca.transform(embedding_matrix) for embedding_matrix in embedding_matrices]
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

# =========================UMAP functions=========================
def reduce_dimensionality_umap(embedding_matrices, n_components=2, n_neighbors=15, min_dist=0.1):
    '''Reduce dimensionality using UMAP.'''
    embedding_stacked = np.vstack(embedding_matrices)
    
    # Initialize umap
    reducer = umap.UMAP(n_components=n_components, 
                        n_neighbors=n_neighbors, 
                        min_dist=min_dist,
                        random_state=42)
    
    reducer.fit(embedding_stacked)
    
    reduced_runs = [reducer.transform(embedding_matrix) for embedding_matrix in embedding_matrices]
    return reduced_runs, reducer

def fit_shared_umap(all_embedding_dicts, n_components=2):
    """Fit a single UMAP reducer on pooled embeddings from multiple configurations.

    Use this to ensure the same 2D projection across separate plots (e.g.,
    different network degrees). Pass the returned reducer as `shared_reducer`
    to `umap_on_means`.

    Args:
        all_embedding_dicts: list of dicts, each {setting: (T, D) array}
            (the mean_embedding_per_setting output from sbert_for_runs).
        n_components: UMAP output dimensionality.

    Returns:
        reducer: fitted UMAP object.
    """
    all_matrices = []
    for emb_dict in all_embedding_dicts:
        for matrix in emb_dict.values():
            all_matrices.append(matrix)

    stacked = np.vstack(all_matrices)
    reducer = umap.UMAP(n_components=n_components, random_state=58)
    reducer.fit(stacked)
    return reducer


def umap_on_means(embedding_per_setting, n_components=2, shared_reducer=None):
    """Apply UMAP to per-setting mean embeddings.

    Args:
        shared_reducer: Pre-fitted UMAP reducer (from fit_shared_umap).
            If provided, skips fitting and uses this reducer for transform.
            If None, fits a new reducer on the provided embeddings.
    """
    settings = list(embedding_per_setting.keys())

    if shared_reducer is not None:
        reducer = shared_reducer
    else:
        mean_matrices = [embedding_per_setting[setting] for setting in settings]
        embedding_stacked = np.vstack(mean_matrices)
        reducer = umap.UMAP(n_components=n_components, random_state=58)
        reducer.fit(embedding_stacked)

    mean_traj = {
        s: reducer.transform(embedding_per_setting[s])
        for s in settings
    }
    return mean_traj, reducer



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
            
            tweets_count = sum(1 for tweet in window if tweet != FC.NO_CONTENT)
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



def all_agent_phq9_cd(network, window_size, shift=1):
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
        phq9_scores = agent.all_phq9_sumscores[::shift]
        
        variances, autocorrs = calculate_agent_cd(phq9_scores, window_size)
        
        cd_results[agent.ID] = {
            'variance': variances.tolist(),
            'autocorrelation': autocorrs.tolist()
        }
    return cd_results



def all_agent__tweet_cd(network, window_size, shift=1):
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
        binary_sequence = [1 if tweet != FC.NO_CONTENT else 0 for tweet in history]
        
        variances, autocorrs = calculate_agent_cd(binary_sequence, window_size, shift)

        cd_results[agent.ID] = {
            'variance': variances.tolist(),
            'autocorrelation': autocorrs.tolist()
        }
    return cd_results


# ── Prompt robustness metrics ─────────────────────────────────────────────────

def compute_prompt_robustness(prompts: list[str], test_scores: list[float],
                               baseline_prompt: str = None,
                               seeds: list = None,
                               labels: list = None,
                               model_name: str = "all-MiniLM-L6-v2") -> dict:
    """Compute the pairwise cosine-similarity matrix across optimised prompts.

    When `baseline_prompt` is provided, it is appended as the final row/column of the
    matrix and labelled "minimal", so the heatmap visualises how each optimised prompt
    relates to its starting point in addition to its peers.

    Args:
        prompts:         Optimised prompt strings (one per seed/run).
        test_scores:     Corresponding test scores (same order as prompts).
        baseline_prompt: Un-optimised starting prompt; appended to the matrix when given.
        seeds:           Optional seeds aligned with `prompts`; used to build the default
                         "seed N" labels if `labels` is not provided.
        labels:          Optional explicit labels aligned with `prompts`. When given,
                         override the seeds-derived defaults. "minimal" is still appended
                         automatically for the baseline.
        model_name:      SBERT model for embedding.

    Returns dict with keys:
        'sim_matrix'  – (N, N) or (N+1, N+1) pairwise cosine similarity
        'labels'      – run labels (and "minimal" appended when baseline is included)
        'test_scores' – echo of input (no entry added for baseline)
        'has_baseline' – True if baseline_prompt was included in the matrix
    """
    all_prompts = list(prompts)
    if baseline_prompt is not None:
        all_prompts.append(baseline_prompt)

    model = generate_sbert_model(model_name=model_name)
    embeddings = model.encode(all_prompts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-10)
    sim_matrix = (normed @ normed.T).astype(float)

    if labels is not None:
        if len(labels) != len(prompts):
            raise ValueError(f"labels length {len(labels)} != prompts length {len(prompts)}")
        resolved_labels = list(labels)
    elif seeds is not None and len(seeds) == len(prompts):
        resolved_labels = [f"seed {s}" for s in seeds]
    else:
        resolved_labels = [f"run {i+1}" for i in range(len(prompts))]
    if baseline_prompt is not None:
        resolved_labels.append("minimal")

    return {
        "sim_matrix": sim_matrix,
        "labels": resolved_labels,
        "test_scores": list(test_scores),
        "has_baseline": baseline_prompt is not None,
    }


