import json
import datetime
from classes.network import RandomNetwork, SocialDistanceAttachment #, ScaleFreeNetwork
from utils.tools.path_manager import PathManager, TestPathManager
import ast, torch, os, random
import numpy as np

class NetworkEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable objects like sets and numpy types."""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def read_in_network_properties(file_path):
    """
    Reads a network properties file and returns a dictionary of its properties.
    Args:
        file_path (str): Path to the saved properties file.
    Returns:
        dict: A dictionary containing the properties of the network.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        properties = json.load(file)
    
    return properties

def _write_meta_file(network, path_manager, args=None):
    """Write a human-readable meta.json alongside net.json for reporting purposes.

    Contains all experiment parameters that are useful for reporting but are not
    required (or are redundant) for the reconstruction path in generate_network.
    """
    degrees = [len(a.agent_connections) for a in network.all_agents]
    mean_degree = float(np.mean(degrees)) if degrees else 0.0

    # Topology statistics on the initial graph. agent.phq9_score holds the
    # initial PHQ-9; well_being["phq9_sumscore"] drifts during the run.
    # For a directed network we build a DiGraph so reciprocity is well defined:
    # network.connections then holds both (u, v) and (v, u) for a mutual pair.
    import networkx as nx
    directed = bool(getattr(network, "directed", False))
    g = nx.DiGraph() if directed else nx.Graph()
    for a in network.all_agents:
        wb = a.well_being or {}
        g.add_node(a.ID,
                   age=float(wb.get("age", 0)),
                   phq9_initial=float(a.phq9_score if a.phq9_score is not None
                                      else wb.get("phq9_sumscore", 0)))
    for conn in network.connections:
        g.add_edge(conn[0].ID, conn[1].ID)

    # Reciprocity is only defined for a directed graph (every undirected edge is
    # trivially mutual). Measured on the directed arc set, before symmetrisation.
    n_arcs = g.number_of_edges()
    if directed:
        # Mutual dyads: arcs present in both directions. Each dyad is two arcs,
        # so summing the reverse-arc test over all arcs counts it twice.
        n_reciprocal_pairs = sum(g.has_edge(v, u) for u, v in g.edges()) // 2
        reciprocity = nx.overall_reciprocity(g) if n_arcs else float("nan")
    else:
        n_reciprocal_pairs = n_arcs
        reciprocity = 1.0 if n_arcs else float("nan")

    # Clustering, assortativity and connectivity are measured on the undirected
    # projection so they stay comparable to the undirected configs (and the
    # calibrated target bands still apply). The projection keeps an edge when
    # either direction is present; for an undirected graph it is g itself.
    gu = g.to_undirected() if directed else g
    clustering = nx.average_clustering(gu) if gu.number_of_nodes() else float("nan")
    try:
        age_assort = nx.numeric_assortativity_coefficient(gu, "age")
    except Exception:
        age_assort = float("nan")
    try:
        phq9_assort = nx.numeric_assortativity_coefficient(gu, "phq9_initial")
    except Exception:
        phq9_assort = float("nan")
    if gu.number_of_nodes():
        lcc_frac = len(max(nx.connected_components(gu), key=len)) / gu.number_of_nodes()
    else:
        lcc_frac = float("nan")

    # Out-clustering (Fagiolo 2007): among each node's out-neighbours, the
    # fraction of ordered pairs (j, k) closed by an arc j->k. This is the
    # directed "friend-of-a-friend" measure (path i->j->k closed by i->k) and is
    # the meaningful clustering for a directed influence network — on these SDA
    # graphs it runs ~half the undirected-projection `clustering`. For an
    # undirected graph in- and out-neighbourhoods coincide, so it reduces to the
    # ordinary clustering above.
    if directed and g.number_of_nodes():
        A    = nx.to_numpy_array(g, nodelist=list(g.nodes()))
        dout = A.sum(axis=1)
        tri  = np.einsum("ij,jk,ik->i", A, A, A)        # (A^2 A^T)_ii
        den  = dout * (dout - 1)                          # ordered out-neighbour pairs
        with np.errstate(invalid="ignore", divide="ignore"):
            per = np.where(den > 0, tri / den, np.nan)
        clustering_out = float(np.nanmean(per)) if np.isfinite(per).any() else float("nan")
    else:
        clustering_out = clustering

    import os
    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": os.environ.get("LLAMA_ID", "meta-llama/Llama-3.1-8B-Instruct"),
        "network_type": type(network).__name__,
        "common": {
            "num_agents":  len(network.all_agents),
            "seed":        network.seed,
            "directed":    network.directed,
            "state":       network.state,
            "final_iteration": network.iterations,
        },
        "phq9_settings": {
            "mode":            getattr(network, "phq9_mode", "llm"),
            "init_zero":       getattr(network, "init_phq9_zero", False),
            "sample_fraction": getattr(network, "sample_phq9", None),
            "cap":             getattr(network, "cap_phq9", False),
            "threshold":       getattr(network, "phq9_threshold", 0),
            "bert_mentalbert": getattr(network, "bert_mentalbert", True),
            "bias_corrected":  getattr(network, "_phq9_bias_table", None) is not None,
            "bias_table_path": getattr(network, "_bias_table_path", None),
            "bias_table":      (np.round(network._phq9_bias_table, 4).tolist()
                                if getattr(network, "_phq9_bias_table", None) is not None else None),
        },
        "topology": {
            "mean_degree":         round(mean_degree, 3),
            "clustering":          round(clustering, 4),
            "clustering_out":      round(clustering_out, 4),
            "age_assort":          round(age_assort, 4),
            "phq9_assort_initial": round(phq9_assort, 4),
            "lcc_frac":            round(lcc_frac, 4),
            "reciprocity":         round(reciprocity, 4),
            "n_reciprocal_pairs":  int(n_reciprocal_pairs),
            "powerlaw_gamma": round(getattr(network, "_powerlaw_gamma", float("nan")), 4),
            "powerlaw_ks":    round(getattr(network, "_powerlaw_ks",    float("nan")), 4),
        },
    }

    if isinstance(network, SocialDistanceAttachment):
        meta["network_params"] = {
            "alpha":      network.alpha,
            "degree":     network.degree,
            "dim":        network.dim,
            "b_fitted":   round(network.b, 6),
            "dist_type":  getattr(network, "dist_type", "gaussian_clusters"),
            "sdc":        network.sdc,
            "age_weight":    getattr(network, "age_weight",    1.0),
            "use_phq9":      getattr(network, "use_phq9",      True),
            "latent_weight": getattr(network, "latent_weight", 1.0),
            "n_clusters":    getattr(network, "n_clusters",    4),
        }
    elif isinstance(network, RandomNetwork):
        meta["network_params"] = {
            "p": network.p,
            "k": network.k,
        }

    if args is not None:
        meta["simulation"] = {
            "rounds":          getattr(args, "rounds",          None),
            "update_fraction": getattr(args, "update_fraction", None),
            "check_point":     getattr(args, "check_point",     None),
            "enforce_ngrams":  getattr(args, "enforce_ngrams",  False),
            "cds_dynamic":     getattr(args, "cds_dynamic",     None),
            "log":             getattr(args, "log",             None),
        }

    meta_path = path_manager.get_run_directory(is_plot=False) / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    return str(meta_path)


def read_out_network_properties(network, seed, dist_per_step, distorted_fracs, args=None):
    """
    Extracts and returns the properties of a network for analysis or storage.
    Supports RandomNetwork, ScaleFreeNetwork, SocialDistanceAttachment.
    Stores it in a dictionary, values can be accessed with the corresponding keys. 
    Useful for effectively extracting network properties. 

    Args:
        network (object): The network object to extract properties from.
        seed (int): The seed used for network generation.

    Returns:
        dict: A dictionary containing the properties of the network:
        - Number of Agents
        - Number of Edges
        - Seed
        - Connections
        - Agents
        - P value (for RandomNetwork)
        - Degree (k) (for RandomNetwork)
        # - Initial Edges (m) (for ScaleFreeNetwork)
        # - Total Degree (for ScaleFreeNetwork)
        # - Degree Distribution (for ScaleFreeNetwork)
    """

    agent_info = []
    connection_IDs = []

    def parse_well_being(wb_dict):
        """Catch nans in well-being dicts."""
        new_dict = {}
        for key, val in wb_dict.items():
            if isinstance(val, (float, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    new_dict[key] = None
                else:
                    if isinstance(val, np.floating):
                        new_dict[key] = float(val)
                    else:
                        new_dict[key] = val
            else:
                new_dict[key] = val
            
        return new_dict

    for agent in network.all_agents:
        agent_info.append({
            "id": agent.ID,
            "phq9": agent.all_phq9_sumscores,
            "wb": parse_well_being(agent.well_being),
            "persona": agent.persona,
            "act_state": agent.activation_state,
            "history": agent.tweethistory,
            "active_hist": agent.active_tweethistory,
            "distorted": agent.distorted_tweets,
            "frac_neigh": agent.frac_distorted_neigh,
            "neighbor_history": agent.neighbor_history,
        })
    
    for conn in network.connections:
        connection_IDs.append((conn[0].ID, conn[1].ID))

    properties = {
        "Number of Agents": len(network.all_agents),
        "Number of Edges": len(network.connections),
        "Seed": seed,
        "State": network.state,
        "Connections": connection_IDs,
        "Agents": agent_info, 
        "Iterations": network.iterations,
        "Distorted Frac": [float(x) for x in distorted_fracs],
        "Dist Step Frac": [float(x) for x in dist_per_step],
        "CDS Info": network.cds_info,
        "Dynamic CDS": getattr(network, 'cds_dynamic', False),
        "Agent_w_Highest_Deg": network.agent_w_highest_deg.ID,
        "directed": network.directed,
        "sample_phq9": getattr(network, 'sample_phq9', None),
        "cap_phq9": getattr(network, 'cap_phq9', False),
        "phq9_threshold": getattr(network, 'phq9_threshold', 0),
        "init_phq9_zero": getattr(network, 'init_phq9_zero', False),
        "phq9_mode": getattr(network, 'phq9_mode', 'llm'),
        "bert_regressor_path": getattr(network, '_bert_regressor_path', None),
        "bert_mentalbert": getattr(network, 'bert_mentalbert', True),
        "bias_corrected": getattr(network, '_phq9_bias_table', None) is not None,
        "bias_table_path": getattr(network, '_bias_table_path', None),
        "phq9_bias_table": (np.round(network._phq9_bias_table, 4).tolist()
                            if getattr(network, '_phq9_bias_table', None) is not None else None),
    }

    # randomness:
    try:
        properties["Torch RNG State"] = network._torch_gen.get_state().tolist()
    except:
        # Fallback if _torch_gen is not set
        properties["Torch RNG State"] = None

    properties["Network RNG State"] = network.rng.bit_generator.state
    properties["Python Random State"] = random.getstate()

    # Add properties specific to RandomNetwork
    if isinstance(network, RandomNetwork):
        properties["P value"] = network.p
        properties["Degree (k)"] = network.k
    
    # Else social distance attachment
    elif isinstance(network, SocialDistanceAttachment):
        properties["sdc"] = network.sdc
        properties["Alpha"] = network.alpha
        properties["Degree"] = network.degree
        properties["Dimension"] = network.dim
        properties["B"] = network.b
    else:
        print("Network should be either random, or social distance attachment")

    path_manager = PathManager(network=network)
    file_output_path = path_manager.get_full_network_path()

    with open(file_output_path, "w", encoding="utf-8") as file:
        json.dump(properties, file, indent=4, cls=NetworkEncoder)

    _write_meta_file(network, path_manager, args=args)
    return file_output_path


def generate_network(args, pipe, file_path=None):
    """
    Load a single network from a saved properties file created by get_network_properties.

    This fully reconstructs:
    - topology (connections)
    - per-agent state (persona, activation_state, tweet histories, frac_distorted_neigh,
      neighbor_history)
    - iterations counter

    Args:
        args: Argument namespace; only used to resolve the on-disk path via
            ``PathManager`` when ``file_path`` is not given (may be None then).
        pipe: Unused here (kept for call-site symmetry with the update path).
        file_path (str | Path, optional): Explicit path to the saved net.json. When
            provided, ``PathManager`` is bypassed so networks stored under
            non-standard sub-directories (e.g. ``debiased/``, ``init_0/``) can be
            loaded directly.

    Returns:
        network: A reconstructed RandomNetwork or ScaleFreeNetwork instance.
    """
    if file_path is None:
        pm = PathManager(args=args)
        file_path = pm.get_full_network_path()
    props = read_in_network_properties(file_path)
    
    # metrics
    distorted_fracs = props["Distorted Frac"]
    dist_per_step = props["Dist Step Frac"]
    
    # network props
    num_agents = props["Number of Agents"]
    seed = props["Seed"]
    iterations = props["Iterations"]
    directed = props.get("directed", False)

    # Create right network type
    if "P value" in props:
        # RandomNetwork
        p = props["P value"]
        network = RandomNetwork(
            num_agents=num_agents,
            seed=seed,
            p=p,
            form_connections=False
        )

    else:
        # SocialDistanceAttachment
        alpha = props["Alpha"]
        degree = props["Degree"]
        dim = props["Dimension"]
        b = props["B"]
        sdc = props["sdc"]
        network = SocialDistanceAttachment(
            num_agents=num_agents,
            alpha=alpha,
            degree=degree,
            dim=dim,
            seed=seed,
            sdc=sdc, 
            form_connections=False
        )
        network.b = b  # set b if needed

    # Make sure we continue from the saved iteration count
    network.iterations = iterations
    network.directed = directed
    
    if "State" in props:
        network.state = props["State"]
    else:
        # Fallback if loading old files without State property
        network.state = "basis"

    network.sample_phq9 = props.get("sample_phq9", None)
    network.cap_phq9 = props.get("cap_phq9", False)
    network.phq9_threshold = props.get("phq9_threshold", 0)
    network.init_phq9_zero = props.get("init_phq9_zero", False)

    phq9_mode = props.get("phq9_mode", "llm")
    network.phq9_mode = phq9_mode
    if phq9_mode == "bert":
        bert_regressor_path = props.get("bert_regressor_path")
        bert_mentalbert = props.get("bert_mentalbert", True)
        network.bert_mentalbert = bert_mentalbert
        network._init_bert_components(
            bert_regressor=None,
            bert_encoder=None,
            bert_regressor_path=bert_regressor_path,
            bert_mentalbert=bert_mentalbert,
            bert_device=None,
        )

    network.cds_info = props["CDS Info"]
    # Pursue the checkpoint's dynamic-CDS choice. Pre-flag checkpoints have no
    # "Dynamic CDS" key but always populated cds_info, so a non-empty cds_info
    # implies it was on; otherwise default to off (neighbor_history only).
    network.cds_dynamic = props.get("Dynamic CDS", len(network.cds_info) > 0)
    network.degree_distribution = {}

    # set randomness: 
    network.rng = np.random.default_rng()
    network.rng.bit_generator.state = props["Network RNG State"]

    # create a map to map index ot id, (currently index is same as id)
    id_to_agent = {agent.ID: agent for agent in network.all_agents}

    # Restore agents
    for agent_data in props["Agents"]:

        ag = id_to_agent[agent_data["id"]]
        ag.all_phq9_sumscores = [int(score) for score in agent_data.get("phq9", [])]
        ag.well_being = agent_data.get("wb", {})
        ag.persona = agent_data.get("persona", "")
        ag.activation_state = agent_data.get("act_state", False)
        ag.tweethistory = list(agent_data.get("history", []))
        ag.active_tweethistory = list(agent_data.get("active_hist", []))
        ag.distorted_tweets = list(agent_data.get("distorted", []))
        ag.frac_distorted_neigh = agent_data.get("frac_neigh", 0.0)
        ag.neighbor_history = list(agent_data.get("neighbor_history", []))
        ag.agent_connections = set()  # will be populated below

        # rebuild degree distribution when adding connections
        network.degree_distribution[ag] = 0

    # Restore connections
    # Clear whatever constructor created
    network.connections = set()

    # reset connections. 
    for id1, id2 in props["Connections"]:
        a1 = id_to_agent[id1]
        a2 = id_to_agent[id2]
        network.add_connection(a1, a2)

    network.agent_w_highest_deg = id_to_agent[props["Agent_w_Highest_Deg"] ]
    return network, distorted_fracs, dist_per_step


def log_network_state(network, seed, dist_per_step, distorted_fracs):
    logged_path = read_out_network_properties(
        network, 
        seed, 
        dist_per_step, 
        distorted_fracs
    )
    print(f"Logged network state to {logged_path}")
    return logged_path


# ─────────────────────────────────────────────────────────────────────────────
#  TestLLMs checkpoint  (no network structure needed)
# ─────────────────────────────────────────────────────────────────────────────

def write_out_tester(tester, model_name: str, temp: float, top_p: float,
                     check_point: int, interaction: bool, mistake_dict: dict):
    """
    Serialise the full state of a :class:`TestLLMs` instance to a JSON file
    so that a crashed run can be resumed exactly where it left off.

    Saved information
    -----------------
    - Run meta  : seed, num_agents, iterations, model_name, interaction flag
    - PHQ-9 sequences: the per-agent permutation order + current index
    - Per-agent : id, persona, well_being, tweethistory, all_phq9_sumscores,
                  _tweets_since_phq9_update, activation_state
    - mistake_dict : accumulated error dictionary so far
    - RNG state : numpy bit-generator state (so sampling is reproducible)

    Returns
    -------
    str  – path the checkpoint was written to
    """
    agent_info = []
    for agent in tester.all_agents:
        agent_info.append({
            "id": agent.ID,
            "persona": agent.persona,
            "well_being": agent.well_being,
            "tweethistory": list(agent.tweethistory),
            "all_phq9_sumscores": [int(s) for s in agent.all_phq9_sumscores],
            "tweets_since_phq9_update": agent._tweets_since_phq9_update,
            "activation_state": agent.activation_state,
        })

    # mistake_dict keys are ints; JSON only allows str keys
    serialisable_mistakes = {str(k): v for k, v in mistake_dict.items()}

    # phq9_sequences / phq9_indices – keys are agent IDs (ints)
    phq9_sequences = {str(k): v for k, v in tester.phq9_sequences.items()}
    phq9_indices   = {str(k): v for k, v in tester.phq9_indices.items()}

    properties = {
        "type": "TestLLMs_checkpoint",
        "model_name": model_name,
        "seed": tester.seed,
        "num_agents": tester.num_agents,
        "iterations": tester.iterations,
        "interaction": interaction,
        "check_point": check_point,
        "temp": temp,
        "top_p": top_p,
        "phq9_sequences": phq9_sequences,
        "phq9_indices": phq9_indices,
        "mistake_dict": serialisable_mistakes,
        "agents": agent_info,
        "RNG State": tester.rng.bit_generator.state,
    }

    pm = TestPathManager(
        model_name=model_name,
        temp=temp,
        top_p=top_p,
        check_point=check_point,
        seed=tester.seed,
        interaction=interaction,
    )
    out_path = pm.get_run_directory(is_plot=False) / "checkpoint.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(properties, f, indent=4, cls=NetworkEncoder)

    return str(out_path)


def read_in_tester(file_path: str) -> dict:
    """
    Load a raw checkpoint dictionary from *file_path*.
    Raises ``FileNotFoundError`` if the file does not exist.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tester_checkpoint(file_path: str):
    """
    Reconstruct a :class:`TestLLMs` instance and its mistake_dict from a
    checkpoint file written by :func:`write_out_tester`.

    Returns
    -------
    tester       – fully-restored TestLLMs instance (no LLM loaded yet)
    mistake_dict – accumulated error dict  {phq9_score: [errors]}
    """
    # Lazy import avoids circular imports at module level
    from utils.create_data.test_phq9_llms import TestLLMs
    from classes.agent import Agent

    props = read_in_tester(file_path)

    # Restore agents from saved data
    agents = []
    for ad in props["agents"]:
        ag = Agent(
            ID=ad["id"],
            persona=ad["persona"],
            well_being=ad.get("well_being"),
        )
        ag.tweethistory = list(ad.get("tweethistory", []))
        ag.all_phq9_sumscores = [int(s) for s in ad.get("all_phq9_sumscores", [])]
        ag._tweets_since_phq9_update = int(ad.get("tweets_since_phq9_update", 0))
        ag.activation_state = bool(ad.get("activation_state", False))
        agents.append(ag)

    # Build TestLLMs with the restored agents (bypasses __init__ agent creation)
    tester = TestLLMs(
        seed=props["seed"],
        num_agents=props["num_agents"],
        agents=agents,
        interaction=props.get("interaction", False),
    )

    # Overwrite the freshly-generated PHQ-9 sequences with the saved ones
    tester.phq9_sequences = {int(k): v for k, v in props["phq9_sequences"].items()}
    tester.phq9_indices   = {int(k): v for k, v in props["phq9_indices"].items()}

    # Restore iteration counter and RNG state
    tester.iterations = props["iterations"]
    tester.rng.bit_generator.state = props["RNG State"]

    # Restore mistake_dict (JSON keys are strings)
    mistake_dict = {int(k): v for k, v in props["mistake_dict"].items()}

    print(
        f"Loaded TestLLMs checkpoint: model={props.get('model_name')}, "
        f"iterations={tester.iterations}, seed={tester.seed}"
    )
    return tester, mistake_dict
