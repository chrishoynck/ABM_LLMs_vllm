import json
from classes.network import RandomNetwork, SocialDistanceAttachment #, ScaleFreeNetwork
from utils.path_manager import PathManager
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

def read_out_network_properties(network, seed, dist_per_step, distorted_fracs):
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
            "frac_neigh": agent.frac_distorted_neigh
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
        "Agent_w_Highest_Deg": network.agent_w_highest_deg.ID,
        "directed": network.directed,
        "sample_phq9": getattr(network, 'sample_phq9', None),
        "cap_phq9": getattr(network, 'cap_phq9', False),
        "phq9_threshold": getattr(network, 'phq9_threshold', 0),
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
        
    return file_output_path


def generate_network(args, pipe):
    """
    Load a single network from a saved properties file created by get_network_properties.

    This fully reconstructs:
    - topology (connections)
    - per-agent state (persona, activation_state, tweet histories, frac_distorted_neigh)
    - iterations counter

    Args:
        file_path (str): Path to the saved properties file.

    Returns:
        network: A reconstructed RandomNetwork or ScaleFreeNetwork instance.
    """
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

    network.cds_info = props["CDS Info"]
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
    