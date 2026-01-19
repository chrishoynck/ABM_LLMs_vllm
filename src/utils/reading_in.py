from classes.network import RandomNetwork, SocialDistanceAttachment #, ScaleFreeNetwork
from utils.path_manager import PathManager
import ast, torch, os, random
import numpy as np

def read_in_network_properties(file_path):
    """
    Reads a network properties file and returns a dictionary of its properties.
    Args:
        file_path (str): Path to the saved properties file.
    Returns:
        dict: A dictionary containing the properties of the network.
    """
    properties = {}

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    for line in lines[2:]:  # Skip the header lines
        key, value = line.strip().split(": ", 1)

        if key in ("Number of Agents", 
                   "Number of Edges", 
                   "Seed", "Iterations", 
                   "Agent_w_Highest_Deg", 
                   "Degree", "Dimension", 
                   #"Initial Edges (m)"
                    ):
            properties[key] = int(value)

        elif key in ( "P value", "Update Fraction", "Alpha", "B"):
            properties[key] = float(value)
        
        elif key in ("sdc", "directed"):
            properties[key] = (value == "True")

        # save distorted fracs as metric
        elif key in ("Distorted Frac" , "Dist Step Frac"):
            # value = value.replace("nan", "0")
            distorted_fracs = ast.literal_eval(value)
            properties[key] = [float(f) for f in distorted_fracs]

        elif key == "Connections":
            # Parse connections as a list of tuples (id1, id2)
            connections = ast.literal_eval(value)
            properties[key] = [(int(a), int(b)) for a, b in connections]

        elif key == "CDS Info":
            cds_info = ast.literal_eval(value)
            properties[key] = [(float(frac_neigh), bool(act_agent), bool(distorted)) for frac_neigh, act_agent, distorted in cds_info]

        elif key in ("Network RNG State", "Torch RNG State", "Python Random State"):
            properties[key] = ast.literal_eval(value)

        elif key == "Agents":
            
            # Parse agents as a list of tuples
            value = value.replace("nan", "None")
            try:
                agents = ast.literal_eval(value)
                properties[key] = agents
            
            # if can't parse, raise error
            except ValueError as e:
                print("Failed to literal_eval Agents value:")
                print(value)
                raise

        
            # parsed_agents = []

            # # ADD WELLBEING -> DONE
            # for agent_id, phq9_sumscores, wellbeing, persona, activation_state, tweethistory, active_tweethistory, distorted_tweethistory, frac_distorted_neigh in agents:
            #     parsed_agents.append(
            #         (
            #             int(agent_id),
            #             phq9_sumscores,
            #             wellbeing,
            #             persona,
            #             activation_state,
            #             tweethistory,
            #             active_tweethistory,
            #             distorted_tweethistory,
            #             float(frac_distorted_neigh),
            #         )
            #     )
            # properties[key] = parsed_agents

        else:
            properties[key] = value
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

    # ADD WELLBEING ->> DONE
    # Collect agent and connection information
    for agent in network.all_agents:
        agent_info.append({
            "id": agent.ID,
            "phq9": agent.all_phq9_sumscores,
            "wb": agent.well_being,
            "persona": agent.persona,
            "act_state": agent.activation_state,
            "history": agent.tweethistory,
            "active_hist": agent.active_tweethistory,
            "distorted": agent.distorted_tweets,
            "frac_neigh": agent.frac_distorted_neigh
        })
    for conn in network.connections:
        connection_IDs.append((conn[0].ID, conn[1].ID))
    
    # Common properties for all network types
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
        "directed": network.directed
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

    # # Add properties specific to ScaleFreeNetwork
    # elif isinstance(network, ScaleFreeNetwork):
    #     properties["Initial Edges (m)"] = network.m
    #     properties["Total Degree"] = network.total_degree
    
    # Else social distance attachment
    elif isinstance(network, SocialDistanceAttachment):
        properties["sdc"] = network.sdc
        properties["Alpha"] = network.alpha
        properties["Degree"] = network.degree
        properties["Dimension"] = network.dim
        properties["B"] = network.b
    else:
        print("Network should be either scale-free, random, or social distance attachment")


    path_manager = PathManager(network=network)
    file_output_path = path_manager.get_full_network_path()

    with open(file_output_path, "w", encoding="utf-8") as file:
        file.write("Network Properties\n")
        file.write("==================\n")
        for key, value in properties.items():
            file.write(f"{key}: {value}\n")
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
    # elif "Initial Edges (m)" in props:
    #     # ScaleFreeNetwork
    #     m = int(props["Initial Edges (m)"])
    #     network = ScaleFreeNetwork(
    #         num_agents=num_agents,
    #         m=m,
    #         seed=seed,
    #         form_connections=False
    #     )

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

    network.cds_info = props["CDS Info"]
    network.degree_distribution = {}

    # set randomness: 
    network.rng = np.random.default_rng()
    network.rng.bit_generator.state = props["Network RNG State"]
   
    # new trick to try and restore python random state
    # if "Python Random State" in props:
    #     # This restores random.randint, random.choice, etc. to the exact point they were saved
    #     random.setstate(props["Python Random State"])

    # Pipeline
    # network._torch_gen = torch.Generator(device=pipe.model.device).manual_seed(seed)
    # state_tensor = torch.ByteTensor(props["Torch RNG State"])
    # network._torch_gen.set_state(state_tensor)

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


    