
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed, pipeline
import os, torch
import sys, argparse
import inspect
from vllm import LLM, SamplingParams
# from src.classes.agent import Agent
import utils.metrics as metrics
from classes.vllm_networks import RandomNetwork, ScaleFreeNetwork, SocialDistanceAttachment
import utils.load_personas as lp
import utils.visualization as vis
import utils.reading_in as ri

######################################################################
### Llama 2 Setup
######################################################################


print(torch.cuda.is_available())
llama_model= "meta-llama/Llama-3.2-1B-Instruct"

# when setting possible enironment variables in the future
MODEL_ID = os.environ.get("LLAMA_ID", llama_model)
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", None)
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

USE_VLLM = False

DTYPE_STR = "bfloat16" if torch.cuda.is_available() else "float32"

# set seeds for reproducibility
SEED = 1234
os.environ["PYTHONHASHSEED"] = str(SEED)   # best set before Python starts                # if you still use np.random.*
set_seed(SEED)                              # seeds Python, NumPy, Torch (HF helper)


# setyp initial llm
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, use_fast=True)

# Ensure a pad token exists (prevents fallback messages)
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

def get_llm():
    """Create a vLLM LLM engine instead of a HF pipeline."""
    llm = LLM(
        model=MODEL_ID,
        dtype=DTYPE_STR,
        trust_remote_code=True,
        download_dir=CACHE_DIR,
    )
    return llm

# def get_pipe():
#     # Pipeline configuration
#     pipe = pipeline(
#     "text-generation",
#     model=MODEL_ID,
#     tokenizer=tokenizer,
#     # local_files_only = True,
#     torch_dtype=DTYPE,
#     device_map="auto",                        # shard/assign automatically
#     trust_remote_code=True,
#     max_new_tokens=256,
#     return_full_text=False,                        
#     )
#     return pipe

def build_network(args, personas, well_being, depressed_personas=None):
    '''Build network based on given arguments.
    Args:
        args: Argument namespace containing network parameters.
        personas: List of personas for agents.
        depressed_personas: List of depressed personas for agents.
    Returns:
        network: Generated network object.
    '''
    if args.net == "sf":
        return ScaleFreeNetwork(
            m=args.m,
            num_agents=args.num_agents,
            seed=args.seed,
            well_being= well_being,
            personas=personas,
            depressed_personas=depressed_personas,
            
        )
    elif args.net == "sda" or args.net == "sdc":
        return SocialDistanceAttachment(
            alpha=args.alpha,
            degree=args.degree,
            dim=args.dim,
            num_agents=args.num_agents,
            seed=args.seed,
            plot= True,
            well_being= well_being,
            personas=personas,
            power_law=(args.net == "sdc"),
            depressed_personas=depressed_personas,
        )
    else:
        return RandomNetwork(
            p=args.p,
            k=args.k,
            num_agents=args.num_agents,
            seed=args.seed,
            personas=personas,
            well_being= well_being,
            depressed_personas=depressed_personas,
        )

def generate_parser():
    "parse all given arguments"
    parser = argparse.ArgumentParser(description="Run LLM agent simulation.")

    # Network Settings
    parser.add_argument("net", nargs="?", choices=["sf", "r", "sda", "sdc"], default="sf", help="Network type: sf=ScaleFree, r=Random, sda=SocialDistanceAttachment")
    parser.add_argument("--rounds", type=int, default=10, help="Number of update rounds")
    parser.add_argument("--num_agents", type=int, default=10, help="Total number of agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")


    # Scale-free specific
    parser.add_argument("--m", type=int, default=2, help="Edges per new node (scale-free)")

    # Random network specific
    parser.add_argument("--p", type=float, default=0.5, help="Edge probability (random network)")
    parser.add_argument("--k", type=int, default=0, help="Regular degree (Watts–Strogatz if >0)")

    # Social distance attachment specific
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha parameter for social distance attachment (scale-free)")
    parser.add_argument("--degree", type=int, default=2, help="Dimensionality of social space (scale-free)")
    parser.add_argument("--dim", type=int, default=2, help="Dimensionality of social space (social distance attachment)")

    # Experiment Settings
    parser.add_argument("--depressed", action="store_true", help="Include depressed personas")
    parser.add_argument("--enforce_ngrams", action="store_true", help="Enforce distorted-language n-grams in tweets")

    # Load existing network
    parser.add_argument(
    "--use_saved_network",
    nargs="?",            # argument is optional
    const=0,              # if flag is given without value -> 0
    type=int,             # if a value is given -> int
    help=(
        "Use saved network properties to reload network. "
        "If provided without a value, just load the network. "
        "If provided with an integer, run that many extra rounds."
    ),
)
    parser.add_argument("--save", action="store_true", help="Save network properties after simulation")

    return parser.parse_args()

def update_network(network, pipe, fracs_dist_step = [], running_fracs = [], rounds=1, seed=42, enforce_ngrams = False):
    """Update the network for one round and return the mean fraction of distorted tweets."""

    # only enforce n-grams if specified
    if enforce_ngrams:
        distorted_tweets = lp.load_distorted_tweets("data/distorted_tweets.csv", numtweets=1000, seed=seed)
    else:
        distorted_tweets = []
    
    n_grams = metrics.load_ngrams_tsv("data/distorted_language_ngrams.tsv")
    
    for _ in range(rounds):
        mean_running_frac, frac_distorted_this_step = network.update_round(tokenizer, pipe, n_grams=n_grams, distorted_tweets=distorted_tweets)
        print(f"Round {network.iterations}: Mean running fraction of distorted agents: {mean_running_frac:.4f}, Fraction distorted this step: {frac_distorted_this_step:.4f} ")
        running_fracs.append(mean_running_frac)
        fracs_dist_step.append(frac_distorted_this_step)
        if network.iterations % 10 == 0:
            print(f"finished round {network.iterations}")
    return running_fracs, network, fracs_dist_step

def run_simulation(args, pipe=None):
    '''Wrapper to run the full simulation from network generation to updates.
    Args:
        args: Argument namespace containing network parameters.
        pipe: LLM pipeline for network generation.
    Returns:
        network: Generated network object.
        running_fracs: List of running fractions of distorted tweets.
        fracs_dist_step: List of fractions of distorted tweets per step.
        '''
    set_seed(SEED)     
    print(type(pipe.model))
    print("GENERATOR: ", inspect.signature(pipe.model.generate))

    # build an argparse-like namespace
    
    personas = None
    # load personas
    if False:
        personas = lp.load_personas_from_file("data/personas_10k.csv", args.num_agents, seed=args.seed)
    
    well_being = lp.load_phq9("data/confidential/phq9.sav", args.num_agents, seed=args.seed)

    # only load depressed personas if specified
    if args.depressed:
        depressed_personas = lp.load_depressed_personas("data/depressed.csv", personass_to_load=1, seed=args.seed)
    else:
        depressed_personas = None

    # build network
    network = build_network(args, well_being=well_being, personas=personas, depressed_personas=depressed_personas)

    # run updates
    running_fracs, network, fracs_dist_step = update_network(network, 
                                                             pipe=pipe, 
                                                             fracs_dist_step=[], 
                                                             running_fracs=[], 
                                                             rounds=args.rounds,
                                                             seed=args.seed, 
                                                             enforce_ngrams=args.enforce_ngrams)
    return network, running_fracs, fracs_dist_step

def update_existing_network(pipe, args, state= 'basis'):
    '''Update an existing network from a file path.
    Args:
        file_path (str): Path to the saved network file.
        pipe: LLM pipeline for network generation.
        args: Argument namespace containing network parameters.
    Returns:
        network: Updated network object.
        running_fracs: List of running fractions of distorted tweets.
        fracs_dist_step: List of fractions of distorted tweets per step.
    '''
    # reload network from saved properties
    file_path = retrieve_existing_net(args)
    network, running_fracs, fracs_dist_step= ri.generate_network(file_path, pipe)
    running_fracs, network, fracs_dist_step = update_network(network, 
                                                             pipe=pipe, 
                                                             fracs_dist_step=[], 
                                                             running_fracs=[], 
                                                             rounds=args.rounds, 
                                                             seed=args.seed, 
                                                             enforce_ngrams=args.enforce_ngrams)

    # tweet_history = [(a.ID, a.tweethistory) for a in network.all_agents]
    if args.save:
        file_output_path = ri.read_out_network_properties(network, args.seed, fracs_dist_step, running_fracs, enforce_ngrams=args.enforce_ngrams, depressed=args.depressed)
        print(f"Network properties saved to {file_output_path}")
    networks = [(state, network)]
    return networks, running_fracs, fracs_dist_step

def retrieve_existing_net(args):
    '''Retrieve file path for existing network based on arguments.
    Args:
        args: Argument namespace containing network parameters.
    Returns:
        file_path (str): Path to the saved network file.
    '''
    if args.enforce_ngrams:
        state = "enforced_ngrams"
    elif args.depressed:
        state = "depressed"
    else:
        state = "basis"
    
    if args.net == "sf":
        what_network = "sf"
        parameter = f'{args.m}'
    elif args.net == "sda":
        what_network = "sda"
        parameter = f'{args.alpha}_d{args.degree}'.replace(".", "_")
    elif args.net == "sdc":
        what_network = "sdc"
        parameter = f'{args.alpha}_d{args.degree}'.replace(".", "_")
    else:
        parameter = f'{str(args.p).replace(".", "_")}'
        what_network = "rand"
    
    file_path = f"data/networks/{state}/{what_network}/{parameter}/num_agents{args.num_agents}_{args.rounds}_net_{args.seed}.txt"
    return file_path

def generate_new_net(args, pipe, state= 'basis',  save_network=True):
    '''Wrapper to generate a new network and run the simulation.
    Args:
        args: Argument namespace containing network parameters.
        pipe: LLM pipeline for network generation.
    Returns:
        networks (list): List containing the generated network.
        running_fracs (list): List of running fractions of distorted tweets.
        fracs_dist_step (list): List of fractions of distorted tweets per step.
    '''
    network, running_fracs, fracs_dist_step = run_simulation(args = args, pipe=pipe)
    # return output file the network is printed to
    if args.save:
        file_output_path = ri.read_out_network_properties(network, args.seed, fracs_dist_step, running_fracs, enforce_ngrams=args.enforce_ngrams, depressed=args.depressed)
        print(f"Network properties saved to {file_output_path}")

    networks = [(state, network)]

    return networks, running_fracs, fracs_dist_step

def retrieve_spcific_net(args, states, seedjes, pipe): 
    '''Wrapper to retrieve specific networks based on different states.
    Args:
        args: Argument namespace containing network parameters.
        states (list): List of states (depressed, enforced_ngrams, basis) to retrieve networks for.
        pipe: LLM pipeline for network generation.
    Returns:
        networks (list): List of retrieved networks.
    '''
    networks = []
    for state in states:

        # one_seed for each run
        for seed in seedjes:
            args.seed = seed
            args.enforce_ngrams = (state == "enforced_ngrams")
            args.depressed = (state == "depressed")
            file_path = retrieve_existing_net(args)

            # reload network from saved properties
            network, running_fracs, fracs_dist_step= ri.generate_network(file_path, pipe)
            networks.append((state, network))

    # print(f"retrieved {len(networks)} networks")
    return networks, running_fracs, fracs_dist_step


if __name__ == "__main__":


    llm = get_llm()
    pipe = llm
    # keep CLI behavior
    args = generate_parser()

    # experiment states
    # states = ["basis"]

    if args.depressed:
        states = ["depressed"]
    elif args.enforce_ngrams:
        states = ["enforced_ngrams"]
    else:
        states = ["basis"]

    #experiment
    states = ["basis", "depressed", "enforced_ngrams"]
    seedjes = [42, 53]
    
    if args.use_saved_network is not None:
        # load in existing network and update if specified
        networks, running_fracs, fracs_dist_step = retrieve_spcific_net(args, states, seedjes, pipe)
        if args.use_saved_network > 0:
            # run additional rounds
            networks, running_fracs, fracs_dist_step = update_existing_network(pipe, args, state=states[0])

    else:
        state = states[0]
        # generate new network
        networks, running_fracs, fracs_dist_step = generate_new_net(args, pipe, state=state)

    network = networks[0]

    # plot TF-IDF PCA
    n_grams = metrics.load_ngrams_tsv("data/distorted_language_ngrams.tsv")
    # global_tf_idf, _, _ = metrics.retrieve_tf_idf(networks, num_steps=100, shift=5, n_grams=n_grams)

    setting_network = {}
    for state, network in networks:
        if state not in setting_network:
            setting_network[state] = [network]
        else:
            setting_network[state].append(network)

    meanvar_tf_idf_per_setting, all_mats_per_setting = metrics.tf_idf_for_runs(setting_network, num_steps=100, shift=5)
    mean_traj, pca = metrics.pca_on_means(meanvar_tf_idf_per_setting, n_components=2)
    

    std_traj, _ = metrics.traj_variance_in_pca_space(all_mats_per_setting, pca)
    # vis.plot_tf_idf_PCA(mean_traj, std_traj, num_steps=100, shift=5, save= args.save)
    vis.plot_tf_idf_PCA_runs(mean_traj, std_traj, num_steps=100, shift=5, save= args.save)
    

    # vis.plot_running_fracs(running_fracs, mmtje, pmtje, enforced_ngrams=args.enforce_ngrams, depressed=args.depressed, type_nn=args.net)
    # vis.plot_distorted_fracs(fracs_dist_step, mmtje, pmtje, enforced_ngrams=args.enforce_ngrams, depressed=args.depressed, type_nn=args.net)
    # vis.print_network(network)

    


