
from transformers import AutoTokenizer, set_seed  #, pipeline, BitsAndBytesConfig
import os, torch
import sys, argparse, time
import numpy as np
import inspect
from vllm import LLM, SamplingParams
import utils.metrics as metrics
from utils.path_manager import PathManager
from classes.network import RandomNetwork,  SocialDistanceAttachment #, ScaleFreeNetwork,
import utils.load_personas as lp
import utils.visualization as vis
import utils.reading_in as ri
import warnings
from utils.test_phq9_llms import TestLLMs


# Suppress the specific ChainedAssignmentError warning from pandas
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    module="pandas.io.spss"
)


######################################################################
### Llama 2 Setup
######################################################################
# # print(torch.cuda.is_available())
llama_model= "meta-llama/Llama-3.1-8B-Instruct"

# # when setting possible enironment variables in the future
MODEL_ID = os.environ.get("LLAMA_ID", llama_model)
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", None)


DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
# DTYPE_STR = "bfloat16" if torch.cuda.is_available() else "float32"

# set seeds for reproducibility
SEED = 1234
# os.environ["PYTHONHASHSEED"] = str(SEED)   # best set before Python starts                # if you still use np.random.*
set_seed(SEED)                              # seeds Python, NumPy, Torch (HF helper)

# setyp initial llm
tokenizer = AutoTokenizer.from_pretrained(
                            MODEL_ID,  
                            cache_dir=CACHE_DIR, 
                            use_fast=True, 
                            # local_files_only=True
                            )
tokenizer.padding_side = "left"

# trying this
os.environ["VLLM_BATCH_INVARIANT"] = "1"

# Ensure a pad token exists (prevents fallback messages)
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

def get_llm():
    """Set up the vLLM engine."""
    print(f"Loading vLLM model: {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",         
        trust_remote_code=True,
        # tensor_parallel_size=1, # Use >1 if you have multiple GPUs
        gpu_memory_utilization=0.90, # Reserve 90% of GPU for vLLM
        seed=SEED,
    )
    return llm

def build_network(args, personas, well_being, depressed_personas=None):
    '''Build network based on given arguments.
    Args:
        args: Argument namespace containing network parameters.
        personas: List of personas for agents.
        well_being: List of well-being scores for agents.
        depressed_personas: List of depressed personas for agents.
    Returns:
        network: Generated network object.
    '''
    # if args.net == "sf":
    #     return ScaleFreeNetwork(
    #         m=args.m,
    #         num_agents=args.num_agents,
    #         seed=args.seed,
    #         well_being= well_being,
    #         personas=personas,
    #         depressed_personas=depressed_personas,
    #         directed=args.directed,
    #     )
    
    if args.net == "sda" or args.net == "sdc":
        return SocialDistanceAttachment(
            alpha=args.alpha,
            degree=args.degree,
            dim=args.dim,
            num_agents=args.num_agents,
            seed=args.seed,
            plot=False,
            well_being= well_being,
            personas=personas,
            sdc=(args.net == "sdc"),
            depressed_personas=depressed_personas,
            directed=args.directed,
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
            directed=args.directed,
        )

def generate_parser():
    "parse all given arguments"
    parser = argparse.ArgumentParser(description="Run LLM agent simulation.")

    # Network Settings
    parser.add_argument("net", nargs="?", choices=["sf", "r", "sda", "sdc"], default="r", help="Network type: sf=ScaleFree, r=Random, sda=SocialDistanceAttachment")
    parser.add_argument("--rounds", type=int, default=0, help="Number of update rounds")
    parser.add_argument("--num_agents", type=int, default=10, help="Total number of agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="List of seeds to run (e.g., --seeds 42 43 44)")
    parser.add_argument("--directed", action="store_true", help="Whether the network is directed")

    #test args
    parser.add_argument("--test_llms", action="store_true", help="Test LLMs on PHQ-9 questionnaire")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter for LLM")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature sampling parameter for LLM")
    parser.add_argument("--check_point", type=int, default=10, help="Checkpoint for PHQ-9 questionnaire")
    
    # Scale-free specific
    # parser.add_argument("--m", type=int, default=2, help="Edges per new node (scale-free)")

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

    # specify if to save network properties after simulation
    parser.add_argument("--save", action="store_true", help="Save network properties after simulation")

    # Load existing network
    parser.add_argument("--use_saved_network",
                        nargs="?",           
                        const=0,             
                        type=int,             
                        help=(
                            "Use saved network properties to reload network. "
                            "If provided without a value, just load the network. "
                            "If provided with an integer, run that many extra rounds."
                        ))

    parser.add_argument("--log", 
                        nargs="?",
                        const=30, 
                        type =int,
                        help="Log network state every N iterations (default: 30)")
    return parser.parse_args()



def update_network(network, 
                    pipe, 
                    fracs_dist_step = [], 
                    running_fracs = [], 
                    rounds=1, 
                    seed=42, 
                    enforce_ngrams = False, 
                    log = False):
    """Update the network for one round and return the mean fraction of distorted tweets."""

    # only enforce n-grams if specified
    if enforce_ngrams:
        distorted_tweets = lp.load_distorted_tweets("data/distorted_tweets.csv", numtweets=1000, seed=seed)
    else:
        distorted_tweets = []
    
    log_path = None
    log_iteration = log if isinstance(log, int) else 30
    
    n_grams = metrics.load_ngrams_tsv("data/distorted_language_ngrams.tsv")
    
    for _ in range(rounds):
        set_seed(seed + network.iterations)  # change seed each round for variability
        mean_running_frac, frac_distorted_this_step = network.update_round(tokenizer, 
                                                                           pipe, 
                                                                           n_grams=n_grams, 
                                                                           distorted_tweets=distorted_tweets, 
                                                                           time_info=False, 
                                                                           check_point=5)
        print(f"Round {network.iterations}: Mean running fraction of distorted agents: {mean_running_frac:.4f}, Fraction distorted this step: {frac_distorted_this_step:.4f} ")
        running_fracs.append(mean_running_frac)
        fracs_dist_step.append(frac_distorted_this_step)
        if network.iterations % 10 == 0:
            print(f"finished round {network.iterations}")
        
        if log and network.iterations % log_iteration == 0: 
            new_dir = ri.log_network_state(network, seed, running_fracs, fracs_dist_step)
            # cleanup old log
            if log_path is not None:
                os.remove(log_path)  
                os.rmdir(os.path.dirname(log_path))

            log_path = new_dir

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
    # set_seed(args.seed)     
    # print(type(pipe.model))
    # print("GENERATOR: ", inspect.signature(pipe.model.generate))

    # build an argparse-like namespace
    
    personas = None
    # load personas
    if True:
        personas = lp.load_personas_from_file("data/personas_short_10k.csv", args.num_agents, seed=args.seed)
    
    well_being = lp.load_phq9("data/confidential/phq9.sav", args.num_agents, seed=args.seed)

    # only load depressed personas if specified
    if args.depressed:
        depressed_personas = lp.load_depressed_personas("data/depressed.csv", personass_to_load=1, seed=args.seed)
    else:
        depressed_personas = None

    # build network
    network = build_network(args, 
                            well_being=well_being, 
                            personas=personas, 
                            depressed_personas=depressed_personas)
    # run updates
    if args.rounds == 0:
        return network, [], []
    
     # run updates
    running_fracs, network, fracs_dist_step = update_network(network, 
                                                             pipe=pipe, 
                                                             fracs_dist_step=[], 
                                                             running_fracs=[], 
                                                             rounds=args.rounds,
                                                             seed=args.seed, 
                                                             enforce_ngrams=args.enforce_ngrams, 
                                                             log = args.log)
    return network, running_fracs, fracs_dist_step

def update_existing_network(pipe, args, network, running_fracs=[], fracs_dist_step=[]):
    '''Update an existing network from a file path.
    Args:
        pipe: LLM pipeline for network generation.
        args: Argument namespace containing network parameters.
    Returns:
        network: Updated network object.
        running_fracs: List of running fractions of distorted tweets.
        fracs_dist_step: List of fractions of distorted tweets per step.
    '''
    # reload network from saved properties

    # network, running_fracs, fracs_dist_step= ri.generate_network(args, pipe)
    running_fracs, network, fracs_dist_step = update_network(network, 
                                                             pipe=pipe, 
                                                             fracs_dist_step=fracs_dist_step, 
                                                             running_fracs=running_fracs, 
                                                             rounds=args.rounds, 
                                                             seed=args.seed, 
                                                             enforce_ngrams=args.enforce_ngrams)

    # tweet_history = [(a.ID, a.tweethistory) for a in network.all_agents]
    if args.save:
        file_output_path = ri.read_out_network_properties(network, 
                                                          args.seed, 
                                                          fracs_dist_step, 
                                                          running_fracs)
        print(f"Network properties saved to {file_output_path}")
    return network, running_fracs, fracs_dist_step


def generate_new_net(args, pipe):
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
        file_output_path = ri.read_out_network_properties(network, 
                                                          args.seed, 
                                                          fracs_dist_step, 
                                                          running_fracs)
        print(f"Network properties saved to {file_output_path}")

    return network, running_fracs, fracs_dist_step


def call_visualizations(network, path, filename, args, running_fracs, fracs_dist_step): 
    """Call visualization functions for the given network.
    Args:
        network: Network object to visualize.
        path_manager (PathManager): Instance of PathManager for directory management.
        args: Argument namespace containing visualization parameters.
        running_fracs: List of running fractions of distorted tweets.
        fracs_dist_step: List of fractions of distorted tweets per step.
    """
    # path = path_manager.get_run_directory(is_plot=True)
    vis.print_subnetworks_phq9(network, path, filename, save=args.save, show_fig=True)

    print("Saving visualizations to directory:", path, "filename:", filename )

    # Frequency of tweeting
    tweet_histories = metrics.obtain_tweet_histories([network])
    mean_var_freqs = metrics.calculate_tweet_frequency_stats(tweet_histories)
    
    vis.plot_tweet_frequency(mean_var_freqs['mean'], mean_var_freqs['variance'], 5, path, filename, save=args.save)
    vis.distorted_info(network.cds_info, path, filename, save=args.save)
    vis.plot_running_fracs(running_fracs, path, filename, save=args.save)
    vis.plot_distorted_fracs(fracs_dist_step, path, filename, save=args.save)


def pca_visualize(all_networks_results, path, filename, args):
    """Perform PCA visualization on TF-IDF results across different network states.

    Args:
        all_networks_results (list): List of tuples containing (state, network) pairs.      
        path_manager (PathManager): Instance of PathManager for directory management.
        args: Argument namespace containing visualization parameters.
    """
    n_grams = metrics.load_ngrams_tsv("data/distorted_language_ngrams.tsv")
    path = path_manager.get_run_directory(is_plot=True)
    # wrapper dealing with multiple networks per setting

    sbert = True
    mentalbert = True
    shift = 10
    num_steps = 30
    n_components = 2

    # still need to process variance better instead of taking var between runs 
    if sbert:
        mean_embedding_per_setting, var_embedding_per_setting, all_mats_per_setting = metrics.sbert_for_runs(all_networks_results, 
                                                      num_steps=num_steps, 
                                                      shift=shift, mentalbert=mentalbert)
    else:
        mean_embedding_per_setting, all_mats_per_setting = metrics.tf_idf_for_runs(all_networks_results, 
                                                                               num_steps=num_steps, 
                                                                               shift=shift, 
                                                                               n_grams=n_grams)
    
    mean_traj, pca = metrics.pca_on_means(mean_embedding_per_setting, n_components=n_components)
    std_traj, _ = metrics.traj_variance_in_pca_space(all_mats_per_setting, pca)

    # vis.plot_tf_idf_PCA(mean_traj, std_traj, num_steps=100, shift=5, save= args.save)
    vis.plot_embedding_PCA_runs(mean_traj, std_traj, num_steps=num_steps, shift=shift, sbert=sbert,  save= args.save, path=path, filename=filename)


def main(args, pipe, states):

    all_networks_results = {}
    for seed in args.seeds:
        args.seed = seed
        
        # set seed when loading in the network. 
        set_seed(seed)
        
        # loop over all states
        for state in states:
            args.enforce_ngrams = (state == "enforced_ngrams")
            args.depressed = (state == "depressed")

            # load in existing network if specified
            if args.use_saved_network is not None:
                print(f"Loading network for state '{state}' and seed {seed}...\n")

                # load in existing network and update if specified
                network, running_fracs, fracs_dist_step= ri.generate_network(args, pipe)
                if args.use_saved_network > 0:
                    print(f"Updating loaded network for {args.use_saved_network} rounds...\n")
                    args.rounds = args.use_saved_network
                    network, running_fracs, fracs_dist_step = update_existing_network(pipe, args, network, running_fracs, fracs_dist_step)
            else:
                print(f"Generating new network for state '{state}' and seed {seed}...\n")
                network, running_fracs, fracs_dist_step = generate_new_net(args, pipe)
            
            # Collect result
            all_networks_results.setdefault(state, []).append({
            "network": network,
            "running_fracs": running_fracs,
            "fracs_dist_step": fracs_dist_step
            })
    return all_networks_results

def test_llms(args, pipe):

    well_being = lp.load_phq9("data/confidential/phq9.sav", args.num_agents, seed=args.seed)
    for i in range(len(well_being)):
        well_being[i]["phq9_sumscore"] = 0

     # load personas
    personas = lp.load_personas_from_file("data/personas_short_10k.csv", args.num_agents, seed=args.seed)
    tester = TestLLMs(well_being=well_being, num_agents=args.num_agents, seed=args.seed, personas=personas)

    temp = args.temp
    top_p = args.top_p
    checkpoint = args.check_point

    rounds = checkpoint *28 + 1 
    
    all_bias, bias_per_phq9, accuracy_per_phq9, all_accuracy = tester.run_simulation(tokenizer=tokenizer, 
                                                                                     pipe=pipe, 
                                                                                     n_rounds=rounds, 
                                                                                     check_point=checkpoint,
                                                                                     temp=temp,
                                                                                     top_p=top_p)

    directory_for_test = f"plots/test/temp_{temp}_top_p_{top_p}_cp_{checkpoint}"
    if not os.path.exists(directory_for_test):
        os.makedirs(directory_for_test)
    vis.plot_bias(bias_per_phq9, all_bias, directory_for_test)
    vis.plot_accuracy(accuracy_per_phq9, all_accuracy, directory_for_test)
    
    return all_accuracy, accuracy_per_phq9


if __name__ == "__main__":


    # VLLM
    pipe = get_llm()
    args = generate_parser()

    if args.depressed:
        states = ["depressed"]
    elif args.enforce_ngrams:
        states = ["enforced_ngrams"]
    else:
        states = ["basis"]
    testje = True

    #experiment
    # states = ["basis", "depressed", "enforce_ngrams"]
    if args.test_llms:
        all_accuracy, accuracy_phq9 = test_llms(args, pipe)
        sys.exit(0)

    print("\n")
    print("="*40)
    print("Starting simulation")
    print("="*40)

    start_time = time.perf_counter()

    #experiment
    # states = ["basis", "depressed", "enforced_ngrams"]
    
    # call main simulation
    all_networks_results= main(args, pipe, states)
    
    # final time 
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Simulation finished in {elapsed_time:.4f} seconds ({elapsed_time/60:.2f} minutes).")

    degree_weighted_phq9 = []

    # analyze one of the networks
    for i in range(len(args.seeds)):
        network_data = all_networks_results[states[0]][i]

        network = network_data["network"]
        running_fracs = network_data["running_fracs"]
        fracs_dist_step = network_data["fracs_dist_step"]
        path_manager = PathManager(network=network)
        
        #Maybe bit ugly here but aggregate degree weighted phq9 scores
        deg_w_phq9 = metrics.degree_weighted_mean(network)
        degree_weighted_phq9.append(deg_w_phq9)

        # Get paths
        data_path = path_manager.get_run_directory(is_plot=False)
        plot_path = path_manager.get_run_directory(is_plot=True)
        data_filename = path_manager.get_network_filename()
        plot_filename = path_manager.get_plot_name()
        
        print("\nVisualizing results for first network...")
        # Clean tweet histories
        metrics.print_histories(network, file_dir = data_path, file_name = data_filename, save=args.save)

        # Visualizations
        call_visualizations(network, plot_path, plot_filename, args, running_fracs, fracs_dist_step)

    vis.plot_degree_weighted_phq9(np.array(degree_weighted_phq9), plot_path, plot_filename, save=args.save)
    print("End of model run.")
        
    #PCA
    pca_visualize(all_networks_results, plot_path, plot_filename, args)
    sys.exit(0)

 
    


