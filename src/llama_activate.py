
from transformers import AutoTokenizer, set_seed  #, pipeline, BitsAndBytesConfig
import gc
import os, torch
import sys, argparse, time
import numpy as np
import inspect
from vllm import LLM, SamplingParams
import utils.metrics as metrics
from utils.path_manager import PathManager, TestPathManager
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
models = [ "Qwen/Qwen3-4B-Instruct-2507", "google/gemma-3-12b-it", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-14B", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.3-70B-Instruct"]

# Short names for --test_llms_model (optional; full HuggingFace IDs also work)
MODEL_ALIASES = {
    "qwen4": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen14": "Qwen/Qwen3-14B",
    "gemma12": "google/gemma-3-12b-it",
    "llama8": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral7": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama70": "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek": "deepseek"
}

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
# DTYPE_STR = "bfloat16" if torch.cuda.is_available() else "float32"

# set seeds for reproducibility
SEED = 1234
# os.environ["PYTHONHASHSEED"] = str(SEED)   # best set before Python starts                # if you still use np.random.*
set_seed(SEED)                              # seeds Python, NumPy, Torch (HF helper)

# trying this
os.environ["VLLM_BATCH_INVARIANT"] = "1"


def get_tokenizer(model_id=None):
    """Load tokenizer for a model. If model_id is None, use MODEL_ID. Sets padding side and pad token."""
    load_id = model_id if model_id is not None else MODEL_ID
    tok = AutoTokenizer.from_pretrained(
        load_id,
        cache_dir=CACHE_DIR,
        use_fast=True,
    )
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# Default tokenizer for main simulation (and single-model test fallback)
tokenizer = get_tokenizer(MODEL_ID)


def how_many_gpus():
    number_of_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {number_of_gpus}")
    return number_of_gpus

def get_llm(model_id=None):
    """Set up the vLLM engine. If model_id is given, load that model; else use MODEL_ID."""
    load_id = model_id if model_id is not None else MODEL_ID
    print(f"Loading vLLM model: {load_id}...")
    gpus_count = how_many_gpus()
    llm = LLM(
        model=load_id,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=gpus_count,
        gpu_memory_utilization=0.90,
        seed=SEED,
        max_model_len=8192,
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
    parser.add_argument("--test_llms_model", type=str, default=None, metavar="MODEL", help="With --test_llms: run only this model. Use short alias (qwen4, qwen14, gemma12, llama8, mistral7) or full HuggingFace ID. If omitted, test all models.")
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
                    log = False,
                    check_point = 10):
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
                                                                           check_point=check_point)
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
        # for i in range(len(personas)):
        #     print("persona:", personas[i])
    
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
                                                             log = args.log,
                                                             check_point=args.check_point)
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
                                                             enforce_ngrams=args.enforce_ngrams,
                                                             log=args.log,
                                                             check_point=args.check_point)

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
    # wrapper dealing with multiple networks per setting

    sbert = True
    mentalbert = True
    use_umap = True
    shift = 5
    num_steps = 10
    n_components = 2

    # Use mean within-run variance for marker size (not between-run variance)
    if sbert:
        mean_embedding_per_setting, var_embedding_per_setting, all_mats_per_setting, mean_phq9_per_setting, mean_within_var_per_setting = metrics.sbert_for_runs(
            all_networks_results,
            num_steps=num_steps,
            shift=shift,
            mentalbert=mentalbert,
        )
    else:
        mean_embedding_per_setting, all_mats_per_setting, mean_within_var_per_setting = metrics.tf_idf_for_runs(
            all_networks_results,
            num_steps=num_steps,
            shift=shift,
            n_grams=n_grams,
        )
        mean_phq9_per_setting = None

    if use_umap:
        mean_traj, _reducer = metrics.umap_on_means(mean_embedding_per_setting, n_components=n_components)
        reduction = "umap"
    else:
        mean_traj, _ = metrics.pca_on_means(mean_embedding_per_setting, n_components=n_components)
        reduction = "pca"

    vis.plot_embedding_PCA_runs(
        mean_traj,
        mean_within_var_per_setting=mean_within_var_per_setting,
        mean_phq9_per_setting=mean_phq9_per_setting,
        num_steps=num_steps,
        shift=shift,
        sbert=sbert,
        mentalbert=mentalbert if sbert else False,
        reduction=reduction,
        save=args.save,
        path=path,
        filename=filename,
    )


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

def _sanitize_model_name(model_id):
    """Sanitize model ID for use in file paths (e.g. 'org/name-1.0' -> 'org_name-1.0')."""
    return model_id.replace("/", "_").replace("\\", "_")


def test_llms(args, pipe, model_name, tokenizer=None, use_deepseek=False):
    """Run PHQ-9 test for one model. If tokenizer is None, uses the module-level tokenizer."""
    tok = tokenizer if tokenizer is not None else globals()["tokenizer"]

    well_being = lp.load_phq9("data/confidential/phq9.sav", args.num_agents, seed=args.seed)
    for i in range(len(well_being)):
        well_being[i]["phq9_sumscore"] = 0

    personas = lp.load_personas_from_file("data/personas_short_10k.csv", args.num_agents, seed=args.seed)
    tester = TestLLMs(well_being=well_being, num_agents=args.num_agents, seed=args.seed, personas=personas, deepseek=use_deepseek)

    temp = args.temp
    top_p = args.top_p
    checkpoint = args.check_point
    rounds = checkpoint * 28 + 1

    data_dir = "data/test/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if use_deepseek:
        model_name = "deepseek"

    all_bias, bias_per_phq9, mae_per_phq9, total_mae = tester.run_simulation(
        tokenizer=tok,
        pipe=pipe,
        n_rounds=rounds,
        check_point=checkpoint,
        temp=temp,
        top_p=top_p,
        model_name=model_name,
    )

    # Build paths via TestPathManager
    tpm = TestPathManager(model_name, temp, top_p, checkpoint, seed=args.seed)

    # Per-seed bias & error plots
    run_plot_dir = str(tpm.get_run_directory(is_plot=True))
    vis.plot_bias(bias_per_phq9, all_bias, run_plot_dir)
    vis.plot_phq9_error(mae_per_phq9, total_mae, run_plot_dir)

    # Combined plots across all seeds (in parent dir)
    agg_plot_dir = str(tpm.get_aggregate_directory(is_plot=True))
    vis.plot_combined_bias_error(
        csv_path=str(tpm.get_results_csv_path()),
        model_name=model_name,
        temp=temp,
        top_p=top_p,
        check_point=checkpoint,
        directory=agg_plot_dir,
    )

    # Export tweets with PHQ-9 to text file
    tester.export_tweets_with_phq9_txt(
        file_path=str(tpm.get_tweets_path()),
        check_point=checkpoint,
        temp=temp,
        top_p=top_p,
        model_name=model_name,
    )

    return total_mae, mae_per_phq9


def run_llm_tests(args):
    """Run PHQ-9 tests for one model (--test_llms_model) or all models; prints summary and exits."""
    if args.test_llms_model:
        model_id = MODEL_ALIASES.get(args.test_llms_model.strip().lower(), args.test_llms_model.strip())
        models_to_run = [model_id]
        if model_id == "deepseek":
            use_deepseek = True
            models_to_run = [MODEL_ALIASES.get("llama8")]
        else:
            use_deepseek = False
        print(f"Testing single model (--test_llms_model): {model_id}")
    else:
        models_to_run = models
    results = {}
    for model_id in models_to_run:
        print(f"\n{'='*50}\nTesting model: {model_id}\n{'='*50}")
        tok = get_tokenizer(model_id)
        pipe = get_llm(model_id=model_id)
        total_mae, mae_per_phq9 = test_llms(args, pipe, model_id, tokenizer=tok, use_deepseek=use_deepseek)
        results[model_id] = {"total_mae": total_mae, "mae_per_phq9": mae_per_phq9}
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    print("\nTest summary (total PHQ-9 MAE per model; lower is better):")
    for mid, res in results.items():
        print(f"  {mid}: {res['total_mae']:.4f}")
    if len(results) > 1:
        vis.plot_model_comparison_by_settings(csv_path="data/test/results.csv", directory="plots/test", save=args.save)
    sys.exit(0)


if __name__ == "__main__":

    args = generate_parser()

    if args.depressed:
        states = ["depressed"]
    elif args.enforce_ngrams:
        states = ["enforced_ngrams"]
    else:
        states = ["basis"]

    if args.test_llms:
        run_llm_tests(args)

    # Main simulation: load default model once
    pipe = get_llm()

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
        
        print(f"\nVisualizing results for seed {network.seed}...")
        # Clean tweet histories
        metrics.print_histories(network, file_dir = data_path, file_name = data_filename, save=args.save)

        # Visualizations
        call_visualizations(network, plot_path, plot_filename, args, running_fracs, fracs_dist_step)

    # Aggregate plots (over runs): save in parent folder with seeds in filename
    parent_plot_path = path_manager.get_aggregate_directory(is_plot=True)
    aggregate_filename = path_manager.get_aggregate_plot_name(args.seeds)
    vis.plot_degree_weighted_phq9(np.array(degree_weighted_phq9), parent_plot_path, aggregate_filename, save=args.save)
    print("End of model run.")

    # PCA/UMAP over runs
    pca_visualize(all_networks_results, parent_plot_path, aggregate_filename, args)
    del pipe
    sys.exit(0)

 
    


