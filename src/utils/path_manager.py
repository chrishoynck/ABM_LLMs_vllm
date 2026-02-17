import os
from pathlib import Path

class PathManager:
    def __init__(self, args=None, network=None):
        """
        Initialize with either parsed args or an existing network object.
        """
        self.base_data = Path("data/networks")
        self.base_plots = Path("plots/networks")
        
        # Extract parameters from args OR network
        if args:
            self.net_type = args.net
            self.params = self._get_params_from_args(args)
            self.state = self._get_state(args.enforce_ngrams, args.depressed)
            self.num_agents = args.num_agents
            self.seed = args.seed
            self.directed = "directed" if args.directed else "undirected"
            self.rounds = args.rounds


        elif network:
            self.net_type = self._infer_net_type(network)
            self.params = self._get_params_from_net(network)
            self.state = network.state # Default, or need to store state in network obj
            self.num_agents = len(network.all_agents)
            self.seed = network.seed # Assuming seed is stored
            self.rounds = network.iterations # Or initial rounds
            self.directed = "directed" if (hasattr(network, 'directed') and network.directed) else "undirected"
        
        self.subparams = self._get_subparams() 
        
    def _get_state(self, enforce_ngrams, depressed):
        if enforce_ngrams: return "enforced_ngrams"
        if depressed: return "depressed"
        return "basis"

    def _get_params_from_args(self, args):
        if args.net == "sf": return f"{args.m}"
        if args.net == "r": return f"{str(args.p).replace('.', '_')}"
        if args.net in ["sda", "sdc"]: 
            return f"{str(args.alpha).replace('.', '_')}_d{args.degree}_dim{args.dim}"
        return "unknown"


    def _get_subparams(self):
        """Parameter part of path without seed (parent folder for all seeds)."""
        return f"rounds{self.rounds}_N{self.num_agents}"
    
    def _get_params_from_net(self, network):
        # Logic to extract m/p/alpha from network object
        if hasattr(network, 'm'): return f"{network.m}"
        if hasattr(network, 'p'): return f"{str(network.p).replace('.', '_')}"
        
        # Add SDA/SDC logic here if those attributes exist on network
        if hasattr(network, 'alpha') and hasattr(network, 'degree') and hasattr(network, 'dim'):
            return f"{str(network.alpha).replace('.', '_')}_d{network.degree}_dim{network.dim}"
        return "unknown"

    def _infer_net_type(self, network):
        ''' Infer network type from its class name. '''
        if "RandomNetwork" in str(type(network)): return "r"
        # if "ScaleFree" in str(type(network)): return "sf"
        if "SocialDistanceAttachment" in str(type(network)) and getattr(network, 'sdc', False): return "sdc"
        return "sda"

    def _get_parent_directory(self, is_plot=False):
        """Parent directory (parameters only, no seed): .../rounds{N}_N{agents}/"""
        base = self.base_plots if is_plot else self.base_data
        return base / self.state / self.net_type / self.directed / self.params / self.subparams

    def get_run_directory(self, is_plot=False):
        """Run-specific directory: .../rounds{N}_N{agents}/seed_{seed}/"""
        path = self._get_parent_directory(is_plot=is_plot) / f"seed_{self.seed}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_aggregate_directory(self, is_plot=False):
        """Parent directory for aggregate (over-runs) plots. Creates dir if needed."""
        path = self._get_parent_directory(is_plot=is_plot)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_network_filename(self):
        """Returns the standard filename for the network text file."""
        return f"net.json"

    def get_full_network_path(self):
        return self.get_run_directory(is_plot=False) / self.get_network_filename()
    
    def get_plot_name(self):
        """Returns filename suffix for run-specific plots (vis code appends .png)."""
        return f"_{self.seed}"

    def get_aggregate_plot_name(self, seeds):
        """Returns filename suffix for aggregate plots (vis code appends .png)."""
        seeds_str = "_".join(map(str, seeds))
        return f"_seeds_{seeds_str}"


class TestPathManager:
    """Path manager for LLM test runs (bias/error plots, tweets, results CSV)."""

    def __init__(self, model_name, temp, top_p, check_point, seed):
        self.base_data = Path("data/test")
        self.base_plots = Path("plots/test")
        self.results_csv = self.base_data / "results.csv"

        # Sanitize model name for filesystem use
        self.safe_name = model_name.replace("/", "_").replace("\\", "_")
        self.model_name = model_name
        self.temp = temp
        self.top_p = top_p
        self.check_point = check_point
        self.seed = seed

    def _settings_dir(self):
        """Settings part of the path: {model}/temp_{t}_top_p_{p}_cp_{cp}"""
        return f"{self.safe_name}/temp_{self.temp}_top_p_{self.top_p}_cp_{self.check_point}"

    def get_run_directory(self, is_plot=False):
        """Per-seed directory: .../seed_{seed}/"""
        base = self.base_plots if is_plot else self.base_data
        path = base / self._settings_dir() / f"seed_{self.seed}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_aggregate_directory(self, is_plot=False):
        """Parent (seed-independent) directory for combined results."""
        base = self.base_plots if is_plot else self.base_data
        path = base / self._settings_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_tweets_path(self):
        """Path for the tweets-with-PHQ-9 text file (per-seed)."""
        return self.get_run_directory(is_plot=False) / "tweets_with_phq9.txt"

    def get_results_csv_path(self):
        """Path to the shared results CSV."""
        return self.results_csv