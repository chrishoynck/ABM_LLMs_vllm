import os
from pathlib import Path
from utils.tools.format_config import FC


def _bias_on_from_path(bias_table_path):
    """Decide whether a run is bias-corrected ("debiased") from a bias_table_path.

    Mirrors the gate in classes/network.py: a missing / "none" / "off" / empty
    value (or a path that doesn't point at an existing file) means the run is
    UNcorrected, so it lands under non_debiased/. Anything pointing at a real
    table is debiased/. Kept in sync with network.py so the read path (args) and
    the write path (network) resolve to the same directory.
    """
    if bias_table_path is None:
        return False
    if isinstance(bias_table_path, str) and bias_table_path.strip().lower() in ("none", "off", ""):
        return False
    return os.path.isfile(bias_table_path)


class PathManager:
    def __init__(self, args=None, network=None):
        """
        Initialize with either parsed args or an existing network object.
        """
        self.base_data = Path(f"data/networks{FC.DIR_SUFFIX}")
        
        # Extract parameters from args OR network
        if args:
            self.net_type = args.net
            self.params = self._get_params_from_args(args)
            self.state = self._get_state(args.enforce_ngrams, args.happy)
            self.num_agents = args.num_agents
            self.seed = args.seed
            self.directed = "directed" if args.directed else "undirected"
            self.rounds = args.rounds
            self.sample_phq9 = getattr(args, 'sample_phq9', None)
            self.cap_phq9 = getattr(args, 'cap_phq9', False)
            self.phq9_threshold = getattr(args, 'phq9_threshold', 0)
            self.init_phq9_zero = getattr(args, 'init_phq9_zero', False)
            # Bias correction is requested via the bias_table_path flag.
            self.bias_corrected = _bias_on_from_path(getattr(args, 'bias_table_path', None))

        elif network:
            self.net_type = self._infer_net_type(network)
            self.params = self._get_params_from_net(network)
            self.state = network.state # Default, or need to store state in network obj
            self.num_agents = len(network.all_agents)
            self.seed = network.seed # Assuming seed is stored
            self.rounds = network.iterations # Or initial rounds
            self.directed = "directed" if (hasattr(network, 'directed') and network.directed) else "undirected"
            self.sample_phq9 = getattr(network, 'sample_phq9', None)
            self.cap_phq9 = getattr(network, 'cap_phq9', False)
            self.phq9_threshold = getattr(network, 'phq9_threshold', 0)
            self.init_phq9_zero = getattr(network, 'init_phq9_zero', False)
            # A loaded bias table on the network means the run was debiased.
            self.bias_corrected = getattr(network, '_phq9_bias_table', None) is not None

        self.subparams = self._get_subparams()
        self.debias_dir = "debiased" if self.bias_corrected else "non_debiased"
        self.phq9_mode = self._get_phq9_mode()
        
    def _get_state(self, enforce_ngrams, happy):
        if enforce_ngrams: return "enforced_ngrams"
        if happy: return "happy"
        return "basis"

    def _get_params_from_args(self, args):
        if args.net == "sf": return f"{args.m}"
        if args.net == "r": return f"{str(args.p).replace('.', '_')}"
        if args.net in ["sda", "sdc"]:
            # {:g} drops trailing .0 so integer degrees keep their old dirnames (d6, not d6.0)
            deg = f"{args.degree:g}".replace('.', '_')
            return f"{str(args.alpha).replace('.', '_')}_d{deg}_dim{args.dim}"
        return "unknown"


    def _get_subparams(self):
        """Parameter part of path without seed (parent folder for all seeds)."""
        return f"rounds{self.rounds}_N{self.num_agents}"

    def _get_phq9_mode(self):
        """Build a subdirectory name for non-default PHQ-9 options (sampling, cap, threshold).
        Returns None when all options are at their defaults (standard checkpoint behaviour)."""
        parts = []
        if self.sample_phq9:
            parts.append(f"sample{str(self.sample_phq9).replace('.', '_')}")
        if self.cap_phq9:
            parts.append("cap")
        if self.phq9_threshold and self.phq9_threshold > 0:
            parts.append(f"thr{str(self.phq9_threshold).replace('.', '_')}")
        if getattr(self, 'init_phq9_zero', False):
            parts.append("init_0")
        return "_".join(parts) if parts else None
    
    def _get_params_from_net(self, network):
        # Logic to extract m/p/alpha from network object
        if hasattr(network, 'm'): return f"{network.m}"
        if hasattr(network, 'p'): return f"{str(network.p).replace('.', '_')}"
        
        # Add SDA/SDC logic here if those attributes exist on network
        if hasattr(network, 'alpha') and hasattr(network, 'degree') and hasattr(network, 'dim'):
            deg = f"{network.degree:g}".replace('.', '_')
            return f"{str(network.alpha).replace('.', '_')}_d{deg}_dim{network.dim}"
        return "unknown"

    def _infer_net_type(self, network):
        ''' Infer network type from its class name. '''
        if "RandomNetwork" in str(type(network)): return "r"
        # if "ScaleFree" in str(type(network)): return "sf"
        if "SocialDistanceAttachment" in str(type(network)) and getattr(network, 'sdc', False): return "sdc"
        return "sda"

    def _get_parent_directory(self):
        """Parent directory (parameters only, no seed):
        .../{directed}/{debiased|non_debiased}/{params}/rounds{N}_N{agents}/[phq9_mode]/

        The debiased/non_debiased level keeps bias-corrected runs from overwriting
        the uncorrected ones at the same parameters (see _bias_on_from_path)."""
        path = (self.base_data / self.state / self.net_type / self.directed
                / self.debias_dir / self.params / self.subparams)
        if self.phq9_mode:
            path = path / self.phq9_mode
        return path

    def get_run_directory(self, is_plot=False):
        """Run-specific directory. Plots go into a plots/ subfolder of the data run dir."""
        path = self._get_parent_directory() / f"seed_{self.seed}"
        if is_plot:
            path = path / "plots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_aggregate_directory(self, is_plot=False):
        """Parent directory for aggregate (over-runs) plots/data. Creates dir if needed."""
        path = self._get_parent_directory()
        if is_plot:
            path = path / "plots"
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

    def __init__(self, model_name, temp, top_p, check_point, seed, interaction=False):
        self.base_data = Path(f"data/test{FC.DIR_SUFFIX}")
        self.base_plots = Path(f"plots/test{FC.DIR_SUFFIX}")
        self.results_csv = self.base_data / "results.csv"

        # Sanitize model name for filesystem use
        self.safe_name = model_name.replace("/", "_").replace("\\", "_")
        self.model_name = model_name
        self.temp = temp
        self.top_p = top_p
        self.check_point = check_point
        self.seed = seed
        self.interaction = "inter" if interaction else "no_inter"

    def _settings_dir(self):
        """Settings part of the path: {model}/temp_{t}_top_p_{p}_cp_{cp}"""
        return f"{self.safe_name}/temp_{self.temp}_top_p_{self.top_p}_cp_{self.check_point}_{self.interaction}"

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
        """Path for the per-agent tweets-with-PHQ-9 CSV (per-seed)."""
        return self.get_run_directory(is_plot=False) / "tweets_with_phq9.csv"

    def get_results_csv_path(self):
        """Path to the shared results CSV."""
        return self.results_csv