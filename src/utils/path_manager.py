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
        if "ScaleFree" in str(type(network)): return "sf"
        if "SocialDistanceAttachment" in str(type(network)) and getattr(network, 'sdc', False): return "sdc"
        return "sda"

    def get_run_directory(self, is_plot=False):
        """Returns the folder path: data/networks/{state}/{type}/{params}/"""
        base = self.base_plots if is_plot else self.base_data
        path = base / self.state / self.net_type /self.directed/ self.params
        path.mkdir(parents=True, exist_ok=True) # Automatically create folders
        return path

    def get_network_filename(self):
        """Returns the standard filename for the network text file."""
        return f"num_agents{self.num_agents}_{self.rounds}_net_{self.seed}.txt"

    def get_full_network_path(self):
        return self.get_run_directory(is_plot=False) / self.get_network_filename()
    
    def get_plot_name(self):
        """Returns a standard filename for plots."""
        return f"_num_agents{self.num_agents}_{self.rounds}_net_{self.seed}.png"