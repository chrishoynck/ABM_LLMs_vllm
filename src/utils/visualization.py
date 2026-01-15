import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def print_network(network, path="", filename="default.png", save=False):
    """
    Print network at one single iteration

    Args:
        network: The network object to visualize.
    """
    print("starting to visualize network...")
    color_map = ['lightblue'] * len(network.all_agents)
    graph = nx.Graph()


    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    
    # Set positions and draw the graph
    plt.figure(figsize=(16,8))
    pos = nx.kamada_kawai_layout(graph, scale=0.6)
    nx.draw(
        graph,
        pos,
        node_color=color_map,
        with_labels=True,
        edge_color="lightgray",
        width=0.2,
        node_size=400,
        font_size=10,
    )
    if save:
        plt.savefig(f"{path}/network_snapshot_{filename}.png", dpi=300)
    # plt.show()
    plt.close()
    return 

def print_network_phq9(network, path="", filename="default.png", save=False):
    """
    Print network at one single iteration

    Args:
        network: The network object to visualize.
    """
    # Create color map based on PHQ-9 scores
    node_colors = []

    if network.directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    for agent in network.all_agents:
        if agent.well_being and "phq9_sumscore" in agent.well_being:
            score = agent.well_being["phq9_sumscore"]
            
            # Normalize score (0-27) to 0-1 range for colormap
            normalized_score = min(max(score / 27.0, 0.0), 1.0)
            node_colors.append(normalized_score)
            graph.add_node(agent.ID, mood=score) 
        else:
            print(f"Agent {agent.ID} has no PHQ-9 score.")
            # Default color (e.g., light blue) if no score is available
            node_colors.append(0.0) # Map 0 to green/low score color
            graph.add_node(agent.ID, mood=None)

    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    
    try: 
        assortativity = nx.numeric_assortativity_coefficient(graph, 'mood')
        print(f"PHQ-9 assortativity: {assortativity}")
    except Exception as e:
        print(f"Could not compute assortativity: {e}")

    
    # Set positions and draw the graph
    plt.figure(figsize=(6,6))
    pos = nx.kamada_kawai_layout(graph, scale=0.6)
    
    # Use a colormap from green (low score) to red (high score)
    cmap = plt.cm.RdYlGn_r 
    ax = plt.gca()
    if len(network.all_agents) <= 50:
        font_size = 10
        show_labels = True
        node_size = 400
    else:
        font_size = max(2, 400 // len(network.all_agents))
        node_size = max(20, 40000 // len(network.all_agents))
        show_labels = False

    nx.draw(
        graph,
        pos,
        node_color=node_colors,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        with_labels=show_labels,
        edge_color="lightgray",
        width=1,
        node_size=node_size,
        font_size= font_size,
    )
    
    # Add a colorbar to indicate the scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=27))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='PHQ-9 Score')
    
    if save:
        plt.savefig(f"{path}/network_snapshot_phq9_{filename}.png", dpi=300)
    # plt.show()
    plt.close()
    return graph


def distorted_info(cds_info, path="", filename="default.png", save=False):
    '''
    This function bins fractions of distorted neighbors, and plots the probability corresponding to that to tweet.
    Args:
        cds_info(List(Tuple)): List of tuples with cds_frac 
    '''
    cds_info = np.array(cds_info)
    cds_frac = cds_info[:, 0]
    tweeted = cds_info[:, 1]
    distorted = cds_info[:, 2]

    # divide fraction of neighbors having cds in previous tweets in bins. 
    bins = 10
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_idx = np.digitize(cds_frac, bin_edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1) # make sure cds_frac == 1 also gets bin

    tweet_prob = []
    distorted_prob = []
    bin_centers = []
    for i in range(bins):
        if i not in bin_idx:
            tweet_prob.append(np.nan)
            distorted_prob.append(np.nan)
        else:
            frac_tweeted = np.mean(tweeted[bin_idx==i])
            frac_distorted = np.mean(distorted[bin_idx==i])
            tweet_prob.append(frac_tweeted)
            distorted_prob.append(frac_distorted)
        bin_centers.append(0.5 * (bin_edges[i] + bin_edges[i+1]))

    width = bin_edges[1] - bin_edges[0] 

    plt.bar(bin_centers, tweet_prob, width=width, align="center", alpha=0.5, edgecolor="black", label="tweet prob")
    plt.bar(bin_centers, distorted_prob, width=width, align="center", alpha=0.5, edgecolor="black", label = "distorted prob")
    plt.xlabel("Fraction of neighbors with CDS (prev round)")
    plt.ylabel("P(tweet)")
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(alpha=0.3, axis="y")
    if save:
        plt.savefig(f"{path}/distorted_info_{filename}.png", dpi=300)
    plt.close()
    # plt.show()

def plot_distorted_fracs(frac_distorted_this_step, 
                         path="", filename="default.png",
                         save=False):
    '''
    This function plots the fraction of distorted tweets per round.
    Args:
        distorted_fracs(List(Float)): List of CDS fractions per round
    '''
    plt.plot(frac_distorted_this_step, marker='o', markersize=2)
    plt.xlabel("Round")
    plt.ylabel("Fraction of distorted tweets")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    if save:    
        plt.savefig(f"{path}/distorted_step_fracs_{filename}.png", dpi=300)
    plt.close()
    
    # plt.show()


def plot_running_fracs(running_fracs, 
                        path="", filename="default.png",
                        save=False):
    '''
    This function plots the running mean fraction of distorted tweets over rounds.
    Args:
        running_fracs(List(Float)): List of running mean fractions over rounds
    '''
    plt.plot(running_fracs, marker='o')
    plt.xlabel("Round")
    plt.ylabel("Mean fraction of distorted active tweets (all agents)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)

    if not os.path.exists(path):
        os.makedirs(path)
    if save:
        plt.savefig(f"{path}/running_fracs_{filename}.png", dpi=300)
    plt.close()
    # plt.show()

def plot_tf_idf_PCA(reduced_runs, 
                    states, 
                    num_steps=100, 
                    shift=5, 
                    path="", 
                    filename="default.png",
                    save=False):
    '''
    This function plots the PCA-reduced TF-IDF data.
    Args:
        reduced_data (np.array): 2D array with reduced TF-IDF data.
        network: The network object.
        num_steps (int): Number of steps used in TF-IDF retrieval.
        shift (int): Shift used in TF-IDF retrieval.
    '''
    plt.figure(figsize=(3, 3))
    plt.title(f'TF-IDF PCA \n (window size={num_steps}, shift={shift})')
    for i, reduced in enumerate(reduced_runs):
        print("shape reduced:", reduced.shape)
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=10, label=states[i])
        plt.plot(reduced[:, 0], reduced[:, 1], alpha=0.4)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    if save:
        plt.savefig(f"{path}/tf_idf_pca_window{num_steps}_shift{shift}_{filename}.png", bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


def plot_tf_idf_PCA_runs(mean_traj,
                        std_traj=None, 
                        num_steps=100, 
                        shift=5, 
                        path="",
                        filename="default.png",
                        save=False):
    """
    Plot PCA-reduced TF-IDF trajectories.

    Args:
        mean_traj: dict[setting] -> array (T, 2)
        std_traj:  dict[setting] -> array (T, 2), optional
        num_steps: TF-IDF window size
        shift:     TF-IDF shift
        save:      whether to save the figure
    """
    plt.figure(figsize=(3, 3))
    plt.title(f"TF-IDF PCA\n(window={num_steps}, shift={shift})")

    for setting, traj in mean_traj.items():
        traj = np.asarray(traj)  # (T, 2)

        if std_traj is not None and setting in std_traj:
            std = np.asarray(std_traj[setting])           # (T, 2)
            std_norm = np.linalg.norm(std, axis=1)        # (T,)
            # scale marker size a bit with std (optional)
            s = 5 + 20 * (std_norm / (std_norm.max() + 1e-8))
        else:
            s = 10

        plt.scatter(traj[:, 0], traj[:, 1], s=s, alpha=0.7, label=setting)
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.5)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    # plt.show()
    if save:
        plt.savefig(f"{path}/tf_idf_pca_runs{num_steps}_shift{shift}_{len(mean_traj)}settings_{filename}.png", bbox_inches='tight', dpi=300) 
    plt.close()

def check_degree_distribution(unique_degrees, frequencies):
    """
    Plot the degree distribution on a log-log scale.
    Args:
        unique_degrees (list of int): Unique degrees in the network.
        frequencies (list of int): Frequencies corresponding to each degree.
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(unique_degrees, frequencies, 'bo')
    plt.title('Degree Distribution (Log-Log Scale)')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    # plt.show()
    plt.close()

def plot_tweet_frequency(mean_freqs, var_freqs, window_size=5, file_path="", filename="default.png", save=False ):
    """
    Plot the mean tweet frequency over time with variance as a shaded region.

    Args:
        mean_freqs (list of float): Mean tweet frequency over time.
        var_freqs (list of float): Variance of tweet frequency over time.
        window_size (int): The size of the sliding window used for calculation.
        save_path (str, optional): Path to save the plot.
    """
    rounds = range(len(mean_freqs))
    std_devs = np.sqrt(var_freqs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, mean_freqs, label='Mean Frequency', color='blue')
    plt.fill_between(rounds, 
                     np.array(mean_freqs) - std_devs, 
                     np.array(mean_freqs) + std_devs, 
                     color='blue', alpha=0.2, label='Standard Deviation')
    
    plt.title(f'Tweet Frequency Over Time (Window Size: {window_size})')
    plt.xlabel('Round')
    plt.ylabel('Frequency')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save:
        plt.savefig(f"{file_path}/tweet_freq_window{window_size}_{filename}.png", dpi=300)
    # plt.show()
    plt.close()