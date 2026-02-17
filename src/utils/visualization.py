import seaborn as sns
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

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

def print_network_phq9(network, path="", filename="default.png", save=False, show_fig=False):
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
    if show_fig:
        plt.show()
    plt.close()
    return graph


def print_subnetworks_phq9(network, path="", filename="default.png", save=False, show_fig=False):
    """
    Print network at one single iteration

    Args:
        network: The network object to visualize.
    """
    # Create color map based on PHQ-9 scores
    if network.directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    if len(network.all_agents[0].all_phq9_sumscores) == 0:
        graph = print_network_phq9(network=network, path=path, filename=filename, save=save, show_fig=show_fig)
        return graph
    else: 
        for agent in network.all_agents:
            score = agent.all_phq9_sumscores[0]  # Initial score at round 0
            graph.add_node(agent.ID, mood=score) 


    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    
    # Set positions and draw the graph
    fig, axes = plt.subplots(2, 5, figsize=(15,10))
    pos = nx.kamada_kawai_layout(graph, scale=0.6)

    axes = axes.flatten()
    cmap = plt.cm.RdYlGn_r 

    if len(network.all_agents) <= 50:
        font_size = 10
        show_labels = True
        node_size = 400
    else:
        font_size = max(2, 400 // len(network.all_agents))
        node_size = max(20, 40000 // len(network.all_agents))
        show_labels = False

    intervals_phq9 = np.linspace(0, network.iterations-1, 10, dtype=int)
    for i, when_questioned in enumerate(intervals_phq9):
        ax = axes[i]
        current_node_colors = []

        for agent in network.all_agents:
            
            score = agent.all_phq9_sumscores[when_questioned]
            
            # Normalize score (0-27) to 0-1 range for colormap
            normalized_score = min(max(score / 27.0, 0.0), 1.0)
            current_node_colors.append(normalized_score)
            graph.nodes[agent.ID]['mood'] = score

        try: 
            assortativity = nx.numeric_assortativity_coefficient(graph, 'mood')
            print(f"PHQ-9 assortativity: {assortativity}")
        except Exception as e:
            print(f"Could not compute assortativity: {e}")

        # Use a colormap from green (low score) to red (high score)
        nx.draw(
            graph,
            pos,
            ax=ax,
            node_color=current_node_colors,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            with_labels=show_labels,
            edge_color="lightgray",
            width=1,
            node_size=node_size,
            font_size= font_size,
        )
        ax.set_title(f"Round: {str(when_questioned)} Assortativity: {assortativity:.2f}")
        ax.axis('off')
    
    plt.tight_layout()
    fig.subplots_adjust(right=0.9) # Make room for cbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    # Add a colorbar to indicate the scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=27))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='PHQ-9 Score (0-27)')
        
    if save:
        plt.savefig(f"{path}/network_snapshot_phq9_{filename}.png", dpi=300, bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close()
    return graph


def plot_degree_weighted_phq9(degree_weighted_phq9, path="", filename="default.png", save=False):
    '''
    Plot PHQ-9 scores against degree weighted by connection weights.
    Args:
        network: The network object.
    '''
    plt.figure(figsize=(4, 4))
    # plt.plot(degree_weighted_phq9)
    mean_dw_phq9 = np.mean(degree_weighted_phq9, axis=0)
    var_dw_phq9 = np.var(degree_weighted_phq9, axis=0)
    std_dw_phq9 = np.sqrt(var_dw_phq9)
    rounds = range(len(mean_dw_phq9))
    plt.plot(rounds, mean_dw_phq9, label='Mean Degree-weighted PHQ-9', color='blue')
    plt.fill_between(rounds,
                        np.array(mean_dw_phq9) - std_dw_phq9,
                        np.array(mean_dw_phq9) + std_dw_phq9,
                        color='blue', alpha=0.2, label='Standard Deviation')

    plt.xlabel("Round")
    plt.title("Degree-weighted PHQ-9 score")
    plt.grid(alpha=0.3)
    if save:
        plt.savefig(f"{path}/degree_weighted_phq9_{filename}.png", dpi=300)
    plt.close()


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
    plt.plot(frac_distorted_this_step, marker='o', markersize=1, linewidth=0.8)
    plt.xlabel("Round")
    plt.ylabel("Fraction of active tweets distorted (this round)")
    plt.title("Fraction distorted per round (among agents who tweeted)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    if save:    
        plt.savefig(f"{path}/frac_distorted_per_round_{filename}.png", dpi=300)
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
    plt.plot(running_fracs, marker='o', markersize=1, linewidth=0.8)
    plt.xlabel("Round")
    plt.ylabel("Mean (over agents) of distortion rate (last 5 tweets)")
    plt.title("Running mean per-agent distortion rate (5-tweet window)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)

    if not os.path.exists(path):
        os.makedirs(path)
    if save:
        plt.savefig(f"{path}/mean_agent_distortion_rate_{filename}.png", dpi=300)
    plt.close()
    # plt.show()

#============ PCA Visualization =============#

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


def plot_embedding_PCA_runs(mean_traj,
                        mean_within_var_per_setting=None,
                        mean_phq9_per_setting=None,
                        num_steps=100,
                        shift=5,
                        path="",
                        filename="default.png",
                        sbert=False,
                        mentalbert=False,
                        reduction="pca",
                        save=False):
    """
    Plot PCA- or UMAP-reduced embedding trajectories. Points can be colored by PHQ-9; marker size by mean within-run variance.

    Args:
        mean_traj (dict[setting]): mean embedding trajectory (T, 2)
        mean_within_var_per_setting (dict[setting], optional): (T,) mean within-window variance per time; used for marker size
        mean_phq9_per_setting (dict[setting], optional): (T,) average PHQ-9 at each time point; used for scatter colors
        num_steps (int): window size
        shift (int): shift of window
        sbert (bool): whether sentence-embedding model was used (vs TF-IDF)
        mentalbert (bool): if True, label as MentalBERT; else SBERT
        reduction (str): "pca" or "umap" for title and filename
        path (str): path to save the figure
        filename (str): filename to save the figure
        save (bool): whether to save the figure
    """
    plt.figure(figsize=(5, 4))
    if sbert:
        embedding = "MentalBERT" if mentalbert else "SBERT"
    else:
        embedding = "TF-IDF"
    reduc_label = reduction.upper()
    plt.title(f"{embedding} {reduc_label}\n(window={num_steps}, shift={shift})")

    # Compute global min/max PHQ-9 across all settings for dynamic color scaling
    phq9_global_min, phq9_global_max = None, None
    if mean_phq9_per_setting is not None:
        all_phq9 = [np.asarray(v) for v in mean_phq9_per_setting.values()]
        if all_phq9:
            phq9_global_min = float(np.min([p.min() for p in all_phq9]))
            phq9_global_max = float(np.max([p.max() for p in all_phq9]))

    sc = None
    for setting, traj in mean_traj.items():
        traj = np.asarray(traj)  # (T, 2)

        if mean_within_var_per_setting is not None and setting in mean_within_var_per_setting:
            mwv = np.asarray(mean_within_var_per_setting[setting])
            if mwv.shape[0] == traj.shape[0]:
                mwv_max = mwv.max()
                s = 5 + 20 * (mwv / (mwv_max + 1e-8)) if mwv_max > 0 else np.full_like(mwv, 10)
            else:
                s = 10
        else:
            s = 10

        # Color by average PHQ-9 if provided (scaled to data range for visibility)
        if mean_phq9_per_setting is not None and setting in mean_phq9_per_setting:
            phq9 = np.asarray(mean_phq9_per_setting[setting])
            if phq9.shape[0] == traj.shape[0]:
                sc = plt.scatter(
                    traj[:, 0], traj[:, 1], c=phq9, s=s, alpha=0.7, label=setting,
                    cmap="viridis", vmin=phq9_global_min, vmax=phq9_global_max
                )
            else:
                plt.scatter(traj[:, 0], traj[:, 1], s=s, alpha=0.7, label=setting)
        else:
            plt.scatter(traj[:, 0], traj[:, 1], s=s, alpha=0.7, label=setting)
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.5, color="gray")

    plt.xlabel(f"{reduc_label} 1")
    plt.ylabel(f"{reduc_label} 2")
    plt.legend()
    plt.grid(alpha=0.3)
    if mean_phq9_per_setting is not None and sc is not None:
        cbar = plt.colorbar(sc, shrink=0.6)
        cbar.set_label("Mean PHQ-9 (system)")
    # plt.show()

    print(f"Saving {reduc_label} plot to folder {path} with filename {filename}")
    if save:
        emb_file = embedding.lower().replace("-", "_")
        plt.savefig(f"{path}/{emb_file}_{reduction}_runs{num_steps}_shift{shift}_{len(mean_traj)}settings_{filename}.png", bbox_inches='tight', dpi=300)
    plt.close()

#============ Network Analysis Visualization =============#

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


#============ Tweet Frequency Visualization =============#

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

#============= TESTING LLMS FOR PHQ-9 =============#
def plot_bias(bias_per_phq9, all_bias, directory="plots"):
    """
    Visualizes the mean bias per PHQ-9 score.
    """
    scores = sorted(bias_per_phq9.keys())
    biases = [bias_per_phq9[s] for s in scores]
    
    # Color code: Red for overestimating, Blue for underestimating
    colors = ['#ff6666' if b > 0 else '#6666ff' for b in biases]
    
    plt.figure(figsize=(10, 6))
    plt.bar(scores, biases, color=colors, edgecolor='black', alpha=0.8)
    
    # Zero line represents perfect accuracy
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5)
    
    plt.xlabel('Ground Truth $PHQ-9$ Score')
    plt.ylabel('Mean Bias')
    plt.title(f'LLM Bias (total bias: {(all_bias)}) ')
    plt.xticks(range(0, 28))
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Annotations for clarity
    # plt.text(0.5, max(biases) if biases and max(biases) > 0 else 2, 
    #          "Overestimating Depression ↑", color='red', fontsize=10, fontweight='bold')
    # plt.text(0.5, min(biases) if biases and min(biases) < 0 else -2, 
    #          "Underestimating Depression ↓", color='blue', fontsize=10, fontweight='bold')
    
    filename = os.path.join(directory, "llm_bias_per_phq9.png")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")

def plot_phq9_error(mae_per_phq9, total_mae, directory="plots"):
    """
    Visualizes the mean absolute error (MAE) per PHQ-9 score.
    Lower values indicate better performance.
    """
    scores = sorted(mae_per_phq9.keys())
    errors = [mae_per_phq9[s] for s in scores]
    
    plt.figure(figsize=(10, 6))
    plt.bar(scores, errors, color='#66b3ff', edgecolor='black', alpha=0.8)
    
    plt.xlabel('Ground Truth $PHQ-9$ Score')
    plt.ylabel('Mean Absolute Error (PHQ-9 points)')
    plt.title(f'LLM PHQ-9 error per score (total MAE: {total_mae:.3f})')
    plt.xticks(range(0, 28))
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    filename = os.path.join(directory, "llm_error_per_phq9.png")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")


def plot_combined_bias_error(csv_path, model_name, temp, top_p, check_point, directory):
    """
    Read results.csv, aggregate bias_per_phq9 and mae_per_phq9 across all
    seeds for a given (model, temp, top_p, check_point), and plot combined
    bias and error bar charts in directory.

    The title includes how many seeds were averaged.
    """
    

    if not os.path.isfile(csv_path):
        print(f"[combined plots] CSV not found: {csv_path} – skipping.")
        return

    df = pd.read_csv(csv_path)

    # Filter to matching configuration
    mask = (
        (df["model_name"] == model_name) &
        (df["temp"] == temp) &
        (df["top_p"] == top_p) &
        (df["check_point"] == check_point)
    )
    relavent_cols = df.loc[mask]

    if relavent_cols.empty:
        print(f"[combined plots] No rows for {model_name} temp={temp} top_p={top_p} cp={check_point} – skipping.")
        return


    n_seeds = len(relavent_cols)

    # weights are the number of agents in each run
    weights = relavent_cols["num_agents"].values.astype(float)
    total_agents = weights.sum()

    # select bias and mae per phq9 for every run 
    bias_cols = [c for c in relavent_cols.columns if c.startswith("bias_phq9_")]
    mae_cols  = [c for c in relavent_cols.columns if c.startswith("mae_phq9_")]

    if not os.path.exists(directory):
        os.makedirs(directory)

    #Combined bias plot (weighted by num_agents)
   
    scores = sorted([int(c.replace("bias_phq9_", "")) for c in bias_cols])
    biases = [float(np.average(relavent_cols[f"bias_phq9_{s}"].values, weights=weights)) for s in scores]
    overall_bias = float(np.average(relavent_cols["avg_phq9_change"].values, weights=weights))

    colors = ["red"	 if b > 0 else"blue" for b in biases]
    plt.figure(figsize=(10, 6))
    plt.bar(scores, biases, color=colors, edgecolor='black', alpha=0.8)
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5)
    plt.xlabel('Ground Truth $PHQ-9$ Score')
    plt.ylabel('Mean Bias')
    plt.title(f'LLM Bias combined over {n_seeds} seed(s), {int(total_agents)} agents (total bias: {overall_bias:.3f})')
    plt.xticks(range(0, 28))
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    fname = os.path.join(directory, "llm_bias_per_phq9.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Combined bias plot saved to {fname}")

    #Combined error plot
    errors = sorted([int(c.replace("mae_phq9_", "")) for c in mae_cols])
    mean_errors = [float(np.average(relavent_cols[f"mae_phq9_{s}"].values, weights=weights)) for s in errors]
    overall_mae = float(np.average(relavent_cols["total_mae"].values, weights=weights))

    plt.figure(figsize=(10, 6))
    plt.bar(scores, mean_errors, color="orange", edgecolor='black', alpha=0.8)
    plt.xlabel('Ground Truth $PHQ-9$ Score')
    plt.ylabel('Mean Absolute Error (PHQ-9 points)')
    plt.title(f'LLM PHQ-9 error – combined over {n_seeds} seed(s), {int(total_agents)} agents (total MAE: {overall_mae:.3f})')
    plt.xticks(range(0, 28))
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    fname = os.path.join(directory, "llm_error_per_phq9.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Combined error plot saved to {fname}")


#============= Critical slowing down Visualization =============#
def plot_agent_cd_heatmaps(network, window, cd_results, metric_name="PHQ-9", path="", filename="default.png", shift=1):
    """
    Plots heatmaps for Variance and Autocorrelation across all agents.
    Agents are sorted on the Y-axis by their final PHQ-9 score.
    """
    # Prepare sorting criteria (Final PHQ9 score per agent)
    agent_scores = []
    for agent in network.all_agents:
        final_score = agent.well_being.get("phq9_sumscore", 0)
        agent_scores.append((agent.ID, final_score))
    
    # Sort agents by score (low to high)
    sorted_agents = sorted(agent_scores, key=lambda x: x[1])
    sorted_ids = [a[0] for a in sorted_agents]
    
    # Reshape data into matrices (Rows = Agents, Cols = Time)
    var_matrix = []
    auto_matrix = []
    phq9_matrix = []

    id_to_agent = {agent.ID: agent for agent in network.all_agents}
    
    for agent_id in sorted_ids:
        phq9_matrix.append(id_to_agent[agent_id].all_phq9_sumscores[::shift])
        var_matrix.append(cd_results[agent_id]['variance'])
        auto_matrix.append(cd_results[agent_id]['autocorrelation'])
        
    var_matrix = np.array(var_matrix)
    auto_matrix = np.array(auto_matrix)
    phq9_matrix = np.array(phq9_matrix)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,7), sharex=True)
    
    # Heatmap for variance
    sns.heatmap(var_matrix, ax=ax1, cmap="YlOrRd", cbar_kws={'label': 'Variance'})
    ax1.set_title(f'Rolling Variance of {metric_name}')
    ax1.set_ylabel('Agents')

    # Heatmap for Autocorrelation
    sns.heatmap(auto_matrix, ax=ax2, cmap="YlGnBu", cbar_kws={'label': 'Autocorr'})
    ax2.set_title(f'Autocorrelation of {metric_name}')
    ax2.set_ylabel('Agents')

    sns.heatmap(phq9_matrix, ax=ax3, cmap="RdYlGn_r", cbar_kws={'label': 'PHQ-9 Score'})
    ax3.set_title(f'{metric_name} Scores Over Time')
    ax3.set_ylabel('Agents')
    ax3.set_xlabel('Time Steps (Rounds)')

    plt.suptitle(f'Critical Slowing Down (Agents sorted on PHQ-9)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"window_{window}_{filename}"))




    # ====================== LLM Bias and Accuracy Visualization ====================== #
def plot_model_comparison_by_settings(
    data=None,
    csv_path=None,
    directory="plots/test",
    save=True):
    """
    Visualization for PHQ-9 performance of the LLMs across different settings.

    Produces two vertically stacked subplots:
      1) Total PHQ-9 MAE (mean absolute error; lower is better)
      2) Mean PHQ-9 bias (signed; positive = overestimation, negative = underestimation)

    Args:
        data: list of dicts with keys
              "model_name", "temp", "top_p", "total_mae", "avg_phq9_change"
              (optional if csv_path is given).
        csv_path: path to results CSV with columns
              model_name, temp, top_p, total_mae, avg_phq9_change.
        directory: folder to save the figure.
        save: whether to save to disk.
    """
    import pandas as pd

    # check data or paths are provided
    if data is None and csv_path is None:
        raise ValueError("Provide either data or csv_path")

    # read in data from csv if data is not provided
    if data is None:
        df = pd.read_csv(csv_path)
        required_cols = ["model_name", "temp", "top_p", "total_mae", "avg_phq9_change"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in results CSV: {missing}")
        data = df[required_cols].to_dict("records")

    # model specifics
    rows = [
        {
            "model": str(r["model_name"]),
            "temp": float(r["temp"]),
            "top_p": float(r["top_p"]),
            "mae": float(r["total_mae"]),
            "bias": float(r["avg_phq9_change"]),
        }
        for r in data
    ]

    # helper function to fix model names
    def short_name(model):
        s = model.split("/")[-1] if "/" in model else model
        return s.replace("-Instruct", "").replace("_", " ")[:24]

    # get models and settings
    models = sorted(set(r["model"] for r in rows))
    settings = sorted(set((r["temp"], r["top_p"]) for r in rows), key=lambda x: (x[0], x[1]))
    n_models = len(models)
    n_settings = len(settings)
    if n_models == 0 or n_settings == 0:
        return

    # create mapping of models and settings to indices
    model_to_idx = {m: i for i, m in enumerate(models)}
    setting_to_idx = {s: j for j, s in enumerate(settings)}
    mae_mat = np.full((n_models, n_settings), np.nan)
    bias_mat = np.full((n_models, n_settings), np.nan)
    for r in rows:
        i = model_to_idx[r["model"]]
        j = setting_to_idx[(r["temp"], r["top_p"])]
        mae_mat[i, j] = r["mae"]
        bias_mat[i, j] = r["bias"]

    # create colourmap
    if n_settings <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_settings, 1)))[:n_settings]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_settings))[:n_settings]

    hatches = ["", "//", "\\\\", "||", "--", "++", "xx", "..", "**", "oo"][:n_settings]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True,
        figsize=(max(8, n_models * (1.5 + 0.5 * n_settings)), 9)
    )
    x = np.arange(n_models)
    total_width = 0.8

    # plot bars with some distance apart for each setting
    bar_width = total_width / n_settings
    for j, (t, p) in enumerate(settings):
        off = (j - (n_settings - 1) / 2) * bar_width
        for i in range(n_models):
            mae_val = mae_mat[i, j]
            bias_val = bias_mat[i, j]
            if not np.isnan(mae_val):
                ax_top.bar(
                    x[i] + off,
                    mae_val,
                    bar_width,
                    color=colors[i],
                    edgecolor="gray",
                    hatch=hatches[j],
                )
            if not np.isnan(bias_val):
                ax_bottom.bar(
                    x[i] + off,
                    bias_val,
                    bar_width,
                    color=colors[i],
                    edgecolor="gray",
                    hatch=hatches[j],
                )

    # match the model name with the color and hatch
    legend_handles = []
    for i, m in enumerate(models):
        legend_handles.append(
            plt.matplotlib.patches.Patch(
                facecolor=colors[i], edgecolor="gray", label=short_name(m)
            )
        )
    for j, (t, p) in enumerate(settings):
        legend_handles.append(
            plt.matplotlib.patches.Patch(
                facecolor="white",
                edgecolor="gray",
                hatch=hatches[j],
                label=f"T={t}, p={p}",
            )
        )

    # set specifics for top subplot (MAE)
    ax_top.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax_top.set_ylabel("Total MAE (PHQ-9 points,\nlower is better)")
    ax_top.set_title("Model comparison by settings – PHQ-9 error (MAE)")
    ax_top.set_ylim(0, 1.0)
    ax_top.grid(axis="y", linestyle="--", alpha=0.3)

    # set specifics for bottom subplot (bias)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([short_name(m) for m in models], rotation=45, ha="right")
    ax_bottom.set_ylabel("Mean PHQ-9 bias\n(positive = overestimate)")
    ax_bottom.set_xlabel("Model")
    ax_bottom.set_title("Model comparison by settings – PHQ-9 bias")
    ax_bottom.axhline(0, color="black", linewidth=1.0)
    ax_bottom.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "model_comparison_by_settings.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close()