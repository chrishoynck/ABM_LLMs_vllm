import seaborn as sns
import os
import argparse
import glob
import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from . import metrics
from .tools.format_config import FC
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
    if cds_info is None or len(cds_info) == 0:
        print("[distorted_info] no CDS info available, skipping plot")
        return
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


def plot_within_variance(mean_within_var_per_setting, shift=5, path="", filename="default.png", save=False):
    """Standalone plot of within-run embedding variance over time."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for setting, mwv in mean_within_var_per_setting.items():
        mwv = np.asarray(mwv)
        time_steps = np.arange(mwv.shape[0]) * shift
        ax.plot(time_steps, mwv, alpha=0.6, label=setting)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Variance")
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    if save and path and filename:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/within_variance_{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_embedding_PCA_runs(mean_traj,
                        mean_within_var_per_setting=None,
                        mean_phq9_per_setting=None,
                        assort_data=None,
                        num_steps=100,
                        shift=5,
                        path="",
                        filename="default.png",
                        sbert=False,
                        mentalbert=False,
                        reduction="pca",
                        save=False,
                        use_sd_band=False):
    """
    Two-panel figure: (a) UMAP/PCA trajectory, (b) assortativity + DW mean PHQ-9.

    Args:
        assort_data (dict): Pre-computed output from plot_phq9_assortativity, containing
            bin_timesteps, bin_assort_mean, bin_assort_std, bin_dw_phq9_mean,
            bin_dw_phq9_min, bin_dw_phq9_max, bin_dw_phq9_sd.
            If None, panel (b) is left empty.
        use_sd_band (bool): If True, show mean ± cross-agent SD band instead of
            the min–max range on panel (b). Default False (min–max).
    """
    # Increased height slightly to accommodate the labels underneath
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    embedding = ("MentalBERT" if mentalbert else "SBERT") if sbert else "TF-IDF"
    reduc_label = reduction.upper()

    # --- Setup Global Color Scaling ---
    phq9_global_min, phq9_global_max = None, None
    if mean_phq9_per_setting is not None:
        all_phq9 = [np.asarray(v) for v in mean_phq9_per_setting.values()]
        if all_phq9:
            phq9_global_min = float(np.min([p.min() for p in all_phq9]))
            phq9_global_max = float(np.max([p.max() for p in all_phq9]))

    sc = None

    # --- Plotting Loop (ax1: UMAP/PCA) ---
    for setting, traj in mean_traj.items():
        traj = np.asarray(traj)
        num_windows = traj.shape[0]

        mwv = None
        if mean_within_var_per_setting is not None and setting in mean_within_var_per_setting:
            mwv = np.asarray(mean_within_var_per_setting[setting])
            if mwv.shape[0] == num_windows:
                mwv_max = mwv.max()
                s = 5 + 20 * (mwv / (mwv_max + 1e-8)) if mwv_max > 0 else np.full_like(mwv, 10)
            else:
                s = 10
        else:
            s = 10

        phq9 = None
        if mean_phq9_per_setting is not None and setting in mean_phq9_per_setting:
            phq9 = np.asarray(mean_phq9_per_setting[setting])

        if phq9 is not None and phq9.shape[0] == num_windows:
            sc = ax1.scatter(
                traj[:, 0], traj[:, 1], c=phq9, s=s, alpha=0.7,
                cmap="viridis", vmin=phq9_global_min, vmax=phq9_global_max
            )
        else:
            ax1.scatter(traj[:, 0], traj[:, 1], s=s, alpha=0.7)

        ax1.plot(traj[:, 0], traj[:, 1], alpha=0.4, color="gray", linewidth=1)

        # Flag start and end points — offset text toward trajectory center
        center = traj.mean(axis=0)
        for pt, label, color in [(traj[0], 'Start', 'black'), (traj[-1], 'End', 'black')]:
            dx = center[0] - pt[0]
            dy = center[1] - pt[1]
            norm = np.sqrt(dx**2 + dy**2) + 1e-10
            off_pts = 14  # offset in points
            ax1.annotate(label, xy=(pt[0], pt[1]),
                         xytext=(off_pts * dx / norm, off_pts * dy / norm),
                         textcoords='offset points',
                         fontsize=8, fontweight='bold', color=color, zorder=11,
                         arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # --- Formatting AX1 ---
    ax1.set_xlabel(f"{reduc_label} 1")
    ax1.set_ylabel(f"{reduc_label} 2")
    ax1.grid(alpha=0.3)

    # Add (a) subscript underneath
    ax1.text(0.5, -0.25, "(a)", transform=ax1.transAxes,
             ha='center', va='top', fontsize=12, fontweight='bold')

    if mean_phq9_per_setting is not None and sc is not None:
        cbar = fig.colorbar(sc, ax=ax1, shrink=0.7)
        cbar.set_label("Mean PHQ-9")

    # --- AX2: Assortativity + DW mean PHQ-9 ---
    if assort_data is not None:
        bin_t = assort_data["bin_timesteps"]
        # Left axis (ax2): assortativity with SD error bars
        ax2.errorbar(bin_t, assort_data["bin_assort_mean"], yerr=assort_data["bin_assort_std"],
                     color='steelblue', fmt='o-', capsize=3, linewidth=1.5, markersize=3,
                     label='Assortativity ± SD')
        # ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("PHQ-9 Assortativity (r)", color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        ax2.grid(alpha=0.3)

        # Right axis: DW mean PHQ-9 with error band
        ax2_right = ax2.twinx()
        dw_mean = assort_data["bin_dw_phq9_mean"]
        ax2_right.plot(bin_t, dw_mean, 's--', color='firebrick',
                       linewidth=1.0, markersize=3, label='DW mean PHQ-9')
        if use_sd_band:
            dw_sd = assort_data["bin_dw_phq9_sd"]
            band_lo = np.maximum(0, dw_mean - 0.1 * dw_sd)
            band_hi = dw_mean + 0.1 * dw_sd
            ax2_right.fill_between(bin_t, band_lo, band_hi,
                                   color='firebrick', alpha=0.15, label='± 0.1 cross-agent SD')
        else:
            band_lo = assort_data["bin_dw_phq9_min"]
            band_hi = assort_data["bin_dw_phq9_max"]
            ax2_right.fill_between(bin_t, band_lo, band_hi,
                                   color='firebrick', alpha=0.15, label='Min–max range')
        ax2_right.set_ylabel("DW PHQ-9 Score", color='firebrick')
        y_lo = np.nanmin(band_lo)
        y_hi = np.nanmax(band_hi)
        margin = max((y_hi - y_lo) * 0.1, 0.5)
        ax2_right.set_ylim(max(0, y_lo - margin), min(27, y_hi + margin))
        ax2_right.tick_params(axis='y', labelcolor='firebrick')

        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=6)

    # Add (b) subscript underneath
    ax2.text(0.5, -0.25, "(b)", transform=ax2.transAxes,
             ha='center', va='top', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # --- Save Logic ---
    if save:
        emb_file = embedding.lower().replace("-", "_")
        full_path = f"{path}/{emb_file}_{reduction}_runs{num_steps}_shift{shift}_{len(mean_traj)}settings_{filename}"
        plt.savefig(full_path, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close()

# def plot_embedding_PCA_runs(mean_traj,
#                         mean_within_var_per_setting=None,
#                         mean_phq9_per_setting=None,
#                         num_steps=100,
#                         shift=5,
#                         path="",
#                         filename="default.png",
#                         sbert=False,
#                         mentalbert=False,
#                         reduction="pca",
#                         save=False):
#     """
#     Plot PCA- or UMAP-reduced embedding trajectories. Points can be colored by PHQ-9; marker size by mean within-run variance.

#     Args:
#         mean_traj (dict[setting]): mean embedding trajectory (T, 2)
#         mean_within_var_per_setting (dict[setting], optional): (T,) mean within-window variance per time; used for marker size
#         mean_phq9_per_setting (dict[setting], optional): (T,) average PHQ-9 at each time point; used for scatter colors
#         num_steps (int): window size
#         shift (int): shift of window
#         sbert (bool): whether sentence-embedding model was used (vs TF-IDF)
#         mentalbert (bool): if True, label as MentalBERT; else SBERT
#         reduction (str): "pca" or "umap" for title and filename
#         path (str): path to save the figure
#         filename (str): filename to save the figure
#         save (bool): whether to save the figure
#     """
#     plt.figure(figsize=(4, 3))
#     if sbert:
#         embedding = "MentalBERT" if mentalbert else "SBERT"
#     else:
#         embedding = "TF-IDF"
#     reduc_label = reduction.upper()
#     plt.title(f"{embedding} {reduc_label}") #\n(window={num_steps}, shift={shift})")

#     # Compute global min/max PHQ-9 across all settings for dynamic color scaling
#     phq9_global_min, phq9_global_max = None, None
#     if mean_phq9_per_setting is not None:
#         all_phq9 = [np.asarray(v) for v in mean_phq9_per_setting.values()]
#         if all_phq9:
#             phq9_global_min = float(np.min([p.min() for p in all_phq9]))
#             phq9_global_max = float(np.max([p.max() for p in all_phq9]))

#     sc = None
#     for setting, traj in mean_traj.items():
#         traj = np.asarray(traj)  # (T, 2)

#         if mean_within_var_per_setting is not None and setting in mean_within_var_per_setting:
#             mwv = np.asarray(mean_within_var_per_setting[setting])
#             if mwv.shape[0] == traj.shape[0]:
#                 mwv_max = mwv.max()
#                 s = 5 + 20 * (mwv / (mwv_max + 1e-8)) if mwv_max > 0 else np.full_like(mwv, 10)
#             else:
#                 s = 10
#         else:
#             s = 10

#         # Color by average PHQ-9 if provided (scaled to data range for visibility)
#         if mean_phq9_per_setting is not None and setting in mean_phq9_per_setting:
#             phq9 = np.asarray(mean_phq9_per_setting[setting])
#             if phq9.shape[0] == traj.shape[0]:
#                 sc = plt.scatter(
#                     traj[:, 0], traj[:, 1], c=phq9, s=s, alpha=0.7, label=setting,
#                     cmap="viridis", vmin=phq9_global_min, vmax=phq9_global_max
#                 )
#             else:
#                 plt.scatter(traj[:, 0], traj[:, 1], s=s, alpha=0.7, label=setting)
#         else:
#             plt.scatter(traj[:, 0], traj[:, 1], s=s, alpha=0.7, label=setting)
#         plt.plot(traj[:, 0], traj[:, 1], alpha=0.5, color="gray")

#     plt.xlabel(f"{reduc_label} 1")
#     plt.ylabel(f"{reduc_label} 2")
#     # plt.legend()
#     plt.grid(alpha=0.3)
#     if mean_phq9_per_setting is not None and sc is not None:
#         cbar = plt.colorbar(sc, shrink=0.6)
#         cbar.set_label("Mean PHQ-9")
#     # plt.show()

#     print(f"Saving {reduc_label} plot to folder {path} with filename {filename}")
#     if save:
#         emb_file = embedding.lower().replace("-", "_")
#         plt.savefig(f"{path}/{emb_file}_{reduction}_runs{num_steps}_shift{shift}_{len(mean_traj)}settings_{filename}.png", bbox_inches='tight', dpi=300)
#     plt.close()

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
    
    plt.figure(figsize=(5, 4))
    plt.bar(scores, biases, color=colors, edgecolor='black', alpha=0.8)
    
    # Zero line represents perfect accuracy
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5)
    
    plt.xlabel('Ground Truth $PHQ-9$ Score')
    plt.ylabel('Mean Bias')
    plt.title(f'LLM Bias (Total Bias: {all_bias:.2f})')
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
    
    plt.figure(figsize=(5, 4))
    plt.bar(scores, errors, color='#66b3ff', edgecolor='black', alpha=0.8)
    
    plt.xlabel('Ground Truth $PHQ-9$ Score')
    plt.ylabel('Mean Absolute Error (PHQ-9 points)')
    plt.title(f'LLM PHQ-9 Error (Total MAE: {total_mae:.2f})')
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


#============= Shared helpers for semantic / echo-chamber plots =========#

def _smooth_series(arr, window):
    """Rolling mean and std for a 1-D array, NaN-aware. Window is centered."""
    arr = np.asarray(arr, dtype=float)
    if window <= 1:
        return arr, np.zeros_like(arr)
    import pandas as pd
    s = pd.Series(arr)
    mean = s.rolling(window, center=True, min_periods=1).mean().values
    std = s.rolling(window, center=True, min_periods=1).std(ddof=0).values
    return mean, std


def _cosine_sim_matrix(embeddings):
    """Cosine similarity matrix from an (N, D) embedding array."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-10)
    return normed @ normed.T


def _build_neighbor_pairs(graph, id_to_idx, network):
    """Set of (min_idx, max_idx) tuples for all connected agent pairs."""
    undirected = graph if not network.directed else graph.to_undirected()
    pairs = set()
    for agent in network.all_agents:
        i = id_to_idx[agent.ID]
        for nid in undirected.neighbors(agent.ID):
            j = id_to_idx.get(nid)
            if j is not None:
                pairs.add((min(i, j), max(i, j)))
    return pairs


def _get_T(network, include_phq9=False):
    """Minimum number of timesteps across all agents."""
    T = min(len(a.tweethistory) for a in network.all_agents)
    if include_phq9:
        T = min(T, min(len(a.all_phq9_sumscores) for a in network.all_agents))
    return T


def _prepare_embs(network, agent_embs, mentalbert, cache_path=None):
    """Load embeddings if needed; return (embs, has_valid).

    If cache_path is given it is forwarded to build_agent_embeddings so that
    embeddings are loaded from / saved to disk automatically.
    """
    if agent_embs is None:
        agent_embs = metrics.build_agent_embeddings(network, mentalbert=mentalbert,
                                                    cache_path=cache_path)
    has_valid = any(e is not None for row in agent_embs for e in row)
    return agent_embs, has_valid


def _save_and_close(fig, save, path, filename, prefix, show_fig):
    """tight_layout → optional save → show/close."""
    plt.tight_layout()
    if save and path and filename:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/{prefix}_{filename}.png", dpi=300, bbox_inches="tight")
    if show_fig:
        plt.show()
    plt.close()


def _plot_smoothed(ax, timesteps, raw, smooth_window, color, label,
                   linewidth=1.5, linestyle='-', fill_alpha=0.15, raw_std=None, raw_n=None):
    """Plot a time series with optional smoothing and ±1 SE band.

    When smooth_window > 1 and raw_std/raw_n are provided: rolling mean ± smoothed cross-agent SE.
    When smooth_window > 1 without raw_std/raw_n: rolling mean ± rolling temporal std.
    When smooth_window == 1 and raw_std is provided: raw ± raw_std.
    """
    if smooth_window > 1:
        sm, temporal_std = _smooth_series(raw, smooth_window)
        if raw_std is not None and raw_n is not None:
            smoothed_std, _ = _smooth_series(raw_std, smooth_window)
            smoothed_n, _ = _smooth_series(np.asarray(raw_n, dtype=float), smooth_window)
            std = smoothed_std / np.sqrt(np.maximum(smoothed_n, 1))
        else:
            std = temporal_std
    else:
        sm, std = raw, raw_std
    ax.plot(timesteps, sm, color=color, linewidth=linewidth, linestyle=linestyle, label=label)
    if std is not None:
        ax.fill_between(timesteps, sm - std, sm + std, color=color, alpha=fill_alpha)
    return sm


#============= Semantic Entrainment & Vector Assortativity =============#

def plot_semantic_entrainment(network, agent_embs=None, mentalbert=True, path="", filename="",
                              save=False, show_fig=True, smooth_window=1, cache_path=None):
    """
    Plots local (neighbor) vs. random cosine similarity over time.

    Positive local−random gap = semantic assortativity (entrainment).
    smooth_window > 1 applies a centered rolling average with ±1 SD band.
    cache_path: optional .npz path forwarded to build_agent_embeddings for save/load.
    """
    import random as _random

    graph, id_to_idx = metrics.build_network_graph(network)
    T = _get_T(network)
    agent_embs, has_valid = _prepare_embs(network, agent_embs, mentalbert, cache_path=cache_path)
    if not has_valid:
        print("No valid tweets found in the network.")
        return None, None, {}

    mean_local_sims, std_local_sims, n_local_sims, mean_global_sims, std_global_sims, n_global_sims = [], [], [], [], [], []
    print("Computing semantic entrainment per timestep (agent@t vs. neighbors@t-1)...")
    for t in range(1, T):  # start at 1: need t-1 for neighbors
        # Agents valid at t (self) and agents valid at t-1 (potential neighbors/random)
        valid_t  = [i for i in range(len(network.all_agents)) if agent_embs[i][t] is not None]
        valid_t1 = {i for i in range(len(network.all_agents)) if agent_embs[i][t-1] is not None}
        if len(valid_t) < 2 or len(valid_t1) < 2:
            mean_local_sims.append(np.nan); std_local_sims.append(np.nan); n_local_sims.append(0)
            mean_global_sims.append(np.nan); std_global_sims.append(np.nan); n_global_sims.append(0)
            continue

        # Precompute normalized embeddings for agents at t and at t-1
        embs_t = np.stack([agent_embs[i][t] for i in valid_t])
        norms_t = np.linalg.norm(embs_t, axis=1, keepdims=True)
        normed_t = embs_t / np.maximum(norms_t, 1e-10)

        valid_t1_list = sorted(valid_t1)
        t1_to_pos = {idx: pos for pos, idx in enumerate(valid_t1_list)}
        embs_t1 = np.stack([agent_embs[i][t-1] for i in valid_t1_list])
        norms_t1 = np.linalg.norm(embs_t1, axis=1, keepdims=True)
        normed_t1 = embs_t1 / np.maximum(norms_t1, 1e-10)

        # Cross-similarity: (agents@t) × (agents@t-1)
        cross_sim = normed_t @ normed_t1.T

        local_sims, global_sims = [], []
        for row, i in enumerate(valid_t):
            agent = network.all_agents[i]
            sources = graph.predecessors(agent.ID) if network.directed else graph.neighbors(agent.ID)
            # Neighbors must have valid embeddings at t-1
            neighbor_pos = [t1_to_pos[id_to_idx[nid]] for nid in sources
                            if id_to_idx.get(nid) is not None and id_to_idx[nid] in t1_to_pos]
            if not neighbor_pos:
                continue
            local_sims.append(np.mean(cross_sim[row, neighbor_pos]))

            # Random baseline: sample from agents valid at t-1 (excluding neighbors and self)
            self_pos = t1_to_pos.get(i)
            exclude = set(neighbor_pos) | ({self_pos} if self_pos is not None else set())
            other_pos = [p for p in range(len(valid_t1_list)) if p not in exclude]
            if len(other_pos) >= len(neighbor_pos):
                sampled = _random.sample(other_pos, len(neighbor_pos))
                global_sims.append(np.mean(cross_sim[row, sampled]))

        mean_local_sims.append(np.mean(local_sims) if local_sims else np.nan)
        std_local_sims.append(np.std(local_sims)   if local_sims else np.nan)
        n_local_sims.append(len(local_sims) if local_sims else 0)
        mean_global_sims.append(np.mean(global_sims) if global_sims else np.nan)
        std_global_sims.append(np.std(global_sims) if global_sims else np.nan)
        n_global_sims.append(len(global_sims) if global_sims else 0)

    # --- Plot ---
    raw_local  = np.array(mean_local_sims)
    raw_std    = np.array(std_local_sims)
    raw_n_local = np.array(n_local_sims)
    raw_global = np.array(mean_global_sims)
    raw_std_global = np.array(std_global_sims)
    raw_n_global = np.array(n_global_sims)
    timesteps  = np.arange(1, T)  # starts at 1 (agent@t vs. neighbors@t-1)
    sfx = f" (smoothed, w={smooth_window})" if smooth_window > 1 else ""

    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_smoothed(ax, timesteps, raw_local, smooth_window, 'teal',
                   f'Mean Local Cosine Similarity{sfx}', linewidth=2, raw_std=raw_std, raw_n=raw_n_local, fill_alpha=0.2)
    _plot_smoothed(ax, timesteps, raw_global, smooth_window, 'gray',
                   f'Mean Random Cosine Similarity (baseline){sfx}', linewidth=1.5, linestyle='--',
                   raw_std=raw_std_global, raw_n=raw_n_global, fill_alpha=0.1)
    ax.set_xlabel("Time Step"); ax.set_ylabel("Cosine Similarity")
    ax.set_title("Semantic Entrainment & Vector Assortativity over Time")
    ax.legend(loc='best'); ax.grid(True, alpha=0.3)

    _save_and_close(fig, save, path, filename, "semantic_entrainment", show_fig)
    return fig, ax, {"mean_local": raw_local, "std_local": raw_std, "mean_global": raw_global}


#============= PHQ-9 × Semantic Alignment ==============================#

def plot_phq9_semantic_alignment(network, agent_embs=None, mentalbert=True, path="", filename="",
                                 save=False, show_fig=True, smooth_window=1, cache_path=None):
    """
    Tests whether PHQ-9 similarity predicts semantic similarity, split by neighbor status.
    NOTE: this is a more direct test of the relationship between PHQ-9 and semantics than the echo chamber plot.
    mentalBERT already encodes some PHQ-9-related signals,
    so we expect a positive correlation between PHQ-9 similarity and semantic similarity even without entrainment.
    key question is whether this correlation is stronger for neighbors than non-neighbors,
    which would suggest that agents are semantically aligning more with those who have similar PHQ-9 scores.

    Top panel: Spearman rho(|(DELTA)PHQ-9|, cosine_sim) over time.
    Bottom panel: mean cosine sim stratified by neighbor/non-neighbor × same/diff PHQ-9.
    smooth_window > 1 applies a centered rolling average with ±1 SD band.
    cache_path: optional .npz path forwarded to build_agent_embeddings for save/load.
    """
    from scipy.stats import spearmanr

    PHQ9_SIM_THRESHOLD = 5

    graph, id_to_idx = metrics.build_network_graph(network)
    T = _get_T(network, include_phq9=True)
    neighbor_pairs = _build_neighbor_pairs(graph, id_to_idx, network)
    agent_embs, has_valid = _prepare_embs(network, agent_embs, mentalbert, cache_path=cache_path)
    if not has_valid:
        print("No valid tweets found.")
        return None, None, {}

    # Per-timestep metric accumulators
    keys = ["spearman_corr", "neigh_same_phq9", "neigh_diff_phq9", "rand_same_phq9", "rand_diff_phq9"]
    series = {k: [] for k in keys}
    bucket_keys = [k for k in keys if k != "spearman_corr"]
    std_series = {k: [] for k in bucket_keys}
    n_series = {k: [] for k in bucket_keys}

    print("Computing PHQ-9 × semantic alignment per timestep...")
    for t in range(T):
        valid_idx = [i for i in range(len(network.all_agents))
                     if agent_embs[i][t] is not None
                     and network.all_agents[i].all_phq9_sumscores[t] is not None]
        if len(valid_idx) < 4:
            for k in keys:
                series[k].append(np.nan)
            for k in bucket_keys:
                std_series[k].append(np.nan)
                n_series[k].append(0)
            continue

        sim_matrix = _cosine_sim_matrix(np.stack([agent_embs[i][t] for i in valid_idx]))
        phq9_scores = np.array([network.all_agents[i].all_phq9_sumscores[t] for i in valid_idx], dtype=float)

        # All unique pairs
        N = len(valid_idx)
        rows, cols = np.triu_indices(N, k=1)
        phq9_diffs = np.abs(phq9_scores[rows] - phq9_scores[cols])
        cos_sims   = sim_matrix[rows, cols]

        if phq9_diffs.std() > 0 and cos_sims.std() > 0:
            corr, _ = spearmanr(phq9_diffs, cos_sims)
        else:
            corr = np.nan
        series["spearman_corr"].append(corr)

        # Stratify: neighbor/non-neighbor × same/diff PHQ-9
        buckets = {"neigh_same_phq9": [], "neigh_diff_phq9": [], "rand_same_phq9": [], "rand_diff_phq9": []}
        for k_pair, (r, c) in enumerate(zip(rows, cols)):
            gi, gj = valid_idx[r], valid_idx[c]
            is_neigh = (min(gi, gj), max(gi, gj)) in neighbor_pairs
            same_phq = phq9_diffs[k_pair] <= PHQ9_SIM_THRESHOLD
            bucket = ("neigh_" if is_neigh else "rand_") + ("same_phq9" if same_phq else "diff_phq9")
            buckets[bucket].append(cos_sims[k_pair])
        for k in buckets:
            series[k].append(np.mean(buckets[k]) if buckets[k] else np.nan)
            std_series[k].append(np.std(buckets[k]) if buckets[k] else np.nan)
            n_series[k].append(len(buckets[k]) if buckets[k] else 0)

    # Plot 
    raw = {k: np.array(v) for k, v in series.items()}
    raw_stds = {k: np.array(v) for k, v in std_series.items()}
    raw_ns = {k: np.array(v) for k, v in n_series.items()}
    timesteps = np.arange(T)
    sfx = f" (smoothed, w={smooth_window})" if smooth_window > 1 else ""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    _plot_smoothed(ax1, timesteps, raw["spearman_corr"], smooth_window, 'purple', None)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_ylabel("Spearman ρ\n(PHQ-9 diff vs. cosine sim)")
    ax1.set_title(f"PHQ-9 Score Similarity × Semantic Similarity Alignment{sfx}")
    ax1.grid(True, alpha=0.3)

    # threshold for "similar" PHQ-9 scores
    TH = PHQ9_SIM_THRESHOLD
    for key, color, ls, lw, label in [
        ("neigh_same_phq9", 'teal',   '-',  2,   f'Neighbor, |ΔPHQ-9| ≤ {TH}'),
        ("neigh_diff_phq9", 'teal',   '--', 2,   f'Neighbor, |ΔPHQ-9| > {TH}'),
        ("rand_same_phq9",  'salmon', '-',  1.5, f'Non-neighbor, |ΔPHQ-9| ≤ {TH}'),
        ("rand_diff_phq9",  'gray',   '--', 1.5, f'Non-neighbor, |ΔPHQ-9| > {TH}'),
    ]:
        _plot_smoothed(ax2, timesteps, raw[key], smooth_window, color, label,
                       linewidth=lw, linestyle=ls, fill_alpha=0.1,
                       raw_std=raw_stds.get(key), raw_n=raw_ns.get(key))
    ax2.set_xlabel("Time Step"); ax2.set_ylabel("Mean Pairwise Cosine Similarity")
    ax2.legend(loc='best', fontsize=8); ax2.grid(True, alpha=0.3)

    _save_and_close(fig, save, path, filename, "phq9_semantic_alignment", show_fig)
    return fig, (ax1, ax2), raw


#============= Depression Echo Chamber ==================================#

def plot_depression_echo_chamber(network, agent_embs=None, mentalbert=True, path="", filename="",
                                 save=False, show_fig=True, smooth_window=1, step=1, cache_path=None):
    """
    Three-panel figure testing whether semantic content drives depression echo chambers.

    Panel 1 — PHQ-9 assortativity + cross-agent PHQ-9 variance (twin axis).
    Panel 2 — Depression-axis alignment (Pearson r: projection vs. PHQ-9).
    Panel 3 — Depression-axis entrainment (local vs. random similarity on depression axis).

    Depression axis fitted with temporal cross-validation (split-half).
    smooth_window > 1 applies a centered rolling average with ±1 SD band.
    step > 1 subsamples computation to every Nth timestep (useful when PHQ-9
    updates every N steps, making intermediate timesteps redundant for panel 1).
    cache_path: optional .npz path forwarded to build_agent_embeddings for save/load.
    """
    import random as _random
    from scipy.stats import pearsonr

    graph, id_to_idx = metrics.build_network_graph(network)
    undirected = graph.to_undirected()
    T = _get_T(network, include_phq9=True)
    agent_embs, has_valid = _prepare_embs(network, agent_embs, mentalbert, cache_path=cache_path)
    if not has_valid:
        print("No valid tweets found.")
        return None, None, {}

    # ── Depression axis: temporal cross-validation (split-half) ──
    T_mid = T // 2

    def _fit_depression_axis(t_start, t_end):
        embs, scores = [], []
        for i, agent in enumerate(network.all_agents):
            for t in range(t_start, t_end):
                emb, phq = agent_embs[i][t], agent.all_phq9_sumscores[t]
                if emb is not None and phq is not None:
                    embs.append(emb); scores.append(float(phq))
        if len(embs) < 2:
            return None
        E, y = np.stack(embs), np.array(scores)
        axis = E.T @ (y - y.mean())
        norm = np.linalg.norm(axis)
        return axis / norm if norm > 1e-10 else None

    axis_from_first  = _fit_depression_axis(0, T_mid)
    axis_from_second = _fit_depression_axis(T_mid, T)
    print(f"Depression axes computed via temporal cross-validation (split at t={T_mid}).")

    # Project embeddings onto the held-out axis
    agent_proj = []
    for i in range(len(network.all_agents)):
        row = []
        for t in range(T):
            emb = agent_embs[i][t]
            axis = axis_from_second if t < T_mid else axis_from_first
            row.append(float(emb @ axis) if (emb is not None and axis is not None) else None)
        agent_proj.append(row)

    # ── Per-timestep metrics (subsampled by step, agent@t vs. neighbors@t-1) ──
    eval_timesteps = [t for t in range(max(1, step), T, step)]  # start at ≥1 for lag
    metric_lists = {"depression_align_r": [], "mean_local_proj": [], "mean_rand_proj": []}
    std_lists = {"mean_local_proj": [], "mean_rand_proj": []}
    n_lists = {"mean_local_proj": [], "mean_rand_proj": []}

    print(f"Computing echo-chamber metrics (step={step}, {len(eval_timesteps)} points, lag-1)...")
    for t in eval_timesteps:
        valid_idx = [i for i in range(len(network.all_agents))
                     if agent_proj[i][t] is not None
                     and network.all_agents[i].all_phq9_sumscores[t] is not None]
        # Agents with valid projections at t-1 (for neighbor/random comparisons)
        valid_t1 = {i for i in range(len(network.all_agents)) if agent_proj[i][t-1] is not None}
        if len(valid_idx) < 4:
            for lst in metric_lists.values():
                lst.append(np.nan)
            for k in std_lists:
                std_lists[k].append(np.nan)
                n_lists[k].append(0)
            continue

        projs = np.array([agent_proj[i][t] for i in valid_idx])
        phq9s = np.array([float(network.all_agents[i].all_phq9_sumscores[t]) for i in valid_idx])

        # Panel 1: depression-axis projection vs. PHQ-9
        if projs.std() > 1e-10 and phq9s.std() > 1e-10:
            r_align, _ = pearsonr(projs, phq9s)
        else:
            r_align = np.nan
        metric_lists["depression_align_r"].append(r_align)

        # Panel 2: local vs. random entrainment on depression axis (agent@t vs. neighbors@t-1)
        # Scale uses all projections at t-1 for consistent normalization
        all_projs_t1 = [agent_proj[i][t-1] for i in range(len(network.all_agents)) if agent_proj[i][t-1] is not None]
        scale = np.std(all_projs_t1) + 1e-10 if all_projs_t1 else 1.0
        local_sims, rand_sims = [], []
        for i in valid_idx:
            proj_i = agent_proj[i][t]
            sources = graph.predecessors(network.all_agents[i].ID) if network.directed \
                      else graph.neighbors(network.all_agents[i].ID)
            neighbor_idx = [id_to_idx[nid] for nid in sources if id_to_idx.get(nid) in valid_t1]
            if not neighbor_idx:
                continue
            local_sims.append(np.mean([1.0 - abs(proj_i - agent_proj[j][t-1]) / scale for j in neighbor_idx]))
            other = [j for j in valid_t1 if j != i and j not in set(neighbor_idx)]
            if len(other) >= len(neighbor_idx):
                sampled = _random.sample(other, len(neighbor_idx))
                rand_sims.append(np.mean([1.0 - abs(proj_i - agent_proj[j][t-1]) / scale for j in sampled]))

        metric_lists["mean_local_proj"].append(np.mean(local_sims) if local_sims else np.nan)
        std_lists["mean_local_proj"].append(np.std(local_sims) if local_sims else np.nan)
        n_lists["mean_local_proj"].append(len(local_sims) if local_sims else 0)
        metric_lists["mean_rand_proj"].append(np.mean(rand_sims)   if rand_sims  else np.nan)
        std_lists["mean_rand_proj"].append(np.std(rand_sims) if rand_sims else np.nan)
        n_lists["mean_rand_proj"].append(len(rand_sims) if rand_sims else 0)

    # ── Plot (2 panels) ──
    raw = {k: np.array(v) for k, v in metric_lists.items()}
    raw_stds = {k: np.array(v) for k, v in std_lists.items()}
    raw_ns = {k: np.array(v) for k, v in n_lists.items()}
    timesteps = np.array(eval_timesteps)
    sfx_parts = []
    if step > 1:
        sfx_parts.append(f"step={step}")
    if smooth_window > 1:
        sfx_parts.append(f"smooth w={smooth_window}")
    sfx = f" ({', '.join(sfx_parts)})" if sfx_parts else ""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Panel 1: depression-axis alignment
    _plot_smoothed(ax1, timesteps, raw["depression_align_r"], smooth_window, 'steelblue', None)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_ylabel("Content–PHQ-9 Alignment (r)\n[depression axis projection]")
    ax1.set_title(f"Depression Echo Chamber Analysis{sfx}")
    ax1.grid(True, alpha=0.3)

    # Panel 2: local vs. random
    _plot_smoothed(ax2, timesteps, raw["mean_local_proj"], smooth_window, 'teal',
                   'Local (neighbors)', linewidth=2,
                   raw_std=raw_stds["mean_local_proj"], raw_n=raw_ns["mean_local_proj"])
    _plot_smoothed(ax2, timesteps, raw["mean_rand_proj"], smooth_window, 'gray',
                   'Random baseline', linewidth=1.5, linestyle='--', fill_alpha=0.1,
                   raw_std=raw_stds["mean_rand_proj"], raw_n=raw_ns["mean_rand_proj"])
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Depression-Axis Similarity\n(neighbors vs. random)")
    ax2.legend(loc='best'); ax2.grid(True, alpha=0.3)

    _save_and_close(fig, save, path, filename, "depression_echo_chamber", show_fig)
    return fig, (ax1, ax2), raw


#============= PHQ-9 Assortativity + Degree-Weighted Mean ==============#

def plot_phq9_assortativity(network, path="", filename="", save=False, show_fig=True,
                            step=10, bin_size=50):
    """
    Standalone plot: PHQ-9 assortativity (left axis) + degree-weighted mean PHQ-9
    with cross-agent SE (right axis), shown as binned points with error bars.

    Computed every `step` timesteps (default 10 to match PHQ-9 update cycle).
    Then aggregated into bins of `bin_size` raw timesteps (default 50).
    - Assortativity: bin mean ± temporal std (how much it fluctuates across PHQ-9 cycles).
    - Degree-weighted PHQ-9: bin-midpoint value ± cross-agent SE.

    Uses ALL agents (no embedding filter), degree-weighted like metrics.degree_weighted_mean.
    """
    graph, id_to_idx = metrics.build_network_graph(network)
    undirected = graph.to_undirected()
    T = _get_T(network, include_phq9=True)
    eval_timesteps = list(range(0, T, step))

    # Precompute per-agent degree (number of connections)
    agent_degrees = np.array([len(a.agent_connections) for a in network.all_agents], dtype=float)
    total_degree = agent_degrees.sum()

    assort_list, dw_mean_list, dw_se_list, dw_sd_list = [], [], [], []

    print(f"Computing PHQ-9 assortativity (step={step}, {len(eval_timesteps)} points)...")
    for t in eval_timesteps:
        # Gather PHQ-9 for ALL agents (no embedding filter)
        phq9s = np.array([float(a.all_phq9_sumscores[t]) if a.all_phq9_sumscores[t] is not None else np.nan
                          for a in network.all_agents])
        valid_mask = ~np.isnan(phq9s)

        if valid_mask.sum() < 4:
            assort_list.append(np.nan); dw_mean_list.append(np.nan); dw_se_list.append(np.nan); dw_sd_list.append(np.nan)
            continue

        # Assortativity
        for node in undirected.nodes():
            idx = id_to_idx[node]
            undirected.nodes[node]['phq9'] = phq9s[idx] if valid_mask[idx] else 0.0
        try:
            assort_list.append(nx.numeric_assortativity_coefficient(undirected, 'phq9'))
        except Exception:
            assort_list.append(np.nan)

        # Degree-weighted mean and SE
        valid_phq9 = phq9s[valid_mask]
        valid_degrees = agent_degrees[valid_mask]
        n = len(valid_phq9)
        if total_degree > 0:
            weights = valid_degrees / valid_degrees.sum()
            dw_mean = np.average(valid_phq9, weights=weights)
            dw_std = np.sqrt(np.average((valid_phq9 - dw_mean) ** 2, weights=weights))
        else:
            dw_mean, dw_std = np.mean(valid_phq9), np.std(valid_phq9)
        dw_mean_list.append(dw_mean)
        dw_sd_list.append(dw_std)
        dw_se_list.append(dw_std / np.sqrt(n))

    raw_timesteps = np.array(eval_timesteps)
    raw_assort = np.array(assort_list)
    raw_dw_mean = np.array(dw_mean_list)
    raw_dw_sd = np.array(dw_sd_list)
    raw_dw_se = np.array(dw_se_list)

    # ── Bin aggregation ──
    points_per_bin = max(1, bin_size // step)
    n_points = len(eval_timesteps)
    n_bins = max(1, n_points // points_per_bin)

    bin_t, bin_assort_mean, bin_assort_std = [], [], []
    bin_dw_mean, bin_dw_min, bin_dw_max, bin_dw_sd = [], [], [], []

    for b in range(n_bins):
        start = b * points_per_bin
        end = min(start + points_per_bin, n_points)
        sl = slice(start, end)

        bin_t.append(np.nanmean(raw_timesteps[sl]))  # bin midpoint

        # Assortativity: mean ± temporal std (can be negative, that's fine)
        a = raw_assort[sl]
        bin_assort_mean.append(np.nanmean(a))
        bin_assort_std.append(np.nanstd(a))

        # Degree-weighted PHQ-9: mean with min–max band (can't go negative)
        bin_dw_mean.append(np.nanmean(raw_dw_mean[sl]))
        bin_dw_min.append(np.nanmin(raw_dw_mean[sl]))
        bin_dw_max.append(np.nanmax(raw_dw_mean[sl]))
        # Cross-agent SD: mean of per-timestep SDs within this bin
        bin_dw_sd.append(np.nanmean(raw_dw_sd[sl]))

    bin_t = np.array(bin_t)
    bin_assort_mean = np.array(bin_assort_mean)
    bin_assort_std = np.array(bin_assort_std)
    bin_dw_mean = np.array(bin_dw_mean)
    bin_dw_min = np.array(bin_dw_min)
    bin_dw_max = np.array(bin_dw_max)
    bin_dw_sd = np.array(bin_dw_sd)

    # ── Plot ──
    sfx = f" (binned, {bin_size} steps)"
    fig, ax_left = plt.subplots(figsize=(5, 4))

    # Left axis: assortativity with error bars (can go negative, no issue)
    ax_left.errorbar(bin_t, bin_assort_mean, yerr=bin_assort_std, color='steelblue',
                     fmt='o-', capsize=3, linewidth=1.5, markersize=4,
                     label='Assortativity ± SD')
    ax_left.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_left.set_xlabel("Time Step")
    ax_left.set_ylabel("PHQ-9 Assortativity (r)", color='steelblue')
    ax_left.tick_params(axis='y', labelcolor='steelblue')
    ax_left.grid(True, alpha=0.3)
    ax_left.set_title(f"PHQ-9 Assortativity & Degree-Weighted Mean{sfx}")

    # Right axis: degree-weighted mean PHQ-9 with min–max band (always ≥ 0)
    ax_right = ax_left.twinx()
    ax_right.plot(bin_t, bin_dw_mean, 's--', color='firebrick', linewidth=1.2,
                  markersize=4, label='DW mean PHQ-9')
    ax_right.fill_between(bin_t, bin_dw_min, bin_dw_max,
                          color='firebrick', alpha=0.15, label='Min–max range')
    ax_right.set_ylabel("Degree-Weighted PHQ-9 Score", color='firebrick')
    y_lo, y_hi = np.nanmin(bin_dw_min), np.nanmax(bin_dw_max)
    margin = max((y_hi - y_lo) * 0.1, 0.5)
    ax_right.set_ylim(max(0, y_lo - margin), min(27, y_hi + margin))
    ax_right.tick_params(axis='y', labelcolor='firebrick')

    # Combined legend
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)

    _save_and_close(fig, save, path, filename, "phq9_assortativity", show_fig)
    return fig, (ax_left, ax_right), {
        "bin_timesteps": bin_t,
        "bin_assort_mean": bin_assort_mean, "bin_assort_std": bin_assort_std,
        "bin_dw_phq9_mean": bin_dw_mean, "bin_dw_phq9_min": bin_dw_min, "bin_dw_phq9_max": bin_dw_max,
        "bin_dw_phq9_sd": bin_dw_sd,
        "raw_timesteps": raw_timesteps, "raw_assort": raw_assort,
        "raw_dw_mean": raw_dw_mean, "raw_dw_sd": raw_dw_sd, "raw_dw_se": raw_dw_se,
    }


#============= PHQ-9 Neighbor Correlation ===============================#

def plot_phq9_neighbor_correlation(network, path="", filename="", save=False, show_fig=True,
                                   time_range=None):
    """
    Scatter of each agent's mean PHQ-9 vs. their neighbors' mean PHQ-9.

    A positive correlation means depressed agents are surrounded by depressed neighbors.

    Args:
        time_range (tuple, optional): (start, end) timestep indices for averaging.
            Supports negative indexing from the end, e.g. (-50, None) = last 50 steps.
            None uses all timesteps.
    """
    from scipy.stats import pearsonr

    graph, id_to_idx = metrics.build_network_graph(network)
    undirected = graph.to_undirected()
    T = _get_T(network, include_phq9=True)

    # Resolve time range (supports negative indexing)
    if time_range is not None:
        t_start, t_end = time_range
        if t_start is not None and t_start < 0:
            t_start = max(0, T + t_start)
        if t_end is not None and t_end < 0:
            t_end = max(0, T + t_end)
        t_start = t_start if t_start is not None else 0
        t_end   = t_end   if t_end   is not None else T
    else:
        t_start, t_end = 0, T
    t_range = range(t_start, t_end)

    # PHQ-9 averaged over the selected time window per agent
    mean_phq9 = {}
    for agent in network.all_agents:
        scores = [agent.all_phq9_sumscores[t] for t in t_range
                  if agent.all_phq9_sumscores[t] is not None]
        if scores:
            mean_phq9[agent.ID] = np.mean(scores)

    own, neigh_mean = [], []
    for agent in network.all_agents:
        if agent.ID not in mean_phq9:
            continue
        neighbor_scores = [mean_phq9[nid] for nid in undirected.neighbors(agent.ID) if nid in mean_phq9]
        if not neighbor_scores:
            continue
        own.append(mean_phq9[agent.ID])
        neigh_mean.append(np.mean(neighbor_scores))

    own, neigh_mean = np.array(own), np.array(neigh_mean)
    r, p = pearsonr(own, neigh_mean)

    # Plot
    range_label = f"t=[{t_start}:{t_end}]" if time_range is not None else "all t"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(own, neigh_mean, alpha=0.5, s=25, c=own, cmap='RdYlGn_r', edgecolors='none')

    m, b = np.polyfit(own, neigh_mean, 1)
    x_line = np.array([own.min(), own.max()])
    ax.plot(x_line, m * x_line + b, color='black', linewidth=1.5,
            label=f'r = {r:.3f} (p = {p:.1e})')

    lim = [min(own.min(), neigh_mean.min()) - 1, max(own.max(), neigh_mean.max()) + 1]
    ax.plot(lim, lim, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.set_xlabel("Agent's Mean PHQ-9 Score")
    ax.set_ylabel("Neighbors' Mean PHQ-9 Score")
    ax.set_title(f"PHQ-9 Neighbor Correlation ({range_label})")
    ax.legend(loc='upper left')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    _save_and_close(fig, save, path, filename, "phq9_neighbor_correlation", show_fig)
    return fig, ax, {"r": r, "p": p}


# ── Prompt optimizer plots ────────────────────────────────────────────────────

def _phq9_severity_color(phq9: int) -> str:
    if phq9 >= 20: return "#8B0000"   # severe — dark red
    if phq9 >= 15: return "#D73027"   # moderately severe — red
    if phq9 >= 10: return "#FC8D59"   # moderate — orange
    if phq9 >= 5:  return "#FEE090"   # mild — yellow
    return "#91CF60"                   # minimal — green


def plot_optimizer_trajectory(trajectory_csv: str, output_dir: str, title: str, mode: str = "tweets"):
    """Line plot of train and validation scores over optimizer steps.

    mode='tweets': higher mean_score (0-10) is better.
    mode='phq9':   lower mean_score (MAE) is better.
    """
    df = pd.read_csv(trajectory_csv)

    fig, ax = plt.subplots(figsize=(9, 5))

    palette = {"train": "#4292C6", "val": "#E6550D"}
    labels  = {"train": "Train", "val": "Validation"}

    for split in ["train", "val"]:
        sub = df[df["split"] == split].sort_values("step")
        if sub.empty:
            continue
        ax.plot(sub["step"], sub["mean_score"], marker="o", markersize=4,
                color=palette[split], label=labels[split], linewidth=1.8)
        ax.fill_between(
            sub["step"],
            sub["mean_score"] - sub["std_score"],
            sub["mean_score"] + sub["std_score"],
            color=palette[split], alpha=0.15,
        )

    ax.set_xlabel("Optimizer step")
    if mode == "phq9":
        ax.set_ylabel("MAE  (↓ better)")
    else:
        ax.set_ylabel("Quality score 0–10  (↑ better)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = os.path.join(output_dir, "trajectory.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Trajectory plot → {out}")
    return out


def plot_test_scores_by_phq9(per_phq9: dict, output_dir: str, title: str, mode: str = "tweets"):
    """Bar plot of test performance grouped by depression severity category.

    per_phq9 keys are integer PHQ-9 values.
    For tweets: values have key 'avg_score'.
    For phq9:   values have key 'avg_mae'.
    Bars show weighted mean per category; error bars show std across PHQ-9 averages.
    """
    from matplotlib.patches import Patch

    score_key = "avg_score" if mode == "tweets" else "avg_mae"

    categories = [
        ("Minimal\n(0–4)",       range(0,  5),  "#91CF60"),
        ("Mild\n(5–9)",          range(5, 10),  "#FEE090"),
        ("Moderate\n(10–14)",    range(10, 15), "#FC8D59"),
        ("Mod. severe\n(15–19)", range(15, 20), "#D73027"),
        ("Severe\n(20–27)",      range(20, 28), "#8B0000"),
    ]

    labels, means, stds, totals, colors = [], [], [], [], []
    for label, phq_range, color in categories:
        vals, ns = [], []
        for k in phq_range:
            if k in per_phq9:
                vals.append(per_phq9[k].get(score_key, 0.0))
                ns.append(per_phq9[k].get("n_samples", 1))
        if not vals:
            continue
        ns = np.array(ns, dtype=float)
        vals = np.array(vals)
        wmean = float(np.average(vals, weights=ns))
        wstd  = float(np.sqrt(np.average((vals - wmean) ** 2, weights=ns)))
        labels.append(label)
        means.append(wmean)
        stds.append(wstd)
        totals.append(int(ns.sum()))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.8,
                  yerr=stds, capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "black"})

    for bar, n in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.5,
                f"n={n}", ha="center", va="center", fontsize=9, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Depression severity")
    if mode == "phq9":
        ax.set_ylabel("MAE  (↓ better)")
    else:
        ax.set_ylabel("Quality score 0–10  (↑ better)")
        ax.set_ylim(0, 10)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    out = os.path.join(output_dir, "test_by_phq9.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_test_mae_and_bias_by_phq9(per_phq9: dict, output_dir: str, title: str):
    """Side-by-side bars: MAE per PHQ-9 category (left) and signed bias (right).

    per_phq9 keys are integer PHQ-9 values; values must contain
    ``avg_mae``, ``avg_bias``, ``std_bias``, and ``n_samples``.

    The bias panel uses ``mean(pred − true)`` so positive bars mean the model
    over-estimates the true score for that severity bracket, negative bars mean
    under-estimation. Error bars on both panels show the (weighted) std of
    per-PHQ-9 averages within the category — same convention as
    ``plot_test_scores_by_phq9``.
    """
    categories = [
        ("Minimal\n(0–4)",       range(0,  5),  "#91CF60"),
        ("Mild\n(5–9)",          range(5, 10),  "#FEE090"),
        ("Moderate\n(10–14)",    range(10, 15), "#FC8D59"),
        ("Mod. severe\n(15–19)", range(15, 20), "#D73027"),
        ("Severe\n(20–27)",      range(20, 28), "#8B0000"),
    ]

    labels, mae_means, mae_stds = [], [], []
    bias_means, bias_stds, totals, colors = [], [], [], []
    for label, phq_range, color in categories:
        maes, biases, ns = [], [], []
        for k in phq_range:
            if k in per_phq9:
                maes.append(per_phq9[k].get("avg_mae", 0.0))
                biases.append(per_phq9[k].get("avg_bias", 0.0))
                ns.append(per_phq9[k].get("n_samples", 1))
        if not maes:
            continue
        ns_arr = np.array(ns, dtype=float)
        mae_arr = np.array(maes)
        bias_arr = np.array(biases)
        wmae = float(np.average(mae_arr, weights=ns_arr))
        wmae_std = float(np.sqrt(np.average((mae_arr - wmae) ** 2, weights=ns_arr)))
        wbias = float(np.average(bias_arr, weights=ns_arr))
        wbias_std = float(np.sqrt(np.average((bias_arr - wbias) ** 2, weights=ns_arr)))
        labels.append(label)
        mae_means.append(wmae)
        mae_stds.append(wmae_std)
        bias_means.append(wbias)
        bias_stds.append(wbias_std)
        totals.append(int(ns_arr.sum()))
        colors.append(color)

    fig, (ax_mae, ax_bias) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    x = np.arange(len(labels))

    bars_mae = ax_mae.bar(x, mae_means, color=colors, edgecolor="white", linewidth=0.8,
                          yerr=mae_stds, capsize=5,
                          error_kw={"elinewidth": 1.5, "ecolor": "black"})
    for bar, n in zip(bars_mae, totals):
        ax_mae.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5,
                    f"n={n}", ha="center", va="center", fontsize=9, color="black")
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(labels, fontsize=10)
    ax_mae.set_xlabel("Depression severity")
    ax_mae.set_ylabel("MAE  (↓ better)")
    ax_mae.set_title("MAE by category")
    ax_mae.grid(True, axis="y", alpha=0.25)
    ax_mae.set_axisbelow(True)

    bars_bias = ax_bias.bar(x, bias_means, color=colors, edgecolor="white", linewidth=0.8,
                            yerr=bias_stds, capsize=5,
                            error_kw={"elinewidth": 1.5, "ecolor": "black"})
    ax_bias.axhline(0, color="black", linewidth=1.0, alpha=0.8)
    if bias_means:
        bias_span = max(abs(min(bias_means + [0])), abs(max(bias_means + [0])), 1e-6)
        offset = 0.05 * bias_span
        for bar, n in zip(bars_bias, totals):
            h = bar.get_height()
            if h >= 0:
                ax_bias.text(bar.get_x() + bar.get_width() / 2, h + offset,
                             f"n={n}", ha="center", va="bottom", fontsize=8)
            else:
                ax_bias.text(bar.get_x() + bar.get_width() / 2, h - offset,
                             f"n={n}", ha="center", va="top", fontsize=8)
    ax_bias.set_xticks(x)
    ax_bias.set_xticklabels(labels, fontsize=10)
    ax_bias.set_xlabel("Depression severity")
    ax_bias.set_ylabel("Bias = mean(pred − true)\n(+ over-estimate / − under-estimate)")
    ax_bias.set_title("Signed bias by category")
    ax_bias.grid(True, axis="y", alpha=0.25)
    ax_bias.set_axisbelow(True)

    fig.suptitle(title)
    fig.tight_layout()
    out = os.path.join(output_dir, "test_scores_by_phq9.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Test MAE+bias plot → {out}")
    return out


def plot_cv_results(cv_records: list, mean_val_mae: float, std_val_mae: float,
                    output_dir: str, title: str):
    """Bar plot of best per-fold val MAE with a horizontal mean line and ±1 std band.

    Used as a pre-flight diagnostic before final BERT training: small bars and a
    narrow std band mean the model is stable across data partitions; tall bars
    or a wide band mean a single 80/10/10 split's reported MAE is partly a
    function of which 10% it happened to draw.
    """
    folds = [r["fold"] for r in cv_records]
    val_maes = [r["best_val_mae"] for r in cv_records]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(folds, val_maes, color="#4292C6", edgecolor="white", linewidth=0.8)
    ax.axhspan(mean_val_mae - std_val_mae, mean_val_mae + std_val_mae,
               color="black", alpha=0.10, label=f"±1 std ({std_val_mae:.3f})")
    ax.axhline(mean_val_mae, color="black", linewidth=1.5, linestyle="--",
               label=f"Mean = {mean_val_mae:.3f}")

    for bar, mae in zip(bars, val_maes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(val_maes),
                f"{mae:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_xlabel("Cross-validation fold")
    ax.set_ylabel("Best val MAE  (↓ better)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="best")

    out = os.path.join(output_dir, "cv_results.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"CV plot → {out}")
    return out


# =========================================================================== #
# Estimator comparison bar plots (MAE + signed bias, with error bars).
#
# Two figures, each a MAE panel + a bias panel side-by-side. Error bars are the
# SD of the per-sample errors (|error| for MAE, signed error for bias), computed
# per seed and averaged over seeds — the within-run spread, not seed-to-seed
# variability. Everything is recomputed from the per-sample test_raw_scores.csv /
# seed<seed>.csv files (columns true_phq9, pred_phq9), so bars/error-bars/table
# share one source of truth.
#
#   Figure 1: BERT non-FT (human-opt) | fine-tuned (human-opt) | non-FT (synthetic)
#   Figure 2: BERT vs post-assessment prompt, each on {synthetic, human-opt} test
#
# Test-set alignment (figure 2): both methods are scored on the SAME data sources
# so the synthetic-vs-human gap is the distribution shift, not a change of test
# set. The "synthetic" (in-distribution) side is the BERT regression test blocks:
# BERT is its own 5-seed average (each seed on its held-out split, one of which is
# test_blocks_seed35.csv), and the prompt is scored on test_blocks_seed35.csv via
# the eval_on_test_blocks_seed35[/_minimal] subdirs — NOT the prompt's own
# optimisation split. The "human-opt" side is the 300-block data/finetune/
# test_posts.csv for every bar (BERT eval_baseline, prompt eval_on_human300,
# minimal minimal_human300).
#
# Run via the shell wrapper run_eval_comparison.sh, or directly:
#   PYTHONPATH=src python -m utils.visualization --out-dir data/test_post/method_comparison
# =========================================================================== #
_EVAL_MODEL_SHORT = "Qwen3.5-27B"
# Match the prompt-sensitivity (SA) palette used elsewhere in the thesis
# (sa_analyze._COLOUR_BY_NAME): Agent=blue, Joint=orange, Neighbour=dark-red.
_EVAL_C_SYNTH = "#2e7ebc"   # synthetic (in-distribution) test  (SA "Agent" blue)
_EVAL_C_HUMAN = "#d96907"   # human-optimized-prompt test       (SA "Joint" orange)
_EVAL_C_FT = "#8d2c03"      # fine-tuned bar in figure 1         (SA "Neighbour" dark-red)

# Subdir under each prompt seed dir holding the OPTIMISED prompt scored on the
# aligned synthetic test (the BERT regression test blocks, test_blocks_seed35.csv),
# rather than the prompt's own optimisation split — see the alignment note above.
_EVAL_PROMPT_SYNTH_SUBDIR = "eval_on_test_blocks_seed35"

# Optional minimal-prompt bars in figure 2 (added only if scored on disk).
# Generated with prompt_optimizer.py --mode phq9-rerun-test --instruction-filename
# minimal_instruction.txt --result-subdir <name>, per seed under prompt_dir. Only
# the seed23 minimal run exists on disk, so this group is a single run (no seed
# error bar); both bars must come from the SAME seed for the shift to be valid.
# Synthetic uses the same test_blocks_seed35 set as the optimised-prompt synthetic
# bar; human-opt uses the 300-block data/finetune/test_posts.csv so that BERT
# (eval_baseline), the optimised prompt (eval_on_human300) and the minimal prompt
# (minimal_human300) are all scored on the SAME 300 human-opt blocks.
_EVAL_MINIMAL_SYNTH_SUBDIR = "eval_on_test_blocks_seed35_minimal"  # minimal prompt on test_blocks_seed35
_EVAL_MINIMAL_HUMAN_SUBDIR = "minimal_human300"                    # minimal prompt on the 300-block human-opt set


def _eval_per_seed_stats(raw_csv: str) -> dict:
    """MAE, bias and the SD of the per-sample errors for one seed's raw scores."""
    df = pd.read_csv(raw_csv)
    err = df["pred_phq9"].astype(float) - df["true_phq9"].astype(float)
    return {
        "mae": float(err.abs().mean()),
        "mae_sd": float(err.abs().std(ddof=1)),     # spread of |error|
        "bias": float(err.mean()),
        "bias_sd": float(err.std(ddof=1)),          # spread of signed error
        "n": int(len(df)),
    }


def _eval_aggregate(raw_csvs: list, label: str) -> dict:
    """Bar height = mean over seeds of the per-seed mean. Two error flavours:
        *_err_sample : mean over seeds of the per-seed per-sample SD (within-run spread)
        *_err_seed   : SD across the per-seed means (between-seed variability)
    """
    if not raw_csvs:
        raise SystemExit(f"[{label}] no raw-score CSVs found — check the paths/seeds.")
    rows = [_eval_per_seed_stats(p) for p in raw_csvs]
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    sd_across = lambda k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0
    return {
        "label": label,
        "mae": mean("mae"), "bias": mean("bias"),
        "mae_err_sample": mean("mae_sd"), "bias_err_sample": mean("bias_sd"),
        "mae_err_seed": sd_across("mae"), "bias_err_seed": sd_across("bias"),
        "n_seeds": len(rows), "n_per_seed": int(np.mean([r["n"] for r in rows])),
    }


def _eval_aggregate_optional(raw_csvs: list, label: str):
    """Like _eval_aggregate but returns None (instead of raising) when no CSVs
    are present — used for optional bars whose data may not be on disk yet."""
    if not raw_csvs:
        return None
    return _eval_aggregate(raw_csvs, label)


def _eval_seed_csvs(eval_dir: str, seeds: list) -> list:
    """Per-sample seed<seed>.csv files in an eval dir (skip *_summary / aggregate)."""
    out = []
    for p in sorted(glob.glob(os.path.join(eval_dir, "seed*.csv"))):
        m = re.fullmatch(r"seed(\d+)\.csv", os.path.basename(p))
        if m and int(m.group(1)) in seeds:
            out.append(p)
    return out


def _eval_perseed_dir_csvs(base: str, seeds: list, rel: str) -> list:
    """<base>/<MODEL_SHORT>_seed<seed>/<rel> for each existing seed."""
    out = []
    for s in seeds:
        p = os.path.join(base, f"{_EVAL_MODEL_SHORT}_seed{s}", rel)
        if os.path.isfile(p):
            out.append(p)
    return out


def collect_eval_comparison(bert_dir: str, bert_ft_dir: str, prompt_dir: str,
                            prompt_eval_subdir: str, bert_seeds: list, prompt_seeds: list,
                            prompt_synth_subdir: str = _EVAL_PROMPT_SYNTH_SUBDIR) -> dict:
    """Resolve every bar's raw-score files and aggregate them into a stats dict."""
    eval_baseline = os.path.join(bert_dir, "eval_baseline")
    eval_finetuned = os.path.join(bert_ft_dir, "eval_finetuned")
    stats = {
        "bert_nonft_human": _eval_aggregate(
            _eval_seed_csvs(eval_baseline, bert_seeds), "BERT non-FT / human-opt"),
        "bert_ft_human": _eval_aggregate(
            _eval_seed_csvs(eval_finetuned, bert_seeds), "BERT fine-tuned / human-opt"),
        "bert_nonft_synth": _eval_aggregate(
            _eval_perseed_dir_csvs(bert_dir, bert_seeds, "test_raw_scores.csv"), "BERT non-FT / synthetic"),
        # Prompt synthetic = optimised prompt on the BERT test blocks (aligned),
        # NOT prompt_dir/{seed}/test_raw_scores.csv (the prompt's own opt split).
        "prompt_synth": _eval_aggregate(
            _eval_perseed_dir_csvs(prompt_dir, prompt_seeds,
                                   os.path.join(prompt_synth_subdir, "test_raw_scores.csv")), "Prompt(opt) / synthetic"),
        "prompt_human": _eval_aggregate(
            _eval_perseed_dir_csvs(prompt_dir, prompt_seeds,
                                   os.path.join(prompt_eval_subdir, "test_raw_scores.csv")), "Prompt(opt) / human-opt"),
    }
    # Optional minimal-prompt group — added to figure 2 only if scored on disk.
    minimal_synth = _eval_aggregate_optional(
        _eval_perseed_dir_csvs(prompt_dir, prompt_seeds,
                               os.path.join(_EVAL_MINIMAL_SYNTH_SUBDIR, "test_raw_scores.csv")),
        "Prompt(min) / synthetic")
    minimal_human = _eval_aggregate_optional(
        _eval_perseed_dir_csvs(prompt_dir, prompt_seeds,
                               os.path.join(_EVAL_MINIMAL_HUMAN_SUBDIR, "test_raw_scores.csv")),
        "Prompt(min) / human-opt")
    if minimal_synth is not None and minimal_human is not None:
        stats["minimal_synth"] = minimal_synth
        stats["minimal_human"] = minimal_human
    elif minimal_synth is not None or minimal_human is not None:
        print("[eval] minimal-prompt group skipped: need BOTH "
              f"{_EVAL_MINIMAL_SYNTH_SUBDIR}/ and {_EVAL_MINIMAL_HUMAN_SUBDIR}/ "
              "test_raw_scores.csv under the prompt seed dirs.")
    return stats


def _eval_annotate(ax, x, height, err, pad, fmt="{:.2f}", fontsize=8):
    """Value label just outside the error bar (handles negative bars for bias)."""
    up = height >= 0
    y = height + (err + pad) * (1 if up else -1)
    ax.annotate(fmt.format(height), (x, y), ha="center",
                va="bottom" if up else "top", fontsize=fontsize)


def _eval_set_ylim(ax, heights, errs, top_headroom=0.22, bot_headroom=0.16, floor_zero=False):
    """Autoscale ignores text labels — expand limits so value/Δ annotations fit.
    floor_zero pins the bottom at 0 (for MAE, which is non-negative).
    Returns (ytop, range) for placing gap labels within the new headroom.
    """
    lo = min(0.0, min(h - e for h, e in zip(heights, errs)))
    hi = max(0.0, max(h + e for h, e in zip(heights, errs)))
    rng = (hi - lo) or 1.0
    bottom = 0.0 if floor_zero else lo - bot_headroom * rng
    ax.set_ylim(bottom, hi + top_headroom * rng)
    return hi, rng


def _eval_style(ax, ylabel, caption):
    """Bias-zero line, y-grid, and the panel caption placed UNDERNEATH (as the
    x-label, below the category ticks) — e.g. "(a) Mean absolute error"."""
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel(caption, fontsize=10, labelpad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def plot_eval_finetune(stats: dict, out_path: str, err_mode: str = "sample"):
    """Figure 1 — BERT regressor: fine-tuning recovers the distribution shift.
    err_mode: "sample" (SD of per-sample errors) or "seed" (between-seed SD)."""
    order = ["bert_nonft_human", "bert_ft_human", "bert_nonft_synth"]
    xlabels = ["Non-finetuned\n(human-opt)",
               "Fine-tuned\n(human-opt)",
               "Non-finetuned\n(synthetic)"]
    colours = [_EVAL_C_HUMAN, _EVAL_C_FT, _EVAL_C_SYNTH]
    x = np.arange(len(order))
    sfx = "seed" if err_mode == "seed" else "sample"

    fig, (ax_mae, ax_bias) = plt.subplots(1, 2, figsize=(6, 2.8))
    for ax, key, ekey, ylab, cap in [
        (ax_mae, "mae", f"mae_err_{sfx}", "MAE", "(a)"),
        (ax_bias, "bias", f"bias_err_{sfx}", "Bias", "(b)"),
    ]:
        h = [stats[k][key] for k in order]
        e = [stats[k][ekey] for k in order]
        ax.bar(x, h, yerr=e, color=colours, edgecolor="black", linewidth=0.6,
               capsize=4, error_kw=dict(elinewidth=1.0))
        _, rng = _eval_set_ylim(ax, h, e, floor_zero=(key == "mae"))
        for xi, hi, ei in zip(x, h, e):
            _eval_annotate(ax, xi, hi, ei, pad=0.03 * rng, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=7.5)
        _eval_style(ax, ylab, cap)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_eval_bert_vs_prompt(stats: dict, out_path: str, err_mode: str = "seed"):
    """Figure 2 — robustness to distribution shift: BERT vs the post-assessment prompt.
    err_mode: "seed" (between-seed SD) or "sample" (SD of per-sample errors)."""
    from matplotlib.patches import Patch

    has_minimal = "minimal_synth" in stats and "minimal_human" in stats
    # Relabel the optimized prompt group when a minimal-prompt group is present,
    # so the two PHQ-9 prompts are distinguishable on the x-axis.
    prompt_label = "Optimized\nprompt" if has_minimal else "PHQ-9 prompt"
    methods = [("BERT+MLP", "bert_nonft_synth", "bert_nonft_human"),
               (prompt_label, "prompt_synth", "prompt_human")]
    if has_minimal:
        methods.append(("Minimal\nprompt", "minimal_synth", "minimal_human"))
    x = np.arange(len(methods))
    w = 0.38
    sfx = "seed" if err_mode == "seed" else "sample"

    fig, (ax_mae, ax_bias) = plt.subplots(1, 2, figsize=(7.2 if has_minimal else 6, 2.8))
    for ax, key, ekey, ylab, cap in [
        (ax_mae, "mae", f"mae_err_{sfx}", "MAE", "(a)"),
        (ax_bias, "bias", f"bias_err_{sfx}", "Bias", "(b)"),
    ]:
        bars = [(-w / 2, 1, _EVAL_C_SYNTH), (+w / 2, 2, _EVAL_C_HUMAN)]
        all_h = [stats[m[sub]][key] for _, sub, _ in bars for m in methods]
        all_e = [stats[m[sub]][ekey] for _, sub, _ in bars for m in methods]
        _, rng = _eval_set_ylim(ax, all_h, all_e, top_headroom=0.15, floor_zero=(key == "mae"))
        for off, sub, colour in bars:
            h = [stats[m[sub]][key] for m in methods]
            e = [stats[m[sub]][ekey] for m in methods]
            ax.bar(x + off, h, w, yerr=e, color=colour, edgecolor="black",
                   linewidth=0.6, capsize=4, error_kw=dict(elinewidth=1.0))
            for xi, hi, ei in zip(x + off, h, e):
                _eval_annotate(ax, xi, hi, ei, pad=0.025 * rng, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in methods], fontsize=7.5)
        _eval_style(ax, ylab, cap)

    legend = [Patch(facecolor=_EVAL_C_SYNTH, edgecolor="black", label="synthetic"),
              Patch(facecolor=_EVAL_C_HUMAN, edgecolor="black", label="human-opt")]
    ax_mae.legend(handles=legend, loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot] {out_path}")


def _print_eval_table(stats: dict):
    print(f"\n{'bar':<28}{'MAE':>8}{'±smpl':>8}{'±seed':>8}"
          f"{'bias':>9}{'±smpl':>8}{'±seed':>8}{'seeds':>7}{'n/seed':>8}")
    for v in stats.values():
        print(f"{v['label']:<28}{v['mae']:>8.3f}{v['mae_err_sample']:>8.3f}{v['mae_err_seed']:>8.3f}"
              f"{v['bias']:>9.3f}{v['bias_err_sample']:>8.3f}{v['bias_err_seed']:>8.3f}"
              f"{v['n_seeds']:>7d}{v['n_per_seed']:>8d}")


def run_eval_comparison(argv=None):
    """CLI entry: build the two estimator-comparison figures + print the table."""
    p = argparse.ArgumentParser(description="MAE/bias comparison bar plots (BERT vs prompt).")
    p.add_argument("--bert-dir", default="data/test_post/bert_regression",
                   help="Holds {MODEL}_seed*/ (synthetic test) and eval_baseline/ (human-opt test).")
    p.add_argument("--bert-ft-dir", default="data/test_post/bert_regression_finetuned",
                   help="Holds eval_finetuned/ (fine-tuned regressor on the human-opt test set).")
    p.add_argument("--prompt-dir", default="data/test_post/optimized_phq9",
                   help="Holds {MODEL}_seed*/ (synthetic test) and the eval-on-prompt subdir (human-opt).")
    p.add_argument("--prompt-eval-subdir", default="eval_on_human300",
                   help="Subdir under each prompt seed dir holding the human-opt-test raw scores "
                        "(the 300-block data/finetune/test_posts.csv eval, paired with BERT's eval_baseline).")
    p.add_argument("--prompt-synth-subdir", default=_EVAL_PROMPT_SYNTH_SUBDIR,
                   help="Subdir under each prompt seed dir holding the aligned synthetic-test raw scores "
                        "(the prompt scored on the BERT test blocks, test_blocks_seed35.csv).")
    p.add_argument("--bert-seeds", type=int, nargs="+", default=[34, 35, 36, 37, 38])
    p.add_argument("--prompt-seeds", type=int, nargs="+", default=[23, 24, 25, 32, 33])
    p.add_argument("--out-dir", default="data/test_post/method_comparison")
    args = p.parse_args(argv)

    stats = collect_eval_comparison(args.bert_dir, args.bert_ft_dir, args.prompt_dir,
                                    args.prompt_eval_subdir, args.bert_seeds, args.prompt_seeds,
                                    prompt_synth_subdir=args.prompt_synth_subdir)
    _print_eval_table(stats)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_eval_finetune(stats, os.path.join(args.out_dir, "fig1_bert_finetune.png"))
    plot_eval_bert_vs_prompt(stats, os.path.join(args.out_dir, "fig2_bert_vs_prompt_robustness.png"))
    print(f"\n[done] figures under {args.out_dir}/")


if __name__ == "__main__":
    run_eval_comparison()