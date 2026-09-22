"""Plotting hub: network drawings, CDS and entrainment plots, PHQ-9 structure, prompt-optimizer figures and the eval-comparison / multimodel CLI."""
"""Plotting hub: network drawings, CDS and entrainment plots, PHQ-9 structure, prompt-optimizer figures and the eval-comparison / multimodel CLI."""
import seaborn as sns
import os
import argparse
import glob
import json
import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from . import metrics
from . import assessors
import pandas as pd

def print_network_phq9(network, path="", filename="default.png", save=False, show_fig=False):
    """Print network at one single iteration

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
    """Print network at one single iteration

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
    '''Plot PHQ-9 scores against degree weighted by connection weights.
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
    '''This function bins fractions of distorted neighbors, and plots the probability corresponding to that to tweet.
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
    '''This function plots the fraction of distorted tweets per round.
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
    '''This function plots the running mean fraction of distorted tweets over rounds.
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
    """Two-panel figure: (a) UMAP/PCA trajectory, (b) assortativity + DW mean PHQ-9.

    Args:
        assort_data (dict): Pre-computed output from plot_phq9_assortativity, containing
            bin_timesteps, bin_assort_mean, bin_assort_std, bin_dw_phq9_mean,
            bin_dw_phq9_min, bin_dw_phq9_max, bin_dw_phq9_sd.
            If None, panel (b) is left empty.
        use_sd_band (bool): If True, show mean ± cross-agent SD band instead of
            the min-max range on panel (b). Default False (min-max).
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

        # Flag start and end points, offset text toward trajectory center
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


def plot_entrainment_grid(cell_trajs, cell_phq9, row_titles, col_titles,
                          reduction="PCA", embedding="MentalBERT",
                          phq9_vmin=None, phq9_vmax=None,
                          path="", filename="", save=False, show_fig=False):
    """Small-multiple grid of per-seed embedding-entrainment trajectories.

    Each cell is one run's mean-embedding trajectory over sliding windows in 2D, one
    dot per window coloured by that window's mean PHQ-9 (RdYlGn_r). The caller fits the
    reduction once per row, so panels in a row share coordinates; each row gets its own
    PHQ-9 colour scale unless `phq9_vmin` / `phq9_vmax` pin it.

    Args:
        cell_trajs (dict): {(row, col): (T, 2) array}; missing cells are drawn empty.
        cell_phq9 (dict): {(row, col): (T,) array} of per-window mean PHQ-9.
        row_titles, col_titles (list): row (topology) and column (seed) labels.
        reduction, embedding (str): names shown in the title.
        phq9_vmin, phq9_vmax (float | None): colour limits; None = per-row scale.
        path, filename, save, show_fig: save controls ({path}/{filename}.png, dpi 300).

    Returns:
        tuple: (fig, axes).
    """
    n_rows = len(row_titles)
    n_cols = len(col_titles)

    # Per-row PHQ-9 colour scales: each setting (row) gets its OWN vmin/vmax and
    # its OWN colourbar, so within-row colour contrast isn't compressed onto one
    # figure-wide scale. An explicit phq9_vmin/vmax pins that bound on every row;
    # left None (the default), each row is scaled to its own finite PHQ-9 data.
    def _row_limits(r):
        vals = [np.asarray(cell_phq9.get((r, c)), float).ravel()
                for c in range(n_cols)
                if cell_phq9.get((r, c)) is not None
                and np.asarray(cell_phq9.get((r, c))).size]
        lo, hi = phq9_vmin, phq9_vmax
        if vals:
            finite = np.concatenate(vals)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                if lo is None:
                    lo = float(finite.min())
                if hi is None:
                    hi = float(finite.max())
        return lo, hi

    # Small panels (no titles, just row labels + one colourbar per row); extra
    # width budgets for the four per-row colourbars stacked on the right.
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.05 * n_cols + 0.6, 1.05 * n_rows),
                             squeeze=False, layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01,
                                    wspace=0.01, hspace=0.01)

    for r in range(n_rows):
        rvmin, rvmax = _row_limits(r)
        sc_row = None                       # last coloured scatter handle in row r
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(alpha=0.25)

            traj = cell_trajs.get((r, c))
            if traj is not None and np.asarray(traj).shape[0] > 0:
                traj = np.asarray(traj)
                phq9 = cell_phq9.get((r, c))
                ax.plot(traj[:, 0], traj[:, 1], color="gray", alpha=0.4,
                        linewidth=0.8, zorder=1)
                if phq9 is not None and np.asarray(phq9).shape[0] == traj.shape[0]:
                    sc_row = ax.scatter(traj[:, 0], traj[:, 1], c=np.asarray(phq9),
                                        cmap="RdYlGn_r", vmin=rvmin, vmax=rvmax,
                                        s=10, alpha=0.9, edgecolors="black",
                                        linewidths=0.2, zorder=2)
                else:
                    ax.scatter(traj[:, 0], traj[:, 1], s=10, alpha=0.9,
                               edgecolors="black", linewidths=0.2, zorder=2)
                # Light start (square) / end (star) markers, readable at grid scale.
                ax.scatter(traj[0, 0], traj[0, 1], marker="s", s=22,
                           facecolors="none", edgecolors="black",
                           linewidths=0.8, zorder=3)
                ax.scatter(traj[-1, 0], traj[-1, 1], marker="*", s=45,
                           facecolors="none", edgecolors="black",
                           linewidths=0.8, zorder=3)

            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=7)

        # One colourbar per row, each on that row's own PHQ-9 scale.
        if sc_row is not None:
            cbar = fig.colorbar(sc_row, ax=list(axes[r]), shrink=0.9,
                                pad=0.01, fraction=0.045)
            cbar.set_label("Mean PHQ-9", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

    if save and path and filename:
        os.makedirs(path, exist_ok=True)
        out = os.path.join(path, f"{filename}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[plot] wrote {out}")
    if show_fig:
        plt.show()
    plt.close(fig)
    return fig, axes


def _phq9_limits(phq9s, vmin, vmax):
    """Fill (vmin, vmax) from the finite PHQ-9 values when not supplied."""
    if (vmin is None or vmax is None) and phq9s:
        allv = np.concatenate([np.asarray(p, float).ravel() for p in phq9s
                               if np.asarray(p).size])
        finite = allv[np.isfinite(allv)] if allv.size else allv
        if finite.size:
            if vmin is None:
                vmin = float(finite.min())
            if vmax is None:
                vmax = float(finite.max())
    return vmin, vmax


def _draw_overlay(ax, trajs, phq9s, vmin, vmax, equal_aspect=False,
                  show_ticks=False, line_color="0.7", dot_size=10):
    """Overlay every trajectory of one setting into a single axes.

    Seeds are NOT visually distinguished (faint shared-colour connecting lines);
    dots are windows coloured by mean PHQ-9 (green→red). start = square, end =
    star. Returns the last scatter handle (for a colourbar).
    """
    sc = None
    for tr, pq in zip(trajs, phq9s):
        tr = np.asarray(tr)
        ax.plot(tr[:, 0], tr[:, 1], color=line_color, alpha=0.4, linewidth=0.7,
                zorder=1)
        sc = ax.scatter(tr[:, 0], tr[:, 1], c=np.asarray(pq), cmap="RdYlGn_r",
                        vmin=vmin, vmax=vmax, s=dot_size, alpha=0.9,
                        edgecolors="none", zorder=2)
        ax.scatter(tr[0, 0], tr[0, 1], marker="s", s=2 * dot_size,
                   facecolors="none", edgecolors="black", linewidths=0.6, zorder=3)
        ax.scatter(tr[-1, 0], tr[-1, 1], marker="*", s=4 * dot_size,
                   facecolors="none", edgecolors="black", linewidths=0.6, zorder=3)
    if show_ticks:
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(alpha=0.25)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    return sc


def plot_entrainment_overlay(trajs, phq9s, seeds=None, row_title="",
                             reduction="PCA", embedding="MentalBERT",
                             phq9_vmin=None, phq9_vmax=None, equal_aspect=True,
                             path="", filename="", save=False, show_fig=False):
    """All seed trajectories of ONE setting overlaid in a single shared axes.

    Every seed was projected with the same per-setting reducer, so one axes is a
    single coordinate system (same x/y scale for all seeds), overlaying shows
    directly whether the seeds drift together or diverge. Seeds are not
    distinguished (no legend); dots = windows coloured by mean PHQ-9 (green→red),
    start = square, end = star.

    Args:
        trajs:  list of (T, 2) reduced trajectories (one per seed).
        phq9s:  list of (T,) per-window mean PHQ-9 (one per seed).
        seeds:  unused (kept for call-site compatibility).
        row_title: setting label for the title (e.g. "SDA undirected").
        equal_aspect: unused (panels fill the axes; ticks show the scale).

    Returns:
        (fig, ax)
    """
    phq9_vmin, phq9_vmax = _phq9_limits(phq9s, phq9_vmin, phq9_vmax)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    sc = _draw_overlay(ax, trajs, phq9s, phq9_vmin, phq9_vmax,
                       equal_aspect=False, show_ticks=True, dot_size=14)
    ax.set_xlabel(f"{reduction} 1")
    ax.set_ylabel(f"{reduction} 2")
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label("Mean PHQ-9", fontsize=9)
    title = f"{embedding} entrainment trajectories ({reduction} 2D)"
    ax.set_title(f"{row_title} — {title}" if row_title else title, fontsize=9)

    plt.tight_layout()
    if save and path and filename:
        os.makedirs(path, exist_ok=True)
        out = os.path.join(path, f"{filename}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[plot] wrote {out}")
    if show_fig:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_entrainment_overlay_grid(cell_trajs, cell_phq9, row_titles, col_titles,
                                  reduction="PCA", embedding="MentalBERT",
                                  equal_aspect=True, path="", filename="",
                                  save=False, show_fig=False):
    """Small grid of per-setting overlays, one cell per setting, own colourbar.

    Each cell overlays all seeds of that setting (its own per-setting PCA), with
    its OWN PHQ-9 colour scale + slim colourbar (so cells are self-contained; the
    PCA axes differ cell to cell anyway). No seed legend; compact panels.

    Args:
        cell_trajs: `{(row, col): [ (T,2) per seed ]}`.
        cell_phq9:  `{(row, col): [ (T,)  per seed ]}`.
        row_titles / col_titles: setting axis labels (e.g. ["SDA","SDC"],
            ["undirected","directed"]).

    Returns:
        (fig, axes)
    """
    n_rows, n_cols = len(row_titles), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.9 * n_cols, 2.2 * n_rows),
                             squeeze=False, layout="constrained")

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            trajs = cell_trajs.get((r, c))
            phq9s = cell_phq9.get((r, c))
            if trajs:
                vmin, vmax = _phq9_limits(phq9s, None, None)  # per-cell scale
                # panels fill the axes (no equal aspect -> no y whitespace) + ticks
                sc = _draw_overlay(ax, trajs, phq9s, vmin, vmax,
                                   equal_aspect=False, show_ticks=True, dot_size=7)
                if sc is not None:
                    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                    cb.ax.tick_params(labelsize=6)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=9)

    if save and path and filename:
        os.makedirs(path, exist_ok=True)
        out = os.path.join(path, f"{filename}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[plot] wrote {out}")
    if show_fig:
        plt.show()
    plt.close(fig)
    return fig, axes


def plot_entrainment_shared(groups_per_col, col_titles, reduction="PCA",
                            embedding="MentalBERT", path="", filename="",
                            save=False, show_fig=False):
    """SDA+SDC variants in ONE shared mapping per column (e.g. per direction).

    Each column is a single shared reduced space (one PCA fit pooling all that
    column's groups), so SDA and SDC trajectories are directly comparable in it.
    Group identity (net × variant) is encoded by **marker shape** so the dot
    colour stays free for mean PHQ-9 (green→red). Each column has its OWN PHQ-9
    colourbar (own scale) and visible axis ticks so the per-column scale is
    readable; panels fill the axes (no forced equal aspect). No figure title.

    Args:
        groups_per_col: `{col_index: [group, ...]}` where each group is
            `{"label": str, "marker": str, "trajs": [ (T,2) ], "phq9s": [ (T,) ]}`.
        col_titles: column labels (e.g. ["undirected", "directed"]).

    Returns:
        (fig, axes)
    """
    n_cols = len(col_titles)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.4 * n_cols + 0.3, 2.55),
                             squeeze=False, layout="constrained")

    for ci in range(n_cols):
        ax = axes[0][ci]
        groups = groups_per_col.get(ci, [])
        # Per-column PHQ-9 scale (own colourbar), so each panel uses its full range.
        col_phq9 = [p for g in groups for p in g["phq9s"]]
        vmin, vmax = _phq9_limits(col_phq9, None, None)
        sc = None
        for g in groups:
            for tr, pq in zip(g["trajs"], g["phq9s"]):
                tr = np.asarray(tr)
                ax.plot(tr[:, 0], tr[:, 1], color="0.8", alpha=0.3,
                        linewidth=0.4, zorder=1)
                sc = ax.scatter(tr[:, 0], tr[:, 1], c=np.asarray(pq),
                                cmap="RdYlGn_r", vmin=vmin, vmax=vmax, s=7,
                                marker=g["marker"], alpha=0.9,
                                edgecolors="black", linewidths=0.15, zorder=2)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel(f"{reduction} 1", fontsize=7)
        ax.set_ylabel(f"{reduction} 2", fontsize=7)
        # Sub-caption underneath: (a) undirected, (b) directed, ...
        ax.text(0.5, -0.30, f"({chr(97 + ci)}) {col_titles[ci]}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
        if sc is not None:
            cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, fraction=0.046)
            cb.set_label("Mean PHQ-9", fontsize=7)
            cb.ax.tick_params(labelsize=6)

    # Marker legend (group identity), drawn in neutral grey since colour = PHQ-9.
    ref = next((col for col in groups_per_col.values() if col), [])
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=g["marker"], linestyle="none",
                      markerfacecolor="0.5", markeredgecolor="0.3",
                      markersize=6, label=g["label"]) for g in ref]
    if handles:
        fig.legend(handles=handles, loc="outside upper center",
                   ncol=len(handles), fontsize=6, frameon=False)
    if save and path and filename:
        os.makedirs(path, exist_ok=True)
        out = os.path.join(path, f"{filename}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[plot] wrote {out}")
    if show_fig:
        plt.show()
    plt.close(fig)
    return fig, axes


#============ Network Analysis Visualization =============#

def check_degree_distribution(unique_degrees, frequencies):
    """Plot the degree distribution on a log-log scale.
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
    """Plot the mean tweet frequency over time with variance as a shaded region.

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
#============= Critical slowing down Visualization =============#
def plot_agent_cd_heatmaps(network, window, cd_results, metric_name="PHQ-9", path="", filename="default.png", shift=1):
    """Plots heatmaps for Variance and Autocorrelation across all agents.
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
    """Plots local (neighbor) vs. random cosine similarity over time.

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
    """Tests whether PHQ-9 similarity predicts semantic similarity, split by neighbor status.
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
    """Three-panel figure testing whether semantic content drives depression echo chambers.

    Panel 1, PHQ-9 assortativity + cross-agent PHQ-9 variance (twin axis).
    Panel 2, Depression-axis alignment (Pearson r: projection vs. PHQ-9).
    Panel 3, Depression-axis entrainment (local vs. random similarity on depression axis).

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
    """Standalone plot: PHQ-9 assortativity (left axis) + degree-weighted mean PHQ-9
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

        # Degree-weighted PHQ-9: mean with min-max band (can't go negative)
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

    # Right axis: degree-weighted mean PHQ-9 with min-max band (always ≥ 0)
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
    """Scatter of each agent's mean PHQ-9 vs. their neighbors' mean PHQ-9.

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
    """Colour of a PHQ-9 score by severity band (green to dark red)."""
    if phq9 >= 20: return "#8B0000"   # severe, dark red
    if phq9 >= 15: return "#D73027"   # moderately severe, red
    if phq9 >= 10: return "#FC8D59"   # moderate, orange
    if phq9 >= 5:  return "#FEE090"   # mild, yellow
    return "#91CF60"                   # minimal, green


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
    `avg_mae`, `avg_bias`, `std_bias`, and `n_samples`.

    The bias panel uses `mean(pred − true)` so positive bars mean the model
    over-estimates the true score for that severity bracket, negative bars mean
    under-estimation. Error bars on both panels show the (weighted) std of
    per-PHQ-9 averages within the category, same convention as
    `plot_test_scores_by_phq9`.
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


# Fine-tuned regressor per-band bias, one line per generator (multi-model arm).
# Same bands / bias convention as plot_test_mae_and_bias_by_phq9; colours are the
# first slots of the categorical palette (all-pairs CVD-safe).
MULTIMODEL_GENERATORS = [
    ("Qwen3.5-27B", "#eb6834", "data/assessors/bert/qwen27_optimized/eval/on_qwen27_optimized"),
    ("Gemma-4-31B-it", "#2a78d6", "data/assessors/bert/gemma4_optimized/eval/on_gemma4_optimized"),
    ("Mistral-Small-3.2-24B", "#1baf7a", "data/assessors/bert/mistral_optimized/eval/on_mistral_optimized"),
]


# Each student generator's 300-block held-out set (human-optimized iter_10 prompt,
# shared personas), keyed by its MULTIMODEL_GENERATORS label: (short name, CSV).
MULTIMODEL_TEST_POSTS = {
    "Qwen3.5-27B": ("Qwen", "data/finetune/qwen/test_posts_qwen.csv"),
    "Gemma-4-31B-it": ("Gemma", "data/finetune/gemma4/test_posts_gemma4.csv"),
}
# The minimal-prompt counterpart of each generator's fine-tune arm (jobs/run_finetune_minimal.job):
# the same 3,000 / 300 personas written under the iter_0 prompt, the regressor fine-tuned on
# those posts and scored on that 300-block set. (300-block test posts, per-seed eval dir);
# skipped in the figure until the job has run.
MULTIMODEL_MINIMAL_ARMS = {
    "Qwen3.5-27B": ("data/finetune/qwen_minimal/test_posts_qwen_minimal.csv",
                    "data/assessors/bert/qwen27_minimal/eval/on_qwen27_minimal"),
    "Gemma-4-31B-it": ("data/finetune/gemma4_minimal/test_posts_gemma4_minimal.csv",
                       "data/assessors/bert/gemma4_minimal/eval/on_gemma4_minimal"),
}
# PHQ-9-axis sensitivity-analysis root per generator (sa_run.sh / sa_phq9_minimal_run.sh,
# jobs/sa_phq9_gemma4.job): <root>/<subdir>/<band>/rep_*/embeddings_sbert.npz, read by
# sa_analyze.phq9_adjacent_band_ladder. Subdirs: (legend suffix, subdir).
MULTIMODEL_SA_ROOTS = {
    "Qwen3.5-27B": "data/sensitivity",
    "Gemma-4-31B-it": "data/sensitivity/gemma4",
}
# The per-agent-seeded rerun of the same two SA axes (jobs 26835629/26835630, sa_run.sh
# with --seed S --per-agent-seed), same bands, reps and anchors; pass it as `sa_roots`.
MULTIMODEL_SA_ROOTS_SEEDED = {
    "Qwen3.5-27B": "data/sensitivity/seeded/qwen",
    "Gemma-4-31B-it": "data/sensitivity/seeded/gemma4",
}
MULTIMODEL_SA_PROMPTS = [("human-opt.", "phq9"), ("minimal", "phq9_minimal_prompt")]
# One visual code for the whole multi-model figure: the MARKER is the generative model
# (`MULTIMODEL_MODEL_MARKERS`) and the COLOUR is the generation prompt the posts were
# written under (`MULTIMODEL_PROMPT_COLOURS`). That leaves the line free, which panel (b)
# uses for the assessor (`MULTIMODEL_ASSESS_STYLES`: solid regressor, dotted LLM); the
# other panels draw solid. Every line is then unique on colour + marker alone.
# Legend wording for the prompt keys (the spelling the text uses).
MULTIMODEL_PROMPT_LABELS = {"human-opt.": "human-optimized", "minimal": "minimal",
                            "TextGrad best": "TextGrad (best)"}
MULTIMODEL_PROMPT_STYLES = {
    "human-opt.": ("-", "o"),
    "minimal": ("--", "^"),
    "TextGrad best": ("-.", "D"),
}
MULTIMODEL_MODEL_MARKERS = {"Qwen3.5-27B": "o", "Gemma-4-31B-it": "^", "Mistral-Small-3.2-24B": "s"}
# The model also carries a line style, so the panels stay readable where two models share a
# colour. Panel (b) is the exception: a row is one model there, and its line is free to
# mean the assessor instead (`MULTIMODEL_ASSESS_STYLES`, dotted = LLM).
MULTIMODEL_MODEL_STYLES = {"Qwen3.5-27B": "-", "Gemma-4-31B-it": "--", "Mistral-Small-3.2-24B": "-."}
# Purple, not grey, for TextGrad: grey reads as an annotation colour in this figure.
MULTIMODEL_PROMPT_COLOURS = {"human-opt.": "#eb6834", "minimal": "#2a78d6", "TextGrad best": "#8e44ad"}
# The paper's two assessors on the teacher base set (Table 1): label, colour, line
# style, marker, per-seed prediction CSV pattern, seeds. Each BERT seed scores its
# own 1,121-1,235-block split (seed 35: 1,125); every prompt seed scores the seed-35 split.
#
# Dropped from panel (b) on 2026-09-21, once the minimal-prompt arms landed: the teacher
# set is itself Qwen-generated under the historical prompts_post.json prompt on personas
# disjoint from the 300-block sets, so these lines were a third prompt variant in a
# different visual code, not a shared baseline. Panel (b) now reads like (a) and (c):
# colour = generator, line style = generation prompt. The rows are kept (this repo is not
# under git) for the SI figure that still shows them; pass them to
# plot_multimodel_linearity_bias via `teacher_assessors=MULTIMODEL_TEACHER_ASSESSORS_SI`.
MULTIMODEL_TEACHER_ASSESSORS_SI = [
    ("MentalBERT+MLP, teacher set", "#3b3b3b", ":", "s",
     "data/assessors/bert/teacher/models/Qwen3.5-27B_seed{seed}/test_raw_scores.csv", [34, 35, 36, 37, 38]),
    ("LLM (TextGrad), teacher set", "#9a9a9a", ":", "P",
     "data/test_post/optimized_phq9/Qwen3.5-27B_seed{seed}/eval_on_test_blocks_seed35/test_raw_scores.csv",
     [23, 24, 25, 32, 33]),
]
MULTIMODEL_TEACHER_ASSESSORS = []  # main figure: no teacher-set lines (see above)
# The LLM assessor on a generator's own 300-block held-out set: the prompted counterpart
# of the fine-tuned regressor of panel (b), drawn in that generator's own row, on the same
# blocks. Fields: generator label (sets the colour), the generation
# prompt those posts were written under, the ASSESSMENT prompt (a MULTIMODEL_ASSESS_STYLES
# key: it sets the line style, while the marker keeps coming from the generation prompt of
# the posts as everywhere else in the figure, so the LLM lines differ only in their line),
# the per-seed prediction CSV pattern, seeds, the block CSV whose
# row order those CSVs follow (they carry no agent_id, so ids -- and the persona
# exclusion -- are attached by position), and which run to draw: "best" = the single seed
# with the lowest validation MAE (the convention panel (c) uses for the TextGrad
# generation prompt, so no error band), "mean" = mean +/- SD over the seeds given.
#   - TextGrad-optimized: re-optimized on `finetune/qwen/train_posts_qwen.csv` from the minimal
#     start prompt (`optimized_phq9_human/`, scripts/assessment/run_textgrad_phq9_human.sh).
#   - minimal: that same start prompt, unoptimized, on the same blocks
#     (`optimized_phq9/*/minimal_human300/`, run_llm_assessor_on_heldout.sh). One run
#     only: the prompt does not depend on the TextGrad seed its directory happens to sit in.
MULTIMODEL_LLM_ASSESSORS = [
    ("Qwen3.5-27B", "human-opt.", "minimal",
     "data/test_post/optimized_phq9/Qwen3.5-27B_seed{seed}/minimal_human300/test_raw_scores.csv",
     [23], "data/finetune/qwen/test_posts_qwen.csv", "mean"),
]
# The same assessor on Gemma's 300-block set (scripts/assessment/run_llm_assessor_on_heldout.sh
# gemma4, 2026-09-03; five seeds where Qwen's row has one, because the minimal assessment
# prompt does not depend on the TextGrad seed its directory sits in, so these are five
# sampling reps of one prompt). It is NOT in the list above on purpose: it reads Gemma's
# posts ~6 PHQ-9 points low, which on the combined figure's shared (b) scale stretches the
# axis to -10 and flattens every regressor line. The per-model figures give it its own row,
# where that magnitude is the point.
MODEL_FIG_LLM_ASSESSORS = MULTIMODEL_LLM_ASSESSORS + [
    ("Gemma-4-31B-it", "human-opt.", "minimal",
     "data/test_post/optimized_phq9/Qwen3.5-27B_seed{seed}/minimal_gemma4300/test_raw_scores.csv",
     [23, 24, 25, 32, 33], "data/finetune/gemma4/test_posts_gemma4.csv", "mean"),
]
# Dropped from the main figure on 2026-09-21: the TextGrad-optimized assessment prompt
# ("TextGrad best", `optimized_phq9_human/Qwen3.5-27B_seed{seed}/test_raw_scores.csv`,
# seeds [23, 24, 25, 32, 33], select "best"). The LLM row now shares the generator's
# sub-panel, where a second assessment prompt would be a third line in a panel whose
# line styles already mean the generation prompt. Append it to the list above to get it
# back.
# Line style per assessment prompt: dotted for the minimal prompt, which is what sets the
# LLM line apart from the fine-tuned regressor sharing its sub-panel (the marker keeps
# coming from the generation prompt the posts were written under).
MULTIMODEL_ASSESS_STYLES = {"bert": "-", "TextGrad best": "-.", "minimal": ":"}
# Teacher gradings (0-10) on the paper's protocol: the 100 test personas, 3 fresh posts
# each per generator (checks/check_grading_consistency.job); one row per block
# (columns persona, phq9, score): generator (MULTIMODEL_GENERATORS label, sets the
# colour), prompt (MULTIMODEL_PROMPT_STYLES key, sets line style + marker), CSV. The
# 10-post grades of the 300-block sets (jobs/grade_posts_multimodel.job) are the
# `grades_<tag>300.csv` files next to them. The minimal and TextGrad rows are the
# paper's original test runs (same 100 personas, same protocol): iter_0 (mean 5.68)
# and the best of the five TextGrad generation prompts, seed 29 (val 7.42, test
# 6.74). Those two have no persona column, their agent_id is the row of
# MULTIMODEL_GRADING_PERSONAS.
MULTIMODEL_GRADINGS = [
    ("Qwen3.5-27B", "human-opt.", "data/test_post/teacher_grades/grades_100x3_qwen.csv"),
    ("Gemma-4-31B-it", "human-opt.", "data/test_post/teacher_grades/grades_100x3_gemma4.csv"),
    ("Qwen3.5-27B", "TextGrad best", "data/test_post/optimized_tweets/Qwen3.5-27B_seed29/test_raw_scores.csv"),
    ("Qwen3.5-27B", "minimal", "data/prompt_optimization_h/qwen27_baseline/iter_0/test_raw_scores.csv"),
]
MULTIMODEL_GRADING_PERSONAS = "data/personas_eval_1000_phq9.csv"
# Panel (c) of the per-model figures, re-based on the paired 300-block held-out sets
# (jobs/grade_posts_panelc.job). `MULTIMODEL_GRADINGS` above mixes two grading paths
# (the human-opt rows were regraded through `grade-posts` in 2026-09, the minimal and
# TextGrad rows are the paper's 2026-05/06 stored optimizer scores, which read +0.25
# high on identical posts) on a 100-persona x 3-post protocol that shares only 28
# personas with the 300-block sets panel (b) scores, and whose block length matters
# (Qwen: 5.78 at 10 posts vs 6.86 at first-3). These rows put every arm on one
# persona set, one block length and one grading path.
# Graded at 3 posts per block, not 10: the rubric scores diversity across the block, so a
# 10-post block punishes repetition far harder than the paper's 3-post protocol (Qwen 5.78
# at 10 posts vs 6.86 on the first 3 of the same blocks, r 0.37; Gemma +0.13). Grading at
# 10 would put the arms on a scale the paper's numbers do not live on and would inflate the
# Qwen-Gemma gap for a block-length reason. The truncation is exact rather than a sample:
# the per-round sampling seed is `seed + round` and the neighbour draw is
# SeedSequence[neighbor_seed, agent_id, round], both keyed on the round index and not on
# the total round count, so the first 3 posts of these 10-round runs are bit-identical to a
# 3-round run. `grades_{qwen,gemma4}300.csv` (10 posts) stay as the block-length control.
# (generator label, prompt key) -> grades CSV (persona, phq9, score).
MULTIMODEL_GRADINGS_300 = {
    ("Qwen3.5-27B", "human-opt."): "data/test_post/teacher_grades/grades_qwen300_first3.csv",
    ("Qwen3.5-27B", "minimal"): "data/test_post/teacher_grades/grades_qwen_minimal300_first3.csv",
    ("Qwen3.5-27B", "TextGrad best"): "data/test_post/teacher_grades/grades_qwen_textgrad300_first3.csv",
    ("Gemma-4-31B-it", "human-opt."): "data/test_post/teacher_grades/grades_gemma4300_first3.csv",
    ("Gemma-4-31B-it", "minimal"): "data/test_post/teacher_grades/grades_gemma4_minimal300_first3.csv",
    ("Gemma-4-31B-it", "TextGrad best"): "data/test_post/teacher_grades/grades_gemma4_textgrad300_first3.csv",
}
# Posts the human reviewer read while editing the generation prompt; 19 of the 300
# test personas are among them (optional exclusion in plot_multimodel_linearity_bias).
MULTIMODEL_PROMPT_ITER_POSTS = "data/prompt_optimization_h/qwen27_baseline/iter_*/posts.csv"
_MM_BAND_SHORT = ["Minimal", "Mild", "Mod.", "Mod. Sev", "Sev"]


def _band_bias_per_seed(paths: list, keep_ids=None, order_ids=None) -> np.ndarray:
    """Signed bias (pred - true) per severity band for each prediction CSV that exists; shape (n_seeds, 5).

    Args:
        paths (list): prediction CSVs (true_phq9, pred_phq9, agent_id).
        keep_ids (set | None): agent ids to keep; None keeps every row.
        order_ids (array | None): agent id per row, for CSVs written without an
            agent_id column (the TextGrad LLM-assessor runs: one row per block, in
            the order of the test-post file).
    """
    rows = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"[multimodel-linearity] missing {p}")
            continue
        d = pd.read_csv(p)
        if "agent_id" not in d.columns and order_ids is not None:
            d["agent_id"] = np.asarray(order_ids)[:len(d)]
        if keep_ids is not None:
            d = d[d["agent_id"].isin(keep_ids)]
        err = d["pred_phq9"] - d["true_phq9"]
        rows.append([err[d["true_phq9"].between(lo, hi)].mean() for lo, hi, _ in _MM_BANDS])
    return np.array(rows, float)


def _textgrad_best_seed(pattern: str, seeds: list):
    """Seed of the TextGrad run with the lowest validation MAE, or None if no run was found.

    Selection is on validation only (`state.json`'s `best_metric`, else the best val
    step of `training_trajectory.csv`), never on the test blocks the figure shows.
    The val split is drawn per seed, so this compares runs on different 40-block
    draws, exactly as the "TextGrad (best)" generation prompt of panel (c) was picked.

    Args:
        pattern (str): prediction-CSV pattern with a `{seed}` field; its directory is the run.
        seeds (list): seeds to choose from.
    """
    scored = []
    for s in seeds:
        run = os.path.dirname(pattern.format(seed=s))
        state, traj = os.path.join(run, "state.json"), os.path.join(run, "training_trajectory.csv")
        if os.path.isfile(state) and "best_metric" in json.load(open(state)):
            scored.append((json.load(open(state))["best_metric"], s))
        elif os.path.isfile(traj):
            val = pd.read_csv(traj).query("split == 'val'")["mean_score"]
            if len(val):
                scored.append((val.min(), s))
    if not scored:
        return None
    best_val, best_seed = min(scored)
    print(f"[multimodel-linearity] TextGrad best seed {best_seed} (val MAE {best_val:.2f} of "
          + ", ".join(f"{s}:{v:.2f}" for v, s in sorted(scored, key=lambda t: t[1])) + ")")
    return best_seed


def _errorbar_annotated(ax, y, err, colour, style, marker, label, fmt="{:.3f}", below=False, annotate=True):
    """Line with capped error bars and, unless `annotate=False`, the value printed at each point (above, or below the bar if `below`)."""
    x = np.arange(len(y))
    ax.errorbar(x, y, yerr=err, fmt=marker, linestyle=style, color=colour, capsize=2.5,
                linewidth=1.6, markersize=5.5, markeredgecolor="black", markeredgewidth=0.5,
                zorder=3, label=label)
    if not annotate:
        return
    sign = -1 if below else 1
    for xi, yi, ei in zip(x, y, err):
        ax.annotate(fmt.format(yi), (xi, yi + sign * ei), xytext=(0, 3 * sign), textcoords="offset points",
                    ha="center", va="top" if below else "bottom", fontsize=6.5, color=colour)


def plot_multimodel_linearity_bias(out_path: str, n_bootstrap: int = 1000, seed: int = 0,
                                   exclude_prompt_iter_personas: bool = False,
                                   teacher_assessors: list | None = None,
                                   emb_name: str = "embeddings_sbert.npz",
                                   sa_roots: dict | None = None):
    """Three panels: (a) PHQ-9-axis SA adjacent-band cosine per generator and prompt, (b) per-band bias of every assessor, (c) teacher gradings.

    Same compact style as the SA figure `sa_analyze.plot_agent_phq9_combined`, whose
    left panel (a) reproduces for every generator: the same-persona S-BERT cosine
    between consecutive PHQ-9 bands from the PHQ-9 conditioning runs
    (`sa_analyze.phq9_adjacent_band_ladder` on `MULTIMODEL_SA_ROOTS[<generator>]`),
    for the human-optimized prompt (3 reps, error bar = SD across reps) and the
    minimal prompt (1 rep, no bar), `MULTIMODEL_SA_PROMPTS`. Personas, their per-band
    scores and the neighbour posts are pinned across bands, reps, prompts and
    generators, so only the conditioned band (and the generator) moves. (b) and (c)
    use the paired 300-block held-out sets / 100 test personas generated with the
    human-optimized prompt. `exclude_prompt_iter_personas` drops the 19 blocks whose
    persona the reviewer saw while editing that prompt (`MULTIMODEL_PROMPT_ITER_POSTS`)
    from the FT lines of (b) and from (c); off by default, so all 300 blocks are used.
    (b) is a stack of sub-panels, one per generator, sharing the band axis, one y-label
    and one y-scale (colour still means the generator). Each row holds the regressor
    fine-tuned on that generator's own posts, for the human-optimized arm and the
    minimal-prompt arm (`MULTIMODEL_MINIMAL_ARMS`) on the same personas, mean +/- SD
    over seeds, and, where such a run exists, the prompted LLM assessor on the same
    blocks (`MULTIMODEL_LLM_ASSESSORS`, the minimal assessment prompt), dotted; its
    last field picks the run drawn, "best" = the validation-best seed alone (no band,
    the same "best of five prompts" convention as the TextGrad line of panel (c)),
    "mean" = mean +/- SD over the seeds listed. A row with both assessors carries a
    small "Assessor" key naming them.
    Teacher-set assessors are off by default (see
    `MULTIMODEL_TEACHER_ASSESSORS`); `teacher_assessors` overlays them for the SI.
    (c) is the teacher's 0-10 grading
    per band (`MULTIMODEL_GRADINGS`: the paper's 100 test personas x 3 posts, so a
    different persona set from (a)/(b)) for each generator under the human-optimized
    prompt and for Qwen under the best TextGrad and the minimal prompt, mean +/- SEM
    over blocks. Colour = generator, line style + marker = prompt
    (`MULTIMODEL_PROMPT_STYLES`) in every panel, so the figure carries one shared
    legend above the panels (titled groups: model, generation prompt, the assessment
    prompts of (b)'s LLM lines, and the teacher-set assessors when those are on) instead
    of three per-panel ones.
    Missing inputs are skipped. Also writes the (a) numbers next to the PNG as
    `<stem>.csv`.

    Args:
        out_path (str): PNG to write.
        n_bootstrap (int): unused, kept for the CLI signature.
        seed (int): unused, kept for the CLI signature.
        exclude_prompt_iter_personas (bool): drop the reviewer-seen personas (see above).
        emb_name (str): SA embedding file panel (a) reads. The default is the plain
            SBERT run; "embeddings_sbert_noemoji.npz" (written by
            `sa_embed --sbert --strip-emoji`) gives the emoji-stripped version, which
            matters because emoji are concentrated in the low-severity bands and pull
            unrelated posts together there.
        teacher_assessors (list | None): teacher-set assessor rows to overlay on (b).
            None uses `MULTIMODEL_TEACHER_ASSESSORS`, which is empty for the main
            figure; pass `MULTIMODEL_TEACHER_ASSESSORS_SI` for the SI version.
        sa_roots (dict | None): SA roots panel (a) reads, per generator label. None
            uses `MULTIMODEL_SA_ROOTS` (the unseeded runs); pass
            `MULTIMODEL_SA_ROOTS_SEEDED` for the per-agent-seeded rerun.

    Returns:
        str | None: `out_path`, or None if no generator had data.
    """
    from .sensitivity.sa_analyze import phq9_adjacent_band_ladder
    from matplotlib.lines import Line2D
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    steps = [f"{a} → {b}" for a, b in zip(_MM_BAND_SHORT[:-1], _MM_BAND_SHORT[1:])]
    x = np.arange(len(_MM_BANDS))
    sa_roots = MULTIMODEL_SA_ROOTS if sa_roots is None else sa_roots
    # Which generators have data: each gets its own row of panel (b), so this is settled
    # before the figure is built.
    active = []
    for label, colour, eval_dir in MULTIMODEL_GENERATORS:
        short, posts_csv = MULTIMODEL_TEST_POSTS.get(label, (None, None))
        seeds = [f for f in sorted(glob.glob(os.path.join(eval_dir, "seed*.csv"))) if "summary" not in f]
        if not posts_csv or not os.path.isfile(posts_csv) or not seeds:
            print(f"[multimodel-linearity] skip {label}: no test posts or eval CSVs")
            continue
        active.append((label, colour, short, posts_csv, seeds))
    if not active:
        return None

    # (b) is a stack of sub-panels, one per generator, each holding that generator's
    # fine-tuned regressor and its prompted LLM assessor (the two sat far enough apart on
    # a shared y-axis to hide each other only while the optimized assessment prompt was
    # drawn too). They share the band axis (only the bottom row is labelled), one y-label
    # centred on the stack and one y-scale.
    llm_rows = [r for r in MULTIMODEL_LLM_ASSESSORS
                if r[0] in {label for label, *_ in active} and os.path.isfile(r[5])
                and any(os.path.isfile(r[3].format(seed=sd)) for sd in r[4])]
    bias_rows = [label for label, *_ in active]
    fig = plt.figure(figsize=(7.5, 1.15 * len(bias_rows) + 0.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2], wspace=0.45)
    ax_lin, ax_grade = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])
    gp = ax_grade.get_position()  # the (a)-(b) gap holds (b)'s shared y-label, the (b)-(c) gap
    ax_grade.set_position([gp.x0 - 0.025, gp.y0, gp.width, gp.height])  # only (c)'s own labels
    inner = gs[0, 1].subgridspec(len(bias_rows), 1, hspace=0.10)
    bias_axes = {}
    for i, key in enumerate(bias_rows):
        bias_axes[key] = fig.add_subplot(inner[i, 0], sharex=bias_axes[bias_rows[0]] if i else None)
    rows = []
    used_gens, used_prompts, teacher_handles = [], set(), []  # what the shared legend has to cover
    seen = set()  # personas the reviewer read during prompt editing
    if exclude_prompt_iter_personas:
        for f in glob.glob(MULTIMODEL_PROMPT_ITER_POSTS):
            seen |= set(pd.read_csv(f)["persona"].astype(str))

    for label, _, short, posts_csv, seeds in active:
        marker = MULTIMODEL_MODEL_MARKERS.get(label, "o")  # the model is the marker everywhere
        mstyle = MULTIMODEL_MODEL_STYLES.get(label, "-")   # and the line outside panel (b)
        ax_bias = bias_axes[label]

        # (a) the SA PHQ-9-axis ladder, one line per prompt (skipped if that SA was not run/embedded).
        for suffix, subdir in MULTIMODEL_SA_PROMPTS:
            ladder = phq9_adjacent_band_ladder(sa_roots.get(label, ""),
                                               emb_name=emb_name, subdir=subdir)
            if ladder is None or ladder.empty:
                continue
            _errorbar_annotated(ax_lin, ladder["cos"].to_numpy(), ladder["std"].to_numpy(),
                                MULTIMODEL_PROMPT_COLOURS[suffix], mstyle, marker,
                                f"{short} ({suffix} prompt)", annotate=False)
            used_prompts.add(suffix)
            rows += [{"generator": label, "prompt": subdir, "step": r.step, "cosine": r.cos, "std": r.std,
                      "n_reps": r.n_reps, "n_anchors": r.n_anchors} for r in ladder.itertuples(index=False)]

        # (b) the regressor fine-tuned on this generator's posts, per prompt arm.
        arms = [("human-opt.", posts_csv, seeds)]
        if label in MULTIMODEL_MINIMAL_ARMS:
            m_csv, m_dir = MULTIMODEL_MINIMAL_ARMS[label]
            m_seeds = [f for f in sorted(glob.glob(os.path.join(m_dir, "seed*.csv"))) if "summary" not in f]
            if os.path.isfile(m_csv) and m_seeds:
                arms.append(("minimal", m_csv, m_seeds))
        for prompt_key, arm_csv, arm_seeds in arms:
            # In (b) the line is the assessor, so every regressor arm is solid and the row's
            # "Assessor" key holds for all of them; the prompt is in the colour.
            style, colour = MULTIMODEL_ASSESS_STYLES["bert"], MULTIMODEL_PROMPT_COLOURS[prompt_key]
            used_prompts.add(prompt_key)
            persona_of = pd.read_csv(arm_csv).groupby("agent_id")["persona"].first().astype(str)
            keep_ids = set(persona_of.index[~persona_of.isin(seen)])
            print(f"[multimodel-linearity] {short} ({prompt_key}): {len(keep_ids)} of {len(persona_of)} blocks kept")
            bias = _band_bias_per_seed(arm_seeds, keep_ids)
            ax_bias.fill_between(x, bias.mean(0) - bias.std(0), bias.mean(0) + bias.std(0),
                                 color=colour, alpha=0.2, linewidth=0)
            ax_bias.plot(x, bias.mean(0), linestyle=style, marker=marker, color=colour, linewidth=1.6,
                         markersize=5.5, markeredgecolor="black", markeredgewidth=0.5, zorder=3,
                         label=f"FT MentalBERT+MLP, {short} posts"
                               + (f" ({prompt_key} prompt)" if prompt_key != "human-opt." else ""))
        used_gens.append((short, marker, mstyle))  # "Qwen" / "Gemma", not the full model id

    # (b) the LLM assessor on the same blocks, in the same row as that generator's regressor.
    llm_axes = set()
    for gen, prompt_key, assess_key, pattern, seeds, blocks_csv, select in llm_rows:
        style, marker = MULTIMODEL_ASSESS_STYLES[assess_key], MULTIMODEL_MODEL_MARKERS.get(gen, "o")
        name = MULTIMODEL_PROMPT_LABELS[assess_key]
        best = _textgrad_best_seed(pattern, seeds) if select == "best" else None
        if select == "best" and best is None:
            continue
        use = [best] if select == "best" else seeds
        blocks = pd.read_csv(blocks_csv).groupby("agent_id", sort=False)["persona"].first().astype(str)
        keep_ids = set(blocks.index[~blocks.isin(seen)])
        bias = _band_bias_per_seed([pattern.format(seed=s) for s in use], keep_ids,
                                   order_ids=blocks.index.to_numpy())
        if not len(bias):
            continue
        short, colour = MULTIMODEL_TEST_POSTS.get(gen, (gen,))[0], MULTIMODEL_PROMPT_COLOURS[prompt_key]
        ax_bias = bias_axes[gen]
        llm_axes.add(gen)
        print(f"[multimodel-linearity] {short} LLM assessor, {name} ({prompt_key} posts): "
              f"{len(keep_ids)} of {len(blocks)} blocks kept, seeds {use}")
        if len(bias) > 1:  # a single run has no across-seed spread to show
            ax_bias.fill_between(x, bias.mean(0) - bias.std(0), bias.mean(0) + bias.std(0),
                                 color=colour, alpha=0.15, linewidth=0)
        ax_bias.plot(x, bias.mean(0), linestyle=style, marker=marker, color=colour, linewidth=1.4,
                     markersize=5, markeredgecolor="black", markeredgewidth=0.5, zorder=2,
                     label=f"LLM {name}, {short} posts")

    # The rows that hold both assessors say which line is which; a row with the regressor
    # alone needs no key, since colour already means the generator.
    for gen in llm_axes:
        handles = [Line2D([], [], color="0.35", linestyle=MULTIMODEL_ASSESS_STYLES["bert"],
                          linewidth=1.6, label="MentalBERT+MLP"),
                   Line2D([], [], color="0.35", linestyle=MULTIMODEL_ASSESS_STYLES["minimal"],
                          linewidth=1.6, label="LLM")]
        leg = bias_axes[gen].legend(handles=handles, title="Assessor", loc="lower left",
                                    fontsize=6, framealpha=0.9, handlelength=2.4,
                                    handletextpad=0.4, borderpad=0.3, labelspacing=0.25)
        leg.get_title().set_fontsize(6)
        leg.get_title().set_fontweight("bold")

    ax_bias = bias_axes[active[0][0]]  # teacher-set lines: not an LLM row, so the first regressor row
    for label, colour, style, marker, pattern, seeds in (
            MULTIMODEL_TEACHER_ASSESSORS if teacher_assessors is None else teacher_assessors):
        bias = _band_bias_per_seed([pattern.format(seed=s) for s in seeds])
        if not len(bias):
            continue
        ax_bias.fill_between(x, bias.mean(0) - bias.std(0), bias.mean(0) + bias.std(0),
                             color=colour, alpha=0.15, linewidth=0)
        ax_bias.plot(x, bias.mean(0), linestyle=style, marker=marker, color=colour, linewidth=1.4,
                     markersize=5, markeredgecolor="black", markeredgewidth=0.5, zorder=2, label=label)
        teacher_handles.append(Line2D([], [], color=colour, linestyle=style, marker=marker, linewidth=1.4,
                                      markersize=5, markeredgecolor="black", markeredgewidth=0.5, label=label))

    for gen, prompt_key, path in MULTIMODEL_GRADINGS:
        if not os.path.isfile(path):
            print(f"[multimodel-linearity] no gradings: {path}")
            continue
        short = MULTIMODEL_TEST_POSTS.get(gen, (gen,))[0]
        label, colour = f"{short} ({prompt_key} prompt)", MULTIMODEL_PROMPT_COLOURS[prompt_key]
        style = MULTIMODEL_MODEL_STYLES.get(gen, "-")
        marker = MULTIMODEL_MODEL_MARKERS.get(gen, "o")
        used_prompts.add(prompt_key)
        d = pd.read_csv(path)
        if "persona" not in d.columns:  # iter_0 raw scores: agent_id indexes the eval-1000 persona file
            d["persona"] = pd.read_csv(MULTIMODEL_GRADING_PERSONAS)["persona"].reindex(d["agent_id"]).to_numpy()
        d = d[~d["persona"].astype(str).isin(seen)].dropna(subset=["score"])
        by_band = [d["score"][d["phq9"].between(lo, hi)] for lo, hi, _ in _MM_BANDS]
        mean = np.array([g.mean() for g in by_band])
        sem = np.array([g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0 for g in by_band])
        _errorbar_annotated(ax_grade, mean, sem, colour, style, marker, label, annotate=False)  # four lines: no value labels

    ax_lin.set_xticks(np.arange(len(steps)))
    ax_lin.set_xticklabels(steps, rotation=30, ha="right", fontsize=8)
    ax_lin.set_ylabel("Cosine similarity", fontsize=9.5)
    ax_grade.set_ylabel("Teacher score (0–10)", fontsize=9.5)
    for ax in (*bias_axes.values(), ax_grade):
        ax.set_xticks(x)
        ax.set_xticklabels(_MM_BAND_SHORT, rotation=30, ha="right", fontsize=8)
    for i, key in enumerate(bias_rows):  # each row: zero line, and the band axis on the last one
        ax = bias_axes[key]
        ax.axhline(0, color="black", linewidth=0.8)
        if i < len(bias_rows) - 1:  # shared band axis: only the bottom row is labelled
            ax.tick_params(labelbottom=False)
    for ax in (ax_lin, ax_grade, *bias_axes.values()):
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.12, y=0.15)
    lims = [ax.get_ylim() for ax in bias_axes.values()]  # every row of (b) on one scale
    lo, hi = min(lo for lo, _ in lims), max(hi for _, hi in lims)
    lo -= (0.15 if llm_axes else 0) * (hi - lo)  # room under the curves for the row's own legend
    for ax in bias_axes.values():
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5]))  # the wider
        # range would otherwise be left with ticks at 0 and -5 only

    # One y-label for the whole (b) stack, centred on it (the rows are shorter than the
    # other two panels, so this cannot ride on one row's ylabel).
    bias_pos = [ax.get_position() for ax in bias_axes.values()]
    fig.canvas.draw()  # the label has to clear whichever row has the widest tick labels
    tick_left = min(lb.get_window_extent(fig.canvas.get_renderer()).x0 for ax in bias_axes.values()
                    for lb in ax.get_yticklabels() if lb.get_text())
    fig.text(fig.transFigure.inverted().transform((tick_left - 6, 0))[0],
             (max(b.y1 for b in bias_pos) + min(b.y0 for b in bias_pos)) / 2,
             "Bias = mean(pred − true)", rotation=90, ha="right", va="center", fontsize=9.5)

    # One legend for the whole figure, split into the two things a line encodes: marker =
    # generative model, colour = generation prompt. Each group is its own titled legend,
    # laid out left to right above the panels.
    groups = [("Generative model",
               [Line2D([], [], color="0.35", linestyle=style, marker=marker, linewidth=1.6, markersize=6,
                       markeredgecolor="black", markeredgewidth=0.5, label=short)
                for short, marker, style in used_gens]),
              ("Generation prompt",
               [Line2D([], [], color=MULTIMODEL_PROMPT_COLOURS[key], linewidth=2.6,
                       label=MULTIMODEL_PROMPT_LABELS[key])
                for key in MULTIMODEL_PROMPT_STYLES if key in used_prompts])]
    if teacher_handles:
        groups.append(("Teacher-set assessor", teacher_handles))
    legs = [fig.legend(handles=handles, title=title, ncol=len(handles), loc="lower left",
                       bbox_to_anchor=(0, 1.01), borderaxespad=0, fontsize=7.5, framealpha=0.9,
                       handlelength=2.0, handletextpad=0.5, borderpad=0.4, columnspacing=1.2)
            for title, handles in groups if handles]
    for leg in legs:
        leg.get_title().set_fontsize(7.5)
        leg.get_title().set_fontweight("bold")
    fig.canvas.draw()  # measure the legends, then centre them as a block just above the panels
    inv = fig.transFigure.inverted()

    # Panel captions, all on one baseline just under the deepest tick label of the three
    # columns (measured, not a fraction of panel height: the (b) rows are half-height).
    columns = [([ax_lin], "(a) Adjacent-band cosine"),
               (list(bias_axes.values()), "(b) Assessor bias per band"),
               ([ax_grade], "(c) Post gradings per band")]
    tight = {id(a): a.get_tightbbox(fig.canvas.get_renderer()).transformed(inv)
             for axes, _ in columns for a in axes}
    cap_y = min(tight[id(a)].y0 for axes, _ in columns for a in axes) - 0.012
    for axes, panel in columns:
        boxes_c = [tight[id(a)] for a in axes]
        fig.text((min(b.x0 for b in boxes_c) + max(b.x1 for b in boxes_c)) / 2, cap_y, panel,
                 ha="center", va="top", fontsize=10.5)

    boxes = [leg.get_window_extent().transformed(inv) for leg in legs]
    gap = 0.025
    y = max(a.get_position().y1 for a in (ax_lin, ax_grade, *bias_axes.values())) + 0.04
    total = sum(b.width for b in boxes) + gap * (len(legs) - 1)
    if total <= 1:  # they fit side by side on one row
        left = (1 - total) / 2
        for leg, b in zip(legs, boxes):
            leg.set_bbox_to_anchor((left, y), transform=fig.transFigure)
            left += b.width + gap
    else:  # too wide (the SI version with the teacher-set rows): one centred row per group
        for leg, b in zip(reversed(legs), reversed(boxes)):  # first group on top
            leg.set_bbox_to_anchor(((1 - b.width) / 2, y), transform=fig.transFigure)
            y += b.height + 0.02
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.splitext(out_path)[0] + ".csv", index=False)
    print(f"Multi-model linearity/bias/grading plot → {out_path}")
    return out_path


# Panel (b) of the per-model figure, top row first: (row label, which run it is).
# The row label is what tells the reader the assessor, so inside one figure the LINE
# is free: every line is solid and only the colour (generation prompt) varies.
# Kept short: the box has to sit inside a narrow axes without running into panel (c),
# so the qualifiers ("minimal PHQ-9 prompt", "fine-tuned on this generator's own
# posts") belong in the caption, not here.
MODEL_FIG_ASSESSORS = ["LLM assessor", "MentalBERT+MLP"]


def plot_model_linearity_bias(label: str, out_path: str, exclude_prompt_iter_personas: bool = False,
                              emb_name: str = "embeddings_sbert.npz", sa_roots: dict | None = None,
                              gradings: dict | None = None):
    """One generator's three panels, with (b) split by ASSESSOR instead of by generator.

    The per-model counterpart of `plot_multimodel_linearity_bias`: same three columns
    and the same compact style, but every line belongs to one generative model, so the
    split stack of (b) is free to mean the assessor. Top row is the prompted LLM
    (`MULTIMODEL_LLM_ASSESSORS`), bottom row the fine-tuned MentalBERT+MLP, each named
    by a label inside its own axes rather than by a line style. The two rows share one
    band axis and one y-label but NOT a y-scale: the two assessors differ by an order of
    magnitude on Gemma, so each row autoscales to its own data. Colour is the generation prompt
    everywhere (`MULTIMODEL_PROMPT_COLOURS`) and the marker is the model, so the figure
    needs a single legend.

    The three panels do NOT share a persona sample, and the figure does not pretend
    they do: (a) is a within-persona conditioning sweep over 60 SA anchors whose PHQ-9
    is overridden per band (it has to be, since the same persona must appear at every
    band), while (b) and (c) are both the 300 held-out personas at 10 posts per block.
    All three are samples of the same eval-1000 pool; (a) shares 14 personas with the
    300 set. `gradings` defaults to `MULTIMODEL_GRADINGS_300` for exactly that reason:
    it puts (c) on the posts (b) scores, unlike the paper's 100-persona x 3-post rows.

    Args:
        label (str): generator, a `MULTIMODEL_GENERATORS` label.
        out_path (str): PNG to write; the (a) numbers go next to it as `<stem>.csv`.
        exclude_prompt_iter_personas (bool): drop the blocks whose persona the reviewer
            saw while editing the generation prompt (`MULTIMODEL_PROMPT_ITER_POSTS`).
        emb_name (str): SA embedding file panel (a) reads.
        sa_roots (dict | None): SA roots per generator. None uses the per-agent-seeded
            rerun (`MULTIMODEL_SA_ROOTS_SEEDED`), which is the default here because the
            unseeded runs share one vLLM seed per round, so their round-0 posts repeat
            across reps; pass `MULTIMODEL_SA_ROOTS` for the unseeded version.
        gradings (dict | None): (generator, prompt) -> grades CSV for (c); None uses
            `MULTIMODEL_GRADINGS_300`.

    Returns:
        str | None: `out_path`, or None if this generator has no test posts / evals.
    """
    from .sensitivity.sa_analyze import phq9_adjacent_band_ladder
    from matplotlib.lines import Line2D
    # Seeded by default: the unseeded SA reps share a vLLM seed, so their round-0 posts
    # duplicate across reps and the error bars are understated.
    sa_roots = MULTIMODEL_SA_ROOTS_SEEDED if sa_roots is None else sa_roots
    gradings = MULTIMODEL_GRADINGS_300 if gradings is None else gradings
    short, posts_csv = MULTIMODEL_TEST_POSTS.get(label, (None, None))
    eval_dir = dict((g, d) for g, _, d in MULTIMODEL_GENERATORS).get(label, "")
    seeds = [f for f in sorted(glob.glob(os.path.join(eval_dir, "seed*.csv"))) if "summary" not in f]
    if not posts_csv or not os.path.isfile(posts_csv) or not seeds:
        print(f"[model-linearity] skip {label}: no test posts or eval CSVs")
        return None
    marker = "o"  # one model per figure, so the marker encodes nothing; keep both figures alike
    x = np.arange(len(_MM_BANDS))
    steps = [f"{a} → {b}" for a, b in zip(_MM_BAND_SHORT[:-1], _MM_BAND_SHORT[1:])]

    seen = set()  # personas the reviewer read during prompt editing
    if exclude_prompt_iter_personas:
        for f in glob.glob(MULTIMODEL_PROMPT_ITER_POSTS):
            seen |= set(pd.read_csv(f)["persona"].astype(str))

    fig = plt.figure(figsize=(7.5, 2.40))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2], wspace=0.45)
    ax_lin, ax_grade = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])
    inner = gs[0, 1].subgridspec(len(MODEL_FIG_ASSESSORS), 1, hspace=0.10)
    bias_axes = {}
    for i, name in enumerate(MODEL_FIG_ASSESSORS):
        bias_axes[name] = fig.add_subplot(inner[i, 0],
                                          sharex=bias_axes[MODEL_FIG_ASSESSORS[0]] if i else None)
    rows, used_prompts = [], set()

    # (a) the SA PHQ-9-axis ladder, one line per generation prompt.
    for suffix, subdir in MULTIMODEL_SA_PROMPTS:
        ladder = phq9_adjacent_band_ladder(sa_roots.get(label, ""), emb_name=emb_name, subdir=subdir)
        if ladder is None or ladder.empty:
            continue
        _errorbar_annotated(ax_lin, ladder["cos"].to_numpy(), ladder["std"].to_numpy(),
                            MULTIMODEL_PROMPT_COLOURS[suffix], "-", marker,
                            MULTIMODEL_PROMPT_LABELS[suffix], annotate=False)
        used_prompts.add(suffix)
        rows += [{"generator": label, "prompt": subdir, "step": r.step, "cosine": r.cos, "std": r.std,
                  "n_reps": r.n_reps, "n_posts": r.n_anchors} for r in ladder.itertuples(index=False)]

    # (b) bottom row: the regressor fine-tuned on this generator's own posts, per arm.
    ax_bert = bias_axes["MentalBERT+MLP"]
    arms = [("human-opt.", posts_csv, seeds)]
    if label in MULTIMODEL_MINIMAL_ARMS:
        m_csv, m_dir = MULTIMODEL_MINIMAL_ARMS[label]
        m_seeds = [f for f in sorted(glob.glob(os.path.join(m_dir, "seed*.csv"))) if "summary" not in f]
        if os.path.isfile(m_csv) and m_seeds:
            arms.append(("minimal", m_csv, m_seeds))
    for prompt_key, arm_csv, arm_seeds in arms:
        colour = MULTIMODEL_PROMPT_COLOURS[prompt_key]
        used_prompts.add(prompt_key)
        persona_of = pd.read_csv(arm_csv).groupby("agent_id")["persona"].first().astype(str)
        keep_ids = set(persona_of.index[~persona_of.isin(seen)])
        bias = _band_bias_per_seed(arm_seeds, keep_ids)
        print(f"[model-linearity] {short} BERT ({prompt_key} posts): {len(keep_ids)} of "
              f"{len(persona_of)} blocks, {len(bias)} seeds")
        ax_bert.fill_between(x, bias.mean(0) - bias.std(0), bias.mean(0) + bias.std(0),
                             color=colour, alpha=0.2, linewidth=0)
        ax_bert.plot(x, bias.mean(0), "-", marker=marker, color=colour, linewidth=1.6,
                     markersize=5.5, markeredgecolor="black", markeredgewidth=0.5, zorder=3)

    # (b) top row: the prompted LLM on the same blocks.
    ax_llm = bias_axes["LLM assessor"]
    for gen, prompt_key, assess_key, pattern, llm_seeds, blocks_csv, select in MODEL_FIG_LLM_ASSESSORS:
        if gen != label or not os.path.isfile(blocks_csv):
            continue
        best = _textgrad_best_seed(pattern, llm_seeds) if select == "best" else None
        if select == "best" and best is None:
            continue
        use = [best] if select == "best" else llm_seeds
        blocks = pd.read_csv(blocks_csv).groupby("agent_id", sort=False)["persona"].first().astype(str)
        keep_ids = set(blocks.index[~blocks.isin(seen)])
        bias = _band_bias_per_seed([pattern.format(seed=s) for s in use], keep_ids,
                                   order_ids=blocks.index.to_numpy())
        if not len(bias):
            continue
        colour = MULTIMODEL_PROMPT_COLOURS[prompt_key]
        used_prompts.add(prompt_key)
        print(f"[model-linearity] {short} LLM ({prompt_key} posts): {len(keep_ids)} of "
              f"{len(blocks)} blocks, seeds {use}")
        if len(bias) > 1:  # a single run has no across-seed spread to show
            ax_llm.fill_between(x, bias.mean(0) - bias.std(0), bias.mean(0) + bias.std(0),
                                color=colour, alpha=0.2, linewidth=0)
        ax_llm.plot(x, bias.mean(0), "-", marker=marker, color=colour, linewidth=1.6,
                    markersize=5.5, markeredgecolor="black", markeredgewidth=0.5, zorder=3)

    # (c) teacher gradings of the same 300-block sets, one line per generation prompt.
    for prompt_key in MULTIMODEL_PROMPT_STYLES:
        path = gradings.get((label, prompt_key))
        if not path or not os.path.isfile(path):
            print(f"[model-linearity] no gradings: {short} ({prompt_key}) {path}")
            continue
        d = pd.read_csv(path)
        d = d[~d["persona"].astype(str).isin(seen)].dropna(subset=["score"])
        by_band = [d["score"][d["phq9"].between(lo, hi)] for lo, hi, _ in _MM_BANDS]
        mean = np.array([g.mean() for g in by_band])
        sem = np.array([g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0 for g in by_band])
        used_prompts.add(prompt_key)
        print(f"[model-linearity] {short} grades ({prompt_key}): {len(d)} blocks, mean {d['score'].mean():.2f}")
        _errorbar_annotated(ax_grade, mean, sem, MULTIMODEL_PROMPT_COLOURS[prompt_key], "-",
                            marker, MULTIMODEL_PROMPT_LABELS[prompt_key], annotate=False)

    ax_lin.set_xticks(np.arange(len(steps)))
    ax_lin.set_xticklabels(steps, rotation=30, ha="right", fontsize=8)
    ax_lin.set_ylabel("Cosine similarity", fontsize=9.5)
    ax_grade.set_ylabel("Teacher score (0–10)", fontsize=9.5)
    # Short tick labels: with a single arm plotted the range is narrow enough that
    # matplotlib picks two decimals, and those run left into (b).
    ax_grade.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    for ax in (*bias_axes.values(), ax_grade):
        ax.set_xticks(x)
        ax.set_xticklabels(_MM_BAND_SHORT, rotation=30, ha="right", fontsize=8)
    for ax in (ax_lin, ax_grade, *bias_axes.values()):
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.12, y=0.15)
    for i, name in enumerate(MODEL_FIG_ASSESSORS):  # the label IS the row's key
        ax = bias_axes[name]
        ax.axhline(0, color="black", linewidth=0.8)
        # Each row keeps its OWN scale. The two assessors differ by an order of magnitude
        # on Gemma (LLM ~-6, regressor ~-2..+1), and one shared scale flattened the
        # regressor row; the row labels and the zero line keep the two readable apart.
        ax.autoscale_view()
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.26 * (y1 - y0))  # room above the curves for the row's label
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5]))
        ax.text(0.03, 0.93, name, transform=ax.transAxes, ha="left", va="top",
                fontsize=7, fontweight="bold", color="0.15",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.6))
        if i < len(MODEL_FIG_ASSESSORS) - 1:  # shared band axis: only the bottom row is labelled
            ax.tick_params(labelbottom=False)

    # One y-label for the (b) stack, centred on it and clear of the widest tick label.
    bias_pos = [ax.get_position() for ax in bias_axes.values()]
    fig.canvas.draw()
    tick_left = min(lb.get_window_extent(fig.canvas.get_renderer()).x0 for ax in bias_axes.values()
                    for lb in ax.get_yticklabels() if lb.get_text())
    fig.text(fig.transFigure.inverted().transform((tick_left - 6, 0))[0],
             (max(b.y1 for b in bias_pos) + min(b.y0 for b in bias_pos)) / 2,
             "Bias = mean(pred − true)", rotation=90, ha="right", va="center", fontsize=9.5)

    # One legend: within a figure the only thing a line encodes is the generation prompt.
    handles = [Line2D([], [], color=MULTIMODEL_PROMPT_COLOURS[k], marker=marker, linewidth=2.2,
                      markersize=5.5, markeredgecolor="black", markeredgewidth=0.5,
                      label=MULTIMODEL_PROMPT_LABELS[k])
               for k in MULTIMODEL_PROMPT_STYLES if k in used_prompts]
    leg = fig.legend(handles=handles, title="Generation prompt", ncol=len(handles), loc="lower left",
                     bbox_to_anchor=(0, 1.01), borderaxespad=0, fontsize=7.5, framealpha=0.9,
                     handlelength=2.0, handletextpad=0.5, borderpad=0.4, columnspacing=1.2)
    leg.get_title().set_fontsize(7.5)
    leg.get_title().set_fontweight("bold")
    fig.canvas.draw()
    inv = fig.transFigure.inverted()

    columns = [([ax_lin], "(a) Adjacent-band cosine"),
               (list(bias_axes.values()), "(b) Assessor bias per band"),
               ([ax_grade], "(c) Post gradings per band")]
    cap_y = min(a.get_tightbbox(fig.canvas.get_renderer()).transformed(inv).y0
                for axes, _ in columns for a in axes) - 0.012
    for axes, panel in columns:
        pos = [a.get_position() for a in axes]
        fig.text((min(b.x0 for b in pos) + max(b.x1 for b in pos)) / 2, cap_y, panel,
                 ha="center", va="top", fontsize=9.5)

    box = leg.get_window_extent().transformed(inv)  # centre the legend above the panels
    y = max(a.get_position().y1 for a in (ax_lin, ax_grade, *bias_axes.values())) + 0.04
    leg.set_bbox_to_anchor(((1 - box.width) / 2, y), transform=fig.transFigure)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.splitext(out_path)[0] + ".csv", index=False)
    print(f"Per-model linearity/bias/grading plot → {out_path}")
    return out_path


def write_multimodel_post_samples(out_path: str, n_per_band: int = 1, seed: int = 0):
    """Markdown file with the same personas' posts from every generator, `n_per_band` blocks per severity band.

    Picks the sample agents from the first generator in `MULTIMODEL_TEST_POSTS`
    (the sets share personas and PHQ-9 targets) so the blocks are paired across
    generators, for reading post quality side by side.

    Args:
        out_path (str): markdown file to write.
        n_per_band (int): sampled blocks (agents) per band.
        seed (int): RNG seed of the agent sample.
    """
    sets = {short: pd.read_csv(csv) for short, csv in MULTIMODEL_TEST_POSTS.values() if os.path.isfile(csv)}
    if not sets:
        return None
    first = next(iter(sets.values()))
    agents = first.groupby("agent_id")[["persona", "phq9"]].first()
    rng = np.random.default_rng(seed)
    lines = ["# Paired post samples (human-optimized prompt, 300-block held-out sets)\n"]
    for lo, hi, name in _MM_BANDS:
        pool = agents.index[agents["phq9"].between(lo, hi)]
        for aid in rng.choice(pool, min(n_per_band, len(pool)), replace=False):
            lines.append(f"\n## {name} ({lo}–{hi}): agent {aid}, PHQ-9 = {agents.loc[aid, 'phq9']:.0f}\n")
            lines.append(f"Persona: {agents.loc[aid, 'persona']}\n")
            for short, d in sets.items():
                lines.append(f"\n**{short}**\n")
                lines += [f"{i + 1}. {t}" for i, t in enumerate(d.loc[d["agent_id"] == aid, "tweet"].astype(str))]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Paired post samples → {out_path}")
    return out_path


MULTIMODEL_LABELS = {"qwen": "Qwen3.5-27B", "gemma4": "Gemma-4-31B-it",
                     "mistral": "Mistral-Small-3.2-24B",
                     "qwen_minimal": "Qwen3.5-27B (minimal prompt)",
                     "gemma4_minimal": "Gemma-4-31B-it (minimal prompt)"}
_MM_BANDS = [(0, 4, "Minimal"), (5, 9, "Mild"), (10, 14, "Moderate"),
             (15, 19, "Mod. severe"), (20, 27, "Severe")]
_MM_OPT_DIR = "data/test_post/optimized_phq9/Qwen3.5-27B_seed{seed}"
# TextGrad assessment prompt optimized on the human-optimized corpus itself
# (train_posts.csv, minimal start prompt; jobs/run_prompt_optimizer_phq9_human.job).
_MM_OPT_HUMAN_DIR = "data/test_post/optimized_phq9_human/Qwen3.5-27B_seed{seed}"


def _multimodel_estimator_files(tag, bert_seeds, prompt_seeds):
    """Estimator name -> per-seed prediction CSVs (columns true_phq9, pred_phq9) for one generator."""
    # `tag` is the generator key (also a MULTIMODEL_LABELS key); `corpus` is its
    # assessor-tree name. The LLM-assessor dirs under optimized_phq9/ still use the
    # old tag spelling, so `held` is unchanged — that tree was not restructured.
    corpus = assessors.resolve(tag)
    held = "human300" if tag == "qwen" else f"{tag}300"
    bert = lambda d: [f"{d}/seed{s}.csv" for s in bert_seeds]
    prompt = lambda sub: [_MM_OPT_DIR.format(seed=s) + f"/{sub}/test_raw_scores.csv" for s in prompt_seeds]
    files = {
        # Each arm's own held-out set is the corpus it was fine-tuned on, so the
        # "own posts" cell is eval_dir(corpus, corpus).
        "BERT base (Qwen-trained)": bert(assessors.eval_dir("teacher", corpus)),
        "BERT fine-tuned (own posts)": bert(assessors.eval_dir(corpus, corpus)),
        "LLM prompt (minimal)": prompt(f"minimal_{held}"),
        "LLM prompt (TextGrad)": prompt(f"eval_on_{held}"),
    }
    if corpus != "qwen27_optimized":  # Qwen-fine-tuned regressors on this generator's posts
        files["BERT fine-tuned (Qwen posts)"] = bert(assessors.eval_dir("qwen27_optimized", corpus))
    # The human-corpus TextGrad run tests on the shared 300 Qwen blocks itself, so its
    # run-dir test_raw_scores.csv is that eval; other generators need a
    # `phq9-rerun-test --out-root data/test_post/optimized_phq9_human` first.
    human_sub = "" if tag == "qwen" else f"eval_on_{held}/"
    files["LLM prompt (TextGrad, human corpus)"] = [
        _MM_OPT_HUMAN_DIR.format(seed=s) + f"/{human_sub}test_raw_scores.csv" for s in prompt_seeds]
    return files


def run_multimodel_summary(argv=None):
    """CLI entry: generator x estimator MAE/bias table (CSV + LaTeX) and the two multimodel figures.

    Reads the per-seed prediction CSVs written by run_finetune.sh (MentalBERT+MLP)
    and run_llm_assessor_on_heldout.sh (Qwen LLM assessor). Missing inputs are
    skipped with a warning, so it can run while jobs are still pending.
    """
    ap = argparse.ArgumentParser(description="Generator x estimator MAE/bias summary.")
    ap.add_argument("--generators", nargs="+", default=["qwen", "gemma4", "mistral"])
    ap.add_argument("--bert-seeds", nargs="+", type=int, default=[34, 35, 36, 37, 38])
    ap.add_argument("--prompt-seeds", nargs="+", type=int, default=[23, 24, 25, 32, 33])
    ap.add_argument("--out-dir", default="data/test_post/method_comparison/multimodel")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    def band(score):
        return next(name for lo, hi, name in _MM_BANDS if lo <= score <= hi)

    rows, band_rows = [], []
    for tag in args.generators:
        gen = MULTIMODEL_LABELS.get(tag, tag)
        for est, paths in _multimodel_estimator_files(tag, args.bert_seeds, args.prompt_seeds).items():
            per_seed, pooled = [], []
            for p in paths:
                if not os.path.isfile(p):
                    print(f"[skip] {gen} / {est}: missing {p}")
                    continue
                df = pd.read_csv(p)
                err = df["pred_phq9"] - df["true_phq9"]
                per_seed.append({"mae": err.abs().mean(), "bias": err.mean()})
                pooled.append(df.assign(err=err))
            if not per_seed:
                continue
            ps = pd.DataFrame(per_seed)
            rows.append({"generator": gen, "estimator": est, "n_seeds": len(ps),
                         "mae": ps["mae"].mean(), "mae_sd": ps["mae"].std(ddof=0),
                         "bias": ps["bias"].mean(), "bias_sd": ps["bias"].std(ddof=0)})
            pooled = pd.concat(pooled)
            pooled["band"] = pooled["true_phq9"].map(band)
            for b, g in pooled.groupby("band", sort=False):
                band_rows.append({"generator": gen, "estimator": est, "band": b, "n": len(g),
                                  "mae": g["err"].abs().mean(), "bias": g["err"].mean()})

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{args.out_dir}/summary.csv", index=False)
    by_band = pd.DataFrame(band_rows)
    order = [name for _, _, name in _MM_BANDS]
    if len(by_band):
        by_band["band"] = pd.Categorical(by_band["band"], order)
        by_band = by_band.sort_values(["generator", "estimator", "band"])
    by_band.to_csv(f"{args.out_dir}/summary_by_band.csv", index=False)
    print(summary.round(2).to_string(index=False))

    # LaTeX table (booktabs, same style as the SI fine-tune table).
    lines = ["\\begin{tabular}{llcc}", "\\toprule",
             "Generator & Estimator & MAE & Bias \\\\", "\\midrule"]
    for gen, g in summary.groupby("generator", sort=False):
        for i, r in enumerate(g.itertuples()):
            name = gen if i == 0 else ""
            lines.append(f"{name} & {r.estimator} & ${r.mae:.2f} \\pm {r.mae_sd:.2f}$ & ${r.bias:+.2f}$ \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    with open(f"{args.out_dir}/table_multimodel.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    plot_multimodel_linearity_bias(f"{args.out_dir}/linearity_bias.png")
    if all(os.path.isdir(r) for r in MULTIMODEL_SA_ROOTS_SEEDED.values()):  # same figure, seeded SA in (a)
        plot_multimodel_linearity_bias(f"{args.out_dir}/linearity_bias_seeded.png",
                                       sa_roots=MULTIMODEL_SA_ROOTS_SEEDED)
    for gen in MULTIMODEL_TEST_POSTS:  # the same three panels per generator, (b) split by assessor
        tag = MULTIMODEL_TEST_POSTS[gen][0].lower()
        plot_model_linearity_bias(gen, f"{args.out_dir}/linearity_bias_{tag}.png")  # seeded (a)
        plot_model_linearity_bias(gen, f"{args.out_dir}/linearity_bias_{tag}_unseeded.png",
                                  sa_roots=MULTIMODEL_SA_ROOTS)
    write_multimodel_post_samples(f"{args.out_dir}/sample_posts.md")
    print(f"[done] -> {args.out_dir}/")


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
# Figure 1: BERT non-FT (human-opt) | fine-tuned (human-opt) | non-FT (synthetic).
# Figure 2: BERT vs assessment prompt, each on {synthetic, human-opt} test.
# Error bars are the SD of the per-sample errors, averaged over seeds. Both
# methods are scored on the same test files, so the synthetic-vs-human gap is
# the distribution shift (see run_eval_comparison for the paths).
# CLI: scripts/assessment/run_eval_comparison.sh.
# =========================================================================== #
_EVAL_MODEL_SHORT = "Qwen3.5-27B"
# Match the prompt-sensitivity (SA) palette used elsewhere in the thesis
# (sa_analyze._COLOUR_BY_NAME): Agent=blue, Joint=orange, Neighbour=dark-red.
_EVAL_C_SYNTH = "#2e7ebc"   # synthetic (in-distribution) test  (SA "Agent" blue)
_EVAL_C_HUMAN = "#d96907"   # human-optimized-prompt test       (SA "Joint" orange)
_EVAL_C_FT = "#8d2c03"      # fine-tuned bar in figure 1         (SA "Neighbour" dark-red)

# Subdir under each prompt seed dir holding the OPTIMISED prompt scored on the
# aligned synthetic test (the BERT regression test blocks, test_blocks_seed35.csv),
# rather than the prompt's own optimisation split, see the alignment note above.
_EVAL_PROMPT_SYNTH_SUBDIR = "eval_on_test_blocks_seed35"

# Optional minimal-prompt bars in figure 2 (added only if scored on disk).
# Generated with prompt_optimizer.py --mode phq9-rerun-test --instruction-filename
# minimal_instruction.txt --result-subdir <name>, per seed under prompt_dir. Only
# the seed23 minimal run exists on disk, so this group is a single run (no seed
# error bar); both bars must come from the SAME seed for the shift to be valid.
# Synthetic uses the same test_blocks_seed35 set as the optimised-prompt synthetic
# bar; human-opt uses the 300-block data/finetune/qwen/test_posts_qwen.csv so that BERT
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
    are present, used for optional bars whose data may not be on disk yet."""
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


def collect_eval_comparison(bert_arm: str, bert_ft_arm: str, prompt_dir: str,
                            prompt_eval_subdir: str, bert_seeds: list, prompt_seeds: list,
                            prompt_synth_subdir: str = _EVAL_PROMPT_SYNTH_SUBDIR,
                            corpus: str = "qwen27_optimized") -> dict:
    """Resolve every bar's raw-score files and aggregate them into a stats dict.

    `bert_arm` / `bert_ft_arm` are assessor arms (utils.assessors.ARMS) and
    `corpus` the held-out set both are scored on; the human-opt bars are that
    arm x corpus cell, the synthetic bars each arm's own native test split.
    """
    eval_baseline = assessors.eval_dir(bert_arm, corpus)
    eval_finetuned = assessors.eval_dir(bert_ft_arm, corpus)
    bert_dir = assessors.models_dir(bert_arm)
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
    # Optional minimal-prompt group, added to figure 2 only if scored on disk.
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


def _print_eval_table(stats: dict):
    """Print the MAE / bias table behind the estimator-comparison figures."""
    print(f"\n{'bar':<28}{'MAE':>8}{'±smpl':>8}{'±seed':>8}"
          f"{'bias':>9}{'±smpl':>8}{'±seed':>8}{'seeds':>7}{'n/seed':>8}")
    for v in stats.values():
        print(f"{v['label']:<28}{v['mae']:>8.3f}{v['mae_err_sample']:>8.3f}{v['mae_err_seed']:>8.3f}"
              f"{v['bias']:>9.3f}{v['bias_err_sample']:>8.3f}{v['bias_err_seed']:>8.3f}"
              f"{v['n_seeds']:>7d}{v['n_per_seed']:>8d}")


def run_eval_comparison(argv=None):
    """CLI entry: print the estimator-comparison table (BERT vs prompt).

    Backs Tables 1-2 of the paper / Results §PHQ-9 Assessment. The two companion
    figures (fig1_bert_finetune, fig2_bert_vs_prompt_robustness) were retired
    2026-09-22; the printed table is the surviving output.
    """
    p = argparse.ArgumentParser(description="MAE/bias comparison table (BERT vs prompt).")
    p.add_argument("--bert-arm", default="teacher",
                   help="Assessor arm for the non-fine-tuned bars (utils.assessors.ARMS).")
    p.add_argument("--bert-ft-arm", default="qwen27_optimized",
                   help="Assessor arm for the fine-tuned bars.")
    p.add_argument("--corpus", default="qwen27_optimized",
                   help="Held-out corpus both arms are scored on (eval/on_<corpus>/).")
    p.add_argument("--prompt-dir", default="data/test_post/optimized_phq9",
                   help="Holds {MODEL}_seed*/ (synthetic test) and the eval-on-prompt subdir (human-opt).")
    p.add_argument("--prompt-eval-subdir", default="eval_on_human300",
                   help="Subdir under each prompt seed dir holding the human-opt-test raw scores "
                        "(the 300-block data/finetune/qwen/test_posts_qwen.csv eval, paired with the BERT arm's "
                        "eval/on_<corpus>/).")
    p.add_argument("--prompt-synth-subdir", default=_EVAL_PROMPT_SYNTH_SUBDIR,
                   help="Subdir under each prompt seed dir holding the aligned synthetic-test raw scores "
                        "(the prompt scored on the BERT test blocks, test_blocks_seed35.csv).")
    p.add_argument("--bert-seeds", type=int, nargs="+", default=[34, 35, 36, 37, 38])
    p.add_argument("--prompt-seeds", type=int, nargs="+", default=[23, 24, 25, 32, 33])
    args = p.parse_args(argv)

    stats = collect_eval_comparison(args.bert_arm, args.bert_ft_arm, args.prompt_dir,
                                    args.prompt_eval_subdir, args.bert_seeds, args.prompt_seeds,
                                    prompt_synth_subdir=args.prompt_synth_subdir,
                                    corpus=args.corpus)
    _print_eval_table(stats)


if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")  # CLI only; never set a backend at import time (the notebook imports this module)
    _CMDS = {"eval-comparison": run_eval_comparison, "multimodel": run_multimodel_summary}
    if len(sys.argv) < 2 or sys.argv[1] not in _CMDS:
        sys.exit(f"usage: python -m utils.visualization {{{'|'.join(_CMDS)}}} [flags]  (--help per command)")
    _CMDS[sys.argv[1]](sys.argv[2:])
