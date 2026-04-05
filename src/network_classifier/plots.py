"""Generate plots for classified networks."""

from pathlib import Path

import contextily as cx
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
import osmnx as ox

from network_classifier.classify import METRICS

# High-contrast palette (colour-blind friendly, up to 10 clusters)
_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
    "#469990",  # teal
]


def _cluster_colors(n: int) -> list[str]:
    """Return *n* high-contrast colours."""
    return _COLORS[:n]


def plot_kde(G: nx.MultiDiGraph, output_dir: Path) -> list[Path]:
    """Save kernel density plots for each centrality metric, grouped by cluster.

    One PNG per metric is saved in *output_dir*, named ``<metric>_kde.png``.
    The betweenness plot uses a log-scaled x-axis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters: dict[int, dict[str, list[float]]] = {}

    for _u, _v, _key, data in G.edges(keys=True, data=True):
        cid = data["cluster"]
        if cid not in clusters:
            clusters[cid] = {m: [] for m in METRICS}
        for m in METRICS:
            clusters[cid][m].append(data[m])

    sorted_ids = sorted(clusters)
    colors = _cluster_colors(len(sorted_ids))

    saved: list[Path] = []
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        use_log = metric == "betweenness"

        for idx, cid in enumerate(sorted_ids):
            vals = np.array(clusters[cid][metric])
            if len(vals) < 2:
                continue

            if use_log:
                vals_kde = np.log10(vals[vals > 0])
                if len(vals_kde) < 2:
                    continue
                kde = gaussian_kde(vals_kde)
                lo, hi = vals_kde.min(), vals_kde.max()
                margin = (hi - lo) * 0.1 or 1e-6
                x_log = np.linspace(lo - margin, hi + margin, 300)
                x_real = 10**x_log
                density = kde(x_log)
                color = colors[idx]
                ax.plot(x_real, density, label=f"Cluster {cid}", color=color)
                ax.fill_between(x_real, density, alpha=0.15, color=color)
            else:
                kde = gaussian_kde(vals)
                lo, hi = vals.min(), vals.max()
                margin = (hi - lo) * 0.1 or 1e-6
                x = np.linspace(lo - margin, hi + margin, 300)
                color = colors[idx]
                ax.plot(x, kde(x), label=f"Cluster {cid}", color=color)
                ax.fill_between(x, kde(x), alpha=0.15, color=color)

        if use_log:
            ax.set_xscale("log")

        ax.set_xlabel(metric.capitalize())
        ax.set_ylabel("Density")
        ax.set_title(f"Kernel Density \u2014 {metric.capitalize()}")
        ax.legend()
        fig.tight_layout()

        path = output_dir / f"{metric}_kde.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_map(G: nx.MultiDiGraph, filepath: Path) -> None:
    """Save a map of edges colored by cluster class."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    gdf_edges = gdf_edges.to_crs(epsg=3857)
    n_clusters = gdf_edges["cluster"].nunique()
    cmap = ListedColormap(_cluster_colors(n_clusters))

    fig, ax = plt.subplots(figsize=(12, 12))
    gdf_edges.plot(
        column="cluster",
        categorical=True,
        legend=True,
        legend_kwds={"title": "Cluster"},
        ax=ax,
        linewidth=0.5,
        cmap=cmap,
    )
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    ax.set_axis_off()
    ax.set_title("Road Network \u2014 Clusters")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_crosstab_heatmap(ct: pd.DataFrame, filepath: Path) -> None:
    """Save a heatmap of the highway class x cluster cross-tabulation."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(ct.columns) * 2), max(5, len(ct) * 0.5)))
    im = ax.imshow(ct.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels([f"Cluster {c}" for c in ct.columns])
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index)

    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            val = ct.values[i, j]
            color = "white" if val > ct.values.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color,
                    fontsize=9)

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Highway class")
    ax.set_title("Highway class x Cluster (km)")
    fig.colorbar(im, ax=ax, label="Extension (km)")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
