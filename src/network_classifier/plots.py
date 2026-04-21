"""Generate plots for classified networks."""

from pathlib import Path

import contextily as cx
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import RegularPolygon
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering
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
    The betweenness and clustering plots use a log-scaled x-axis (zero
    values are excluded from the KDE since log(0) is undefined).
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
        use_log = metric in ("betweenness", "clustering")

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
    fig.savefig(filepath, dpi=300)
    plt.close(fig)


def plot_umatrix(
    som,
    neuron_label_grid: np.ndarray,
    filepath: Path,
    n_clusters: int,
) -> None:
    """Save a U-matrix of the trained SOM alongside neuron-cluster assignments.

    Renders the hexagonal SOM grid as actual hexagons (pointy-top), using the
    Euclidean coordinates returned by ``som.get_euclidean_coordinates()``.

    Left panel: U-matrix (normalized inter-neuron distances). Darker cells
    indicate larger gaps between neighbouring neurons — i.e. cluster borders.
    Right panel: cluster id assigned to each neuron after the second-stage
    K-Means on the SOM codebook.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    umatrix = som.distance_map()
    xx, yy = som.get_euclidean_coordinates()
    grid_x, grid_y = umatrix.shape

    # Pointy-top hexagons whose horizontal width equals 1 (matching minisom's
    # unit spacing between neurons in the same row).
    hex_radius = 1.0 / np.sqrt(3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: U-matrix ----
    norm = Normalize(vmin=float(umatrix.min()), vmax=float(umatrix.max()))
    cmap_u = plt.get_cmap("bone_r")

    for i in range(grid_x):
        for j in range(grid_y):
            axes[0].add_patch(
                RegularPolygon(
                    (xx[i, j], yy[i, j]),
                    numVertices=6,
                    radius=hex_radius,
                    orientation=0.0,
                    facecolor=cmap_u(norm(umatrix[i, j])),
                    edgecolor="gray",
                    linewidth=0.3,
                )
            )

    _setup_hex_axis(axes[0], xx, yy)
    axes[0].set_title("U-Matrix \u2014 neuron distances")
    sm_u = plt.cm.ScalarMappable(cmap=cmap_u, norm=norm)
    fig.colorbar(sm_u, ax=axes[0], label="Normalized distance")

    # ---- Right: cluster assignment per neuron ----
    cluster_colors = _cluster_colors(n_clusters)
    cluster_cmap = ListedColormap(cluster_colors)
    cluster_norm = Normalize(vmin=-0.5, vmax=n_clusters - 0.5)

    for i in range(grid_x):
        for j in range(grid_y):
            axes[1].add_patch(
                RegularPolygon(
                    (xx[i, j], yy[i, j]),
                    numVertices=6,
                    radius=hex_radius,
                    orientation=0.0,
                    facecolor=cluster_colors[int(neuron_label_grid[i, j])],
                    edgecolor="white",
                    linewidth=0.5,
                )
            )

    _setup_hex_axis(axes[1], xx, yy)
    axes[1].set_title("SOM neurons \u2014 cluster assignment")
    sm_c = plt.cm.ScalarMappable(cmap=cluster_cmap, norm=cluster_norm)
    cbar = fig.colorbar(sm_c, ax=axes[1], ticks=range(n_clusters))
    cbar.set_label("Cluster")

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def _setup_hex_axis(ax, xx: np.ndarray, yy: np.ndarray) -> None:
    """Configure axis limits and aspect ratio for a hex SOM panel."""
    margin = 1.0
    ax.set_xlim(float(xx.min()) - margin, float(xx.max()) + margin)
    ax.set_ylim(float(yy.min()) - margin, float(yy.max()) + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")


def plot_silhouette_vs_k(
    scores: dict[int, float],
    selected_k: int,
    filepath: Path,
) -> None:
    """Save a line plot of silhouette score vs k, highlighting the selected k.

    Parameters
    ----------
    scores : dict[int, float]
        Mapping of k to its average silhouette score.
    selected_k : int
        The k used for the final clustering (highlighted on the curve).
    filepath : Path
        Destination PNG path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    ks = sorted(scores)
    values = [scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, values, marker="o", color="#4363d8", linewidth=2)

    if selected_k in scores:
        ax.scatter(
            [selected_k],
            [scores[selected_k]],
            color="#e6194b",
            s=120,
            zorder=5,
            label=f"Selected k = {selected_k}",
        )
        ax.axvline(selected_k, color="#e6194b", linestyle="--", alpha=0.4)

    ax.set_xticks(ks)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score vs k")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def _build_linkage_matrix(model: AgglomerativeClustering) -> np.ndarray:
    """Build a scipy-style linkage matrix from a fitted sklearn model.

    Requires the model to have been fitted with ``compute_distances=True``.
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = model.n_leaves_
    for i, merge in enumerate(model.children_):
        count = 0
        for child in merge:
            if child < n_samples:
                count += 1
            else:
                count += counts[child - n_samples]
        counts[i] = count

    return np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)


def plot_dendrogram(
    model: AgglomerativeClustering,
    n_clusters: int,
    filepath: Path,
    truncate_p: int = 30,
) -> None:
    """Save a dendrogram plot from a fitted AgglomerativeClustering model.

    Parameters
    ----------
    model : AgglomerativeClustering
        Fitted model with ``compute_distances=True``.
    n_clusters : int
        Number of clusters selected (used to draw the cut line).
    filepath : Path
        Destination PNG path.
    truncate_p : int
        Show only the last *p* merges (``truncate_mode='lastp'``).
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    Z = _build_linkage_matrix(model)

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=truncate_p,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0,
        above_threshold_color="#4363d8",
    )

    # Draw horizontal line at the cut distance
    if n_clusters >= 2:
        cut_distance = (
            Z[-(n_clusters - 1), 2] + Z[-n_clusters, 2]
        ) / 2 if n_clusters <= len(Z) else Z[-1, 2]
        ax.axhline(
            y=cut_distance,
            color="#e6194b",
            linestyle="--",
            linewidth=1.5,
            label=f"Cut (k={n_clusters})",
        )
        ax.legend()

    ax.set_xlabel("Cluster size")
    ax.set_ylabel("Distance")
    ax.set_title("Dendrogram")
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
