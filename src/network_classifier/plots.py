"""Generate plots for classified networks."""

from pathlib import Path

import contextily as cx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import RegularPolygon
from scipy.cluster.hierarchy import dendrogram
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


def plot_violin(G: nx.MultiDiGraph, output_dir: Path) -> list[Path]:
    """Save violin plots for each centrality metric, grouped by cluster.

    One PNG per metric is saved in *output_dir*, named ``<metric>_violin.png``.
    Betweenness and clustering values are log10-transformed (zeros excluded)
    before plotting so the y-axis tick labels read as ``10^x``.
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

        data_per_cluster: list[np.ndarray] = []
        positions: list[int] = []
        face_colors: list[str] = []
        for idx, cid in enumerate(sorted_ids):
            vals = np.asarray(clusters[cid][metric], dtype=float)
            if use_log:
                vals = np.log10(vals[vals > 0])
            if len(vals) < 2:
                continue
            data_per_cluster.append(vals)
            positions.append(cid)
            face_colors.append(colors[idx])

        if not data_per_cluster:
            plt.close(fig)
            continue

        parts = ax.violinplot(
            data_per_cluster,
            positions=positions,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        for body, color in zip(parts["bodies"], face_colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.6)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            if key in parts:
                parts[key].set_color("#333333")
                parts[key].set_linewidth(1.0)

        ax.set_xticks(positions)
        ax.set_xticklabels([f"Cluster {cid}" for cid in positions])
        ax.set_xlabel("Cluster")
        ylabel = metric.capitalize()
        if use_log:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _pos: f"$10^{{{v:g}}}$")
            )
            ylabel = f"{metric.capitalize()} (log scale)"
        ax.set_ylabel(ylabel)
        ax.set_title(f"Violin plot \u2014 {metric.capitalize()}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        path = output_dir / f"{metric}_violin.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_pca_scatter(
    X_pca: np.ndarray,
    labels: np.ndarray,
    pca_info: dict,
    filepath: Path,
) -> None:
    """Save a scatter plot of samples projected onto PC1/PC2, coloured by cluster."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels)
    unique = sorted(set(int(c) for c in labels))
    colors = _cluster_colors(len(unique))
    evr = pca_info["explained_variance_ratio"]

    fig, ax = plt.subplots(figsize=(8, 7))
    for color, cid in zip(colors, unique):
        mask = labels == cid
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=8,
            alpha=0.5,
            color=color,
            label=f"Cluster {cid}",
            edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    ax.grid(True, alpha=0.3)
    legend = ax.legend(
        title="Cluster", loc="best", markerscale=2.0, frameon=True
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


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


def plot_performance(
    performance_per_k: dict[int, dict[str, float]],
    selected_k: int,
    filepath: Path,
) -> None:
    """Save a 2x2 performance plot (silhouette, CHI, V-measure, WCSS) vs k.

    Parameters
    ----------
    performance_per_k : dict[int, dict[str, float]]
        Mapping ``{k: {"silhouette", "chi", "v_measure", "wcss"}}``.
    selected_k : int
        The k used for the final clustering (highlighted on every panel).
    filepath : Path
        Destination PNG path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    ks = sorted(performance_per_k)

    panels = [
        ("silhouette", "Silhouette score"),
        ("chi", "Calinski-Harabasz"),
        ("v_measure", "V-Measure"),
        ("wcss", "WCSS"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.flatten()

    for ax, (key, ylabel) in zip(axes_flat, panels):
        values = [performance_per_k[k][key] for k in ks]
        ax.plot(ks, values, marker="o", color="#4363d8", linewidth=2)

        if selected_k in performance_per_k:
            ax.scatter(
                [selected_k],
                [performance_per_k[selected_k][key]],
                color="#e6194b",
                s=120,
                zorder=5,
            )
            ax.axvline(selected_k, color="#e6194b", linestyle="--", alpha=0.4)

        ax.set_xticks(ks)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Number of clusters (k)")
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




