"""Export graphs to GraphML or GeoPackage."""

from pathlib import Path

import networkx as nx
import osmnx as ox


def export_graphml(G: nx.MultiDiGraph, filepath: str) -> None:
    """Save graph as GraphML."""
    ox.save_graphml(G, filepath)


def export_geopackage(G: nx.MultiDiGraph, filepath: str) -> None:
    """Save graph as GeoPackage."""
    ox.save_graph_geopackage(G, filepath=filepath)


def export_txt(
    filepath: str | Path,
    *,
    city: str,
    method: str,
    n_clusters: int,
    model_metrics: dict[str, float],
    pca_info: dict | None = None,
) -> None:
    """Save a plain-text report with model metrics."""
    _LABELS = {
        "inertia": "Inertia",
        "silhouette_score": "Silhouette Score",
        "n_iter": "Iterations",
        "quantization_error": "Quantization Error",
        "topographic_error": "Topographic Error",
        "grid_side": "Grid Side",
        "n_neurons": "Neurons",
        "kmeans_silhouette": "KMeans Silhouette (codebook)",
        "kmeans_inertia": "KMeans Inertia (codebook)",
        "kmeans_calinski_harabasz_score": "KMeans Calinski-Harabasz (codebook)",
        "calinski_harabasz_score": "Calinski-Harabasz Score",
        "bic": "BIC",
        "aic": "AIC",
        "log_likelihood": "Log-Likelihood",
        "converged": "Converged",
        "v_measure": "V-Measure (vs highway class)",
    }
    _INT_KEYS = {"n_iter", "grid_side", "n_neurons", "n_leaves"}

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Network Classifier — Model Metrics")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"City:       {city}")
    lines.append(f"Method:     {method.upper()}")
    lines.append(f"Clusters:   {n_clusters}")
    lines.append("")

    for key, value in model_metrics.items():
        label = _LABELS.get(key, key)
        if isinstance(value, (str, bool)):
            lines.append(f"  {label:40s} {value}")
        elif key in _INT_KEYS:
            lines.append(f"  {label:40s} {value}")
        else:
            lines.append(f"  {label:40s} {value:.6f}")

    if pca_info is not None:
        evr = pca_info["explained_variance_ratio"]
        ev = pca_info["explained_variance"]
        sv = pca_info["singular_values"]
        components = pca_info["components"]
        feature_names = pca_info["feature_names"]

        lines.append("")
        lines.append("-" * 60)
        lines.append("PCA Parameters")
        lines.append("-" * 60)
        lines.append(f"  {'Components':40s} 2")
        lines.append(f"  {'Input features':40s} {', '.join(feature_names)}")
        lines.append(
            f"  {'Explained variance ratio (PC1)':40s} {evr[0]:.6f}"
        )
        lines.append(
            f"  {'Explained variance ratio (PC2)':40s} {evr[1]:.6f}"
        )
        lines.append(
            f"  {'Cumulative explained variance':40s} {evr[0] + evr[1]:.6f}"
        )
        lines.append(f"  {'Explained variance (PC1)':40s} {ev[0]:.6f}")
        lines.append(f"  {'Explained variance (PC2)':40s} {ev[1]:.6f}")
        lines.append(f"  {'Singular value (PC1)':40s} {sv[0]:.6f}")
        lines.append(f"  {'Singular value (PC2)':40s} {sv[1]:.6f}")
        lines.append("")
        lines.append("  Loadings (rows=PC, cols=feature):")
        header = " " * 8 + "".join(f"{name:>14s}" for name in feature_names)
        lines.append(header)
        for i, row in enumerate(components, start=1):
            cells = "".join(f"{v:>14.6f}" for v in row)
            lines.append(f"    PC{i}{cells}")

    lines.append("")
    Path(filepath).write_text("\n".join(lines), encoding="utf-8")
