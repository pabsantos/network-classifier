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
    }
    _INT_KEYS = {"n_iter", "grid_side", "n_neurons"}

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
        if key in _INT_KEYS:
            lines.append(f"  {label:40s} {value}")
        else:
            lines.append(f"  {label:40s} {value:.6f}")

    lines.append("")
    Path(filepath).write_text("\n".join(lines), encoding="utf-8")
