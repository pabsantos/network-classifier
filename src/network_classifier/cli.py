"""Command-line interface for network-classifier."""

import argparse
import re
from pathlib import Path

from rich.console import Console
from rich.table import Table
from network_classifier.centrality import compute_centrality
from network_classifier.classify import classify_edges, highway_cluster_crosstab
from network_classifier.export import export_geopackage, export_graphml, export_txt
from network_classifier.graph import load_graph
from network_classifier.plots import (
    plot_crosstab_heatmap,
    plot_kde,
    plot_map,
    plot_umatrix,
)

console = Console()


def _default_output(city: str, fmt: str) -> str:
    """Generate a default output filename from the city name."""
    slug = re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")
    ext = "graphml" if fmt == "graphml" else "gpkg"
    return f"{slug}.{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="network-classifier",
        description="Classify road networks using edge betweenness centrality.",
    )
    parser.add_argument("city", help='City name (e.g. "Curitiba, Brazil")')
    parser.add_argument(
        "-f",
        "--format",
        required=True,
        choices=["graphml", "gpkg"],
        help="Output format: graphml or gpkg",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: derived from city name)",
    )
    parser.add_argument(
        "-m",
        "--method",
        default=None,
        choices=["kmeans", "som"],
        help="Clustering method (default: no clustering)",
    )
    parser.add_argument(
        "-k",
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters (default: auto-select best k)",
    )

    args = parser.parse_args()

    output = args.output or _default_output(args.city, args.format)
    output_path = Path(output)

    console.log(f"Loading graph for [bold]{args.city}[/bold]...")
    G = load_graph(args.city, "drive")
    console.log(f"Graph loaded: [green]{G.number_of_nodes()}[/green] nodes, [green]{G.number_of_edges()}[/green] edges")

    console.log("Computing centrality metrics (betweenness, clustering, degree)...")
    G = compute_centrality(G)
    console.log("[green]Centrality metrics computed.[/green]")

    if args.method is not None:
        console.log(
            f"Classifying edges using [bold]{args.method.upper()}[/bold]..."
        )
        G, k, model_metrics, extras = classify_edges(
            G, args.method, args.n_clusters
        )
        console.log(
            f"[green]Classification complete.[/green] "
            f"Selected [bold]{k}[/bold] clusters"
        )
        _print_model_metrics(args.method, model_metrics)

        plot_dir = output_path.parent / "output"

        console.log("Generating KDE plots...")
        kde_paths = plot_kde(G, plot_dir)
        for p in kde_paths:
            console.log(f"  Saved [bold]{p}[/bold]")

        map_path = plot_dir / "map.png"
        console.log("Generating cluster map...")
        plot_map(G, map_path)
        console.log(f"  Saved [bold]{map_path}[/bold]")

        console.log("Generating highway x cluster heatmap...")
        ct = highway_cluster_crosstab(G)
        heatmap_path = plot_dir / "highway_cluster_heatmap.png"
        plot_crosstab_heatmap(ct, heatmap_path)
        console.log(f"  Saved [bold]{heatmap_path}[/bold]")

        if args.method == "som":
            umatrix_path = plot_dir / "umatrix.png"
            console.log("Generating SOM U-matrix...")
            plot_umatrix(
                extras["som"], extras["neuron_label_grid"], umatrix_path, k
            )
            console.log(f"  Saved [bold]{umatrix_path}[/bold]")

        txt_path = plot_dir / "model_metrics.txt"
        export_txt(
            txt_path,
            city=args.city,
            method=args.method,
            n_clusters=k,
            model_metrics=model_metrics,
        )
        console.log(f"  Saved [bold]{txt_path}[/bold]")

    console.log(f"Exporting to [bold]{output}[/bold]...")
    if args.format == "graphml":
        export_graphml(G, output)
    else:
        export_geopackage(G, output)
    console.log("[bold green]Done.[/bold green]")


def _print_model_metrics(method: str, metrics: dict) -> None:
    """Print model evaluation metrics."""
    table = Table(
        title=f"{method.upper()} Metrics",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    labels = {
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
    int_keys = {"n_iter", "grid_side", "n_neurons"}

    for key, value in metrics.items():
        label = labels.get(key, key)
        if key in int_keys:
            table.add_row(label, str(value))
        else:
            table.add_row(label, f"{value:.4f}")

    console.print(table)
