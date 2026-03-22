"""Command-line interface for network-classifier."""

import argparse
import re

from rich.console import Console
from rich.table import Table
from network_classifier.centrality import compute_centrality
from network_classifier.classify import classify_edges, cluster_summary
from network_classifier.export import export_geopackage, export_graphml
from network_classifier.graph import load_graph

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
        "-n",
        "--network-type",
        default="drive",
        choices=["drive", "walk", "bike", "all"],
        help="Network type (default: drive)",
    )
    parser.add_argument(
        "-m",
        "--method",
        default=None,
        choices=["gmm", "kmeans"],
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

    console.log(f"Loading graph for [bold]{args.city}[/bold] (network_type={args.network_type})...")
    G = load_graph(args.city, args.network_type)
    console.log(f"Graph loaded: [green]{G.number_of_nodes()}[/green] nodes, [green]{G.number_of_edges()}[/green] edges")

    console.log("Computing centrality metrics (betweenness, closeness, degree)...")
    G = compute_centrality(G)
    console.log("[green]Centrality metrics computed.[/green]")

    if args.method is not None:
        console.log(
            f"Classifying edges using [bold]{args.method.upper()}[/bold]..."
        )
        G, k, model_metrics = classify_edges(G, args.method, args.n_clusters)
        console.log(
            f"[green]Classification complete.[/green] "
            f"Selected [bold]{k}[/bold] clusters"
        )
        _print_model_metrics(args.method, model_metrics)
        summary = cluster_summary(G)
        _print_cluster_summary(summary)

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
        "bic": "BIC",
        "aic": "AIC",
        "log_likelihood": "Log-Likelihood",
        "n_iter": "Iterations",
    }

    for key, value in metrics.items():
        label = labels.get(key, key)
        if key == "n_iter":
            table.add_row(label, str(value))
        else:
            table.add_row(label, f"{value:.4f}")

    console.print(table)


def _print_cluster_summary(summary: dict) -> None:
    """Print per-cluster distribution of centrality metrics."""
    for cluster_id in sorted(summary):
        metrics = summary[cluster_id]
        count = int(metrics["betweenness"]["count"])

        console.print(
            f"\n[bold]Cluster {cluster_id}[/bold] ({count} edges)"
        )

        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        for metric_name in ("betweenness", "closeness", "degree"):
            stats = metrics[metric_name]
            table.add_row(
                metric_name,
                f"{stats['mean']:.6f}",
                f"{stats['std']:.6f}",
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}",
            )

        console.print(table)
