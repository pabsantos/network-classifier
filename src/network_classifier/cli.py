"""Command-line interface for network-classifier."""

import argparse
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from network_classifier.centrality import compute_centrality
from network_classifier.classify import classify_edges, highway_cluster_crosstab
from network_classifier.export import export_geopackage, export_graphml, export_txt
from network_classifier.graph import (
    load_graph,
    load_graph_from_bbox,
    load_graph_from_polygon,
)
from network_classifier.plots import (
    plot_crosstab_heatmap,
    plot_dendrogram,
    plot_kde,
    plot_map,
    plot_silhouette_vs_k,
    plot_umatrix,
)

console = Console()


def _default_output(label: str, fmt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    ext = "graphml" if fmt == "graphml" else "gpkg"
    return f"{slug}.{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="network-classifier",
        description="Classify road networks using edge betweenness centrality.",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "city",
        nargs="?",
        default=None,
        help='City name (e.g. "Curitiba, Brazil")',
    )
    source.add_argument(
        "--bbox",
        type=str,
        metavar="WEST,SOUTH,EAST,NORTH",
        help="Bounding box as comma-separated floats: west,south,east,north",
    )
    source.add_argument(
        "--polygon",
        metavar="FILE",
        help="Path to a GeoJSON file whose first polygon defines the boundary",
    )

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
        help="Output file path (default: derived from input name)",
    )
    parser.add_argument(
        "-m",
        "--method",
        default=None,
        choices=["kmeans", "som", "fkmeans", "hc_sl", "hc_cl", "hc_ward", "hc_al"],
        help="Clustering method (default: no clustering). "
             "hc_sl=single linkage, hc_cl=complete linkage, "
             "hc_ward=Ward, hc_al=average linkage",
    )
    parser.add_argument(
        "-k",
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters (default: auto-select best k)",
    )

    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--bbox" and i + 1 < len(argv):
            argv = argv[:i] + [f"--bbox={argv[i + 1]}"] + argv[i + 2:]
            break
    args = parser.parse_args(argv)

    if args.bbox is not None:
        parts = args.bbox.split(",")
        if len(parts) != 4:
            parser.error("--bbox requires exactly 4 comma-separated values: west,south,east,north")
        west, south, east, north = (float(v) for v in parts)
        label = f"bbox_{west}_{south}_{east}_{north}"
        output = args.output or _default_output(label, args.format)
        output_path = Path(output)
        console.log(
            f"Loading graph for bbox [bold]{west},{south},{east},{north}[/bold]..."
        )
        G = load_graph_from_bbox(west, south, east, north, "drive")
    elif args.polygon is not None:
        label = Path(args.polygon).stem
        output = args.output or _default_output(label, args.format)
        output_path = Path(output)
        console.log(f"Loading graph from polygon [bold]{args.polygon}[/bold]...")
        G = load_graph_from_polygon(args.polygon, "drive")
    else:
        label = args.city
        output = args.output or _default_output(label, args.format)
        output_path = Path(output)
        console.log(f"Loading graph for [bold]{label}[/bold]...")
        G = load_graph(label, "drive")
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

        if "silhouette_scores" in extras:
            silhouette_path = plot_dir / "silhouette_vs_k.png"
            console.log("Generating silhouette score vs k plot...")
            plot_silhouette_vs_k(
                extras["silhouette_scores"], k, silhouette_path
            )
            console.log(f"  Saved [bold]{silhouette_path}[/bold]")

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

        if "hc_model" in extras:
            dendro_path = plot_dir / "dendrogram.png"
            console.log("Generating dendrogram...")
            plot_dendrogram(extras["hc_model"], k, dendro_path)
            console.log(f"  Saved [bold]{dendro_path}[/bold]")

        txt_path = plot_dir / "model_metrics.txt"
        export_txt(
            txt_path,
            city=label,
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
        "fpc": "Fuzzy Partition Coefficient",
        "linkage": "Linkage Method",
        "n_leaves": "Leaves",
    }
    int_keys = {"n_iter", "grid_side", "n_neurons", "n_leaves"}
    str_keys = {"linkage"}

    for key, value in metrics.items():
        label = labels.get(key, key)
        if key in str_keys:
            table.add_row(label, str(value))
        elif key in int_keys:
            table.add_row(label, str(value))
        else:
            table.add_row(label, f"{value:.4f}")

    console.print(table)
