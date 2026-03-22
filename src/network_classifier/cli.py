"""Command-line interface for network-classifier."""

import argparse
import re

from rich.console import Console
from network_classifier.centrality import compute_centrality
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

    args = parser.parse_args()

    output = args.output or _default_output(args.city, args.format)

    console.log(f"Loading graph for [bold]{args.city}[/bold] (network_type={args.network_type})...")
    G = load_graph(args.city, args.network_type)
    console.log(f"Graph loaded: [green]{G.number_of_nodes()}[/green] nodes, [green]{G.number_of_edges()}[/green] edges")

    console.log("Computing centrality metrics (betweenness, closeness, degree)...")
    G = compute_centrality(G)
    console.log("[green]Centrality metrics computed.[/green]")

    console.log(f"Exporting to [bold]{output}[/bold]...")
    if args.format == "graphml":
        export_graphml(G, output)
    else:
        export_geopackage(G, output)
    console.log("[bold green]Done.[/bold green]")
