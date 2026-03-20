"""Export graphs to GraphML or GeoPackage."""

import networkx as nx
import osmnx as ox


def export_graphml(G: nx.MultiDiGraph, filepath: str) -> None:
    """Save graph as GraphML."""
    ox.save_graphml(G, filepath)


def export_geopackage(G: nx.MultiDiGraph, filepath: str) -> None:
    """Save graph as GeoPackage."""
    ox.save_graph_geopackage(G, filepath=filepath)
