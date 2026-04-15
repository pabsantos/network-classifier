"""Load street network graphs via OSMnx."""

import networkx as nx
import osmnx as ox


def load_graph(place: str, network_type: str = "drive") -> nx.MultiDiGraph:
    """Download a street network graph for the given place.

    Parameters
    ----------
    place : str
        Name of the city/region (e.g. "Curitiba, Brazil").
    network_type : str
        One of "drive", "walk", "bike", "all".

    Returns
    -------
    nx.MultiDiGraph
        The street network graph.
    """
    G = ox.graph_from_place(place, network_type=network_type)
    # Consolidate nearby intersections (e.g. dual carriageways, complex
    # junctions) into single nodes so centrality reflects topology rather
    # than OSM digitisation artifacts. Requires a projected graph.
    G_proj = ox.project_graph(G)
    G = ox.consolidate_intersections(
        G_proj, tolerance=15, rebuild_graph=True, dead_ends=True
    )
    # Consolidation marks the graph as unsimplified; collapse any degree-2
    # chains that remain between consolidated nodes so edges map 1:1 to
    # logical road segments.
    if not G.graph.get("simplified"):
        G = ox.simplify_graph(G)
    return G
