"""Load street network graphs via OSMnx."""

import osmnx as ox
import networkx as nx


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
    return ox.graph_from_place(place, network_type=network_type)
