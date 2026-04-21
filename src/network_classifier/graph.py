"""Load street network graphs via OSMnx."""

import json
from pathlib import Path

import networkx as nx
import osmnx as ox
from shapely.geometry import shape


def _consolidate(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
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


def load_graph(place: str, network_type: str = "drive") -> nx.MultiDiGraph:
    G = ox.graph_from_place(place, network_type=network_type)
    return _consolidate(G)


def load_graph_from_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    network_type: str = "drive",
) -> nx.MultiDiGraph:
    G = ox.graph_from_bbox(
        (west, south, east, north), network_type=network_type
    )
    return _consolidate(G)


def load_graph_from_polygon(
    polygon_path: str | Path, network_type: str = "drive"
) -> nx.MultiDiGraph:
    data = json.loads(Path(polygon_path).read_text())
    # Accept a Feature, FeatureCollection (first feature), or bare geometry.
    if data.get("type") == "FeatureCollection":
        geom = data["features"][0]["geometry"]
    elif data.get("type") == "Feature":
        geom = data["geometry"]
    else:
        geom = data
    polygon = shape(geom)
    G = ox.graph_from_polygon(polygon, network_type=network_type)
    return _consolidate(G)
