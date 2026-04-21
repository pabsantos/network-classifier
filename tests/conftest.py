"""Shared fixtures for network-classifier tests."""

import networkx as nx
import numpy as np
import pytest


@pytest.fixture()
def sample_graph() -> nx.MultiDiGraph:
    """Build a small synthetic MultiDiGraph with centrality attributes.

    The graph has 10 nodes arranged in a chain with a few cross-edges,
    giving enough topological variety for clustering to produce distinct
    groups.  Every edge carries ``betweenness``, ``clustering``, ``degree``,
    ``length``, and ``highway`` attributes so that downstream modules
    (classify, export, plots) can consume it without hitting KeyError.
    """
    rng = np.random.RandomState(42)
    G = nx.MultiDiGraph()

    for i in range(10):
        G.add_node(i, x=float(i), y=0.0)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 9),
        (0, 5), (2, 7), (1, 8),
    ]

    highway_types = [
        "residential", "secondary", "tertiary", "primary",
        "residential", "secondary", "tertiary", "primary",
        "residential", "trunk", "primary", "secondary",
    ]

    for idx, (u, v) in enumerate(edges):
        G.add_edge(
            u, v, key=0,
            betweenness=rng.exponential(0.05),
            clustering=rng.uniform(0.0, 0.5),
            degree=float(rng.randint(2, 10)),
            length=rng.uniform(50, 500),
            highway=highway_types[idx],
        )

    return G


@pytest.fixture()
def classified_graph(sample_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return the sample graph after K-Means classification (k=3)."""
    from network_classifier.classify import classify_edges

    G, _k, _metrics, _extras = classify_edges(sample_graph, "kmeans", n_clusters=3)
    return G
