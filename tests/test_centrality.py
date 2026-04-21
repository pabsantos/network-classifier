"""Tests for network_classifier.centrality."""

import networkx as nx

from network_classifier.centrality import compute_centrality


def _make_simple_graph() -> nx.MultiDiGraph:
    """A tiny graph where centrality values are easy to reason about."""
    G = nx.MultiDiGraph()
    #  0 --> 1 --> 2 --> 3
    for i in range(4):
        G.add_node(i)
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        G.add_edge(u, v, key=0, length=100.0)
    return G


class TestComputeCentrality:
    def test_attributes_added(self):
        G = _make_simple_graph()
        G = compute_centrality(G)
        for u, v, key, data in G.edges(keys=True, data=True):
            assert "betweenness" in data
            assert "clustering" in data
            assert "degree" in data

    def test_betweenness_non_negative(self):
        G = _make_simple_graph()
        G = compute_centrality(G)
        for *_, data in G.edges(data=True):
            assert data["betweenness"] >= 0.0

    def test_clustering_in_range(self):
        G = _make_simple_graph()
        G = compute_centrality(G)
        for *_, data in G.edges(data=True):
            assert 0.0 <= data["clustering"] <= 1.0

    def test_degree_positive(self):
        G = _make_simple_graph()
        G = compute_centrality(G)
        for *_, data in G.edges(data=True):
            assert data["degree"] > 0

    def test_center_edge_highest_betweenness(self):
        """In a chain 0->1->2->3, edge 1->2 should have the highest
        betweenness because it lies on the most shortest paths."""
        G = _make_simple_graph()
        G = compute_centrality(G)
        scores = {
            (u, v): data["betweenness"]
            for u, v, _key, data in G.edges(keys=True, data=True)
        }
        assert scores[(1, 2)] >= scores[(0, 1)]
        assert scores[(1, 2)] >= scores[(2, 3)]
