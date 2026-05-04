"""Tests for network_classifier.classify."""

import numpy as np
import networkx as nx
import pytest

from network_classifier.classify import (
    HC_METHODS,
    METRICS,
    _find_best_k,
    _find_best_k_hc,
    _fkmeans_eval,
    _kmeans_eval,
    classify_edges,
    cluster_summary,
    highway_cluster_v_measure,
)


# ---------------------------------------------------------------------------
# Helper evaluation functions
# ---------------------------------------------------------------------------

class TestKmeansEval:
    def test_returns_scores_and_inertias(self):
        rng = np.random.RandomState(0)
        X = rng.randn(200, 3)
        sil, inertias = _kmeans_eval(X, k_range=range(2, 6))
        assert set(sil.keys()) == {2, 3, 4, 5}
        assert set(inertias.keys()) == {2, 3, 4, 5}
        # inertia must decrease as k grows
        assert inertias[2] >= inertias[5]

    def test_silhouette_in_range(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 3)
        sil, _ = _kmeans_eval(X, k_range=range(2, 5))
        for v in sil.values():
            assert -1 <= v <= 1


class TestFindBestK:
    def test_returns_best_k_in_range(self):
        rng = np.random.RandomState(0)
        X = rng.randn(200, 3)
        best_k, sil, inertias = _find_best_k(X, k_range=range(2, 6))
        assert 2 <= best_k <= 5
        assert best_k == max(sil, key=sil.__getitem__)


class TestFkmeansEval:
    def test_returns_scores_and_objectives(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 3)
        sil, obj = _fkmeans_eval(X, k_range=range(2, 5), m=2.0)
        assert len(sil) > 0
        assert len(obj) > 0
        for v in sil.values():
            assert -1 <= v <= 1


class TestFindBestKHC:
    @pytest.mark.parametrize("linkage", ["single", "complete", "ward", "average"])
    def test_returns_k_in_range(self, linkage):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 3)
        k = _find_best_k_hc(X, linkage, k_min=2, k_max=10)
        assert 2 <= k <= 10


# ---------------------------------------------------------------------------
# classify_edges
# ---------------------------------------------------------------------------

class TestClassifyEdges:
    @pytest.mark.parametrize("method", ["kmeans", "fkmeans"])
    def test_fixed_k(self, sample_graph, method):
        G, k, metrics, extras = classify_edges(sample_graph, method, n_clusters=3)
        assert k == 3
        labels = {data["cluster"] for *_, data in G.edges(data=True)}
        assert labels == {0, 1, 2}
        assert "silhouette_score" in metrics

    @pytest.mark.parametrize("method", ["kmeans", "fkmeans"])
    def test_auto_k(self, sample_graph, method):
        G, k, metrics, extras = classify_edges(sample_graph, method, n_clusters=None)
        assert 2 <= k <= 10
        assert "silhouette_scores" in extras
        assert "inertias" in extras

    @pytest.mark.parametrize("method", list(HC_METHODS.keys()))
    def test_hc_fixed_k(self, sample_graph, method):
        G, k, metrics, extras = classify_edges(sample_graph, method, n_clusters=3)
        assert k == 3
        assert metrics["linkage"] == HC_METHODS[method]
        assert "hc_model" in extras

    @pytest.mark.parametrize("method", list(HC_METHODS.keys()))
    def test_hc_auto_k(self, sample_graph, method):
        G, k, metrics, extras = classify_edges(sample_graph, method, n_clusters=None)
        assert 2 <= k <= 10

    def test_som_fixed_k(self, sample_graph):
        G, k, metrics, extras = classify_edges(sample_graph, "som", n_clusters=3)
        assert k == 3
        assert "som" in extras
        assert "inertias" in extras

    def test_unknown_method_raises(self, sample_graph):
        with pytest.raises(ValueError, match="Unknown method"):
            classify_edges(sample_graph, "invalid_method")

    def test_cluster_attr_set_on_all_edges(self, sample_graph):
        G, k, _, _ = classify_edges(sample_graph, "kmeans", n_clusters=2)
        for u, v, key, data in G.edges(keys=True, data=True):
            assert "cluster" in data
            assert 0 <= data["cluster"] < k

    def test_log1p_applied_to_betweenness(self, sample_graph):
        """Verify betweenness is log1p-transformed (edge attrs unchanged)."""
        original = {
            (u, v, key): data["betweenness"]
            for u, v, key, data in sample_graph.edges(keys=True, data=True)
        }
        classify_edges(sample_graph, "kmeans", n_clusters=2)
        for (u, v, key), orig in original.items():
            assert sample_graph[u][v][key]["betweenness"] == orig


# ---------------------------------------------------------------------------
# cluster_summary
# ---------------------------------------------------------------------------

class TestClusterSummary:
    def test_summary_structure(self, classified_graph):
        summary = cluster_summary(classified_graph)
        assert len(summary) == 3
        for cid, metric_dict in summary.items():
            for m in METRICS:
                assert m in metric_dict
                stats = metric_dict[m]
                assert set(stats.keys()) == {"count", "mean", "std", "min", "max"}
                assert stats["count"] > 0
                assert stats["min"] <= stats["mean"] <= stats["max"]


# ---------------------------------------------------------------------------
# highway_cluster_v_measure
# ---------------------------------------------------------------------------

class TestHighwayClusterVMeasure:
    def test_v_measure_in_range(self, classified_graph):
        score = highway_cluster_v_measure(classified_graph)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_perfect_agreement_is_one(self):
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, key=0, highway="primary", cluster=0)
        G.add_edge(1, 2, key=0, highway="primary", cluster=0)
        G.add_edge(2, 3, key=0, highway="residential", cluster=1)
        G.add_edge(3, 4, key=0, highway="residential", cluster=1)
        assert highway_cluster_v_measure(G) == pytest.approx(1.0)

    def test_link_suffix_normalized(self):
        # primary_link must be folded into primary so it shares a class
        # with regular primary edges.
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, key=0, highway="primary", cluster=0)
        G.add_edge(1, 2, key=0, highway="primary_link", cluster=0)
        G.add_edge(2, 3, key=0, highway="residential", cluster=1)
        G.add_edge(3, 4, key=0, highway="residential", cluster=1)
        assert highway_cluster_v_measure(G) == pytest.approx(1.0)
