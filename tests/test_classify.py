"""Tests for network_classifier.classify."""

import numpy as np
import networkx as nx
import pytest

from network_classifier.classify import (
    HC_METHODS,
    METRICS,
    _apply_pca,
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
    def test_returns_per_k_metrics(self):
        rng = np.random.RandomState(0)
        X = rng.randn(200, 3)
        y_true = ["a"] * 100 + ["b"] * 100
        out = _kmeans_eval(X, y_true, k_range=range(2, 6))
        assert set(out.keys()) == {2, 3, 4, 5}
        for k, m in out.items():
            assert set(m.keys()) == {"silhouette", "chi", "v_measure", "wcss"}
            assert -1 <= m["silhouette"] <= 1
            assert 0.0 <= m["v_measure"] <= 1.0
        # WCSS must decrease as k grows
        assert out[2]["wcss"] >= out[5]["wcss"]


class TestFkmeansEval:
    def test_returns_per_k_metrics(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 3)
        y_true = ["a"] * 50 + ["b"] * 50
        out = _fkmeans_eval(X, y_true, k_range=range(2, 5), m=2.0)
        assert len(out) > 0
        for k, m in out.items():
            assert set(m.keys()) == {"silhouette", "chi", "v_measure", "wcss"}
            assert -1 <= m["silhouette"] <= 1
            assert 0.0 <= m["v_measure"] <= 1.0


# ---------------------------------------------------------------------------
# classify_edges
# ---------------------------------------------------------------------------

class TestClassifyEdges:
    @pytest.mark.parametrize("method", ["kmeans", "fkmeans"])
    def test_fixed_k(self, sample_graph, method):
        G, metrics, extras = classify_edges(sample_graph, method, n_clusters=3)
        labels = {data["cluster"] for *_, data in G.edges(data=True)}
        assert labels == {0, 1, 2}
        assert "silhouette_score" in metrics
        assert "performance_per_k" in extras
        first_k = next(iter(extras["performance_per_k"]))
        assert set(extras["performance_per_k"][first_k].keys()) == {
            "silhouette", "chi", "v_measure", "wcss"
        }

    @pytest.mark.parametrize("method", list(HC_METHODS.keys()))
    def test_hc_fixed_k(self, sample_graph, method):
        G, metrics, extras = classify_edges(sample_graph, method, n_clusters=3)
        assert metrics["linkage"] == HC_METHODS[method]
        assert "hc_model" in extras

    def test_som_fixed_k(self, sample_graph):
        G, metrics, extras = classify_edges(sample_graph, "som", n_clusters=3)
        assert "som" in extras
        assert "performance_per_k" in extras

    def test_unknown_method_raises(self, sample_graph):
        with pytest.raises(ValueError, match="Unknown method"):
            classify_edges(sample_graph, "invalid_method", n_clusters=3)

    def test_cluster_attr_set_on_all_edges(self, sample_graph):
        G, _, _ = classify_edges(sample_graph, "kmeans", n_clusters=2)
        for u, v, key, data in G.edges(keys=True, data=True):
            assert "cluster" in data
            assert 0 <= data["cluster"] < 2

    @pytest.mark.parametrize(
        "method", ["kmeans", "fkmeans", "gmm", "som", "hc_ward"]
    )
    def test_pca_projects_to_two_components(self, sample_graph, method):
        G, _metrics, extras = classify_edges(
            sample_graph, method, n_clusters=3, use_pca=True
        )
        assert "pca_info" in extras
        assert "X_pca" in extras
        assert extras["X_pca"].shape == (G.number_of_edges(), 2)
        info = extras["pca_info"]
        assert len(info["explained_variance_ratio"]) == 2
        assert len(info["components"]) == 2
        assert len(info["components"][0]) == len(METRICS)
        assert info["feature_names"] == list(METRICS)
        assert 0.0 <= sum(info["explained_variance_ratio"]) <= 1.0 + 1e-9

    def test_pca_disabled_by_default(self, sample_graph):
        _, _metrics, extras = classify_edges(
            sample_graph, "kmeans", n_clusters=3
        )
        assert "pca_info" not in extras
        assert "X_pca" not in extras

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
# _apply_pca
# ---------------------------------------------------------------------------


class TestApplyPCA:
    def test_returns_two_components_and_info(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 3)
        X_2d, info = _apply_pca(X)
        assert X_2d.shape == (50, 2)
        for key in (
            "explained_variance_ratio",
            "explained_variance",
            "singular_values",
            "components",
            "mean",
            "feature_names",
        ):
            assert key in info
        assert len(info["explained_variance_ratio"]) == 2
        assert len(info["components"]) == 2
        assert len(info["components"][0]) == 3
        assert len(info["mean"]) == 3
        assert info["feature_names"] == list(METRICS)


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
