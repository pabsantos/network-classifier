"""Tests for network_classifier.plots."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering

from network_classifier.plots import (
    _build_linkage_matrix,
    plot_dendrogram,
    plot_pca_scatter,
    plot_performance,
)


class TestPlotPerformance:
    def test_creates_file(self, tmp_path):
        performance = {
            k: {
                "silhouette": 0.5 - 0.05 * k,
                "chi": 100.0 + 10 * k,
                "v_measure": 0.3 + 0.02 * k,
                "wcss": 200.0 / k,
            }
            for k in range(2, 6)
        }
        filepath = tmp_path / "performance.png"
        plot_performance(performance, selected_k=3, filepath=filepath)
        assert filepath.exists()
        assert filepath.stat().st_size > 0


class TestBuildLinkageMatrix:
    def test_shape(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 3)
        model = AgglomerativeClustering(
            n_clusters=3, linkage="ward", compute_distances=True
        )
        model.fit(X)
        Z = _build_linkage_matrix(model)
        assert Z.shape == (49, 4)  # n_samples - 1 rows, 4 columns
        # distances should be non-negative
        assert (Z[:, 2] >= 0).all()
        # counts should sum to n_samples at the last merge
        assert Z[-1, 3] == 50


class TestPlotPCAScatter:
    def test_creates_file(self, tmp_path):
        rng = np.random.RandomState(0)
        X_pca = rng.randn(60, 2)
        labels = rng.randint(0, 3, size=60)
        pca_info = {
            "explained_variance_ratio": [0.65, 0.25],
            "explained_variance": [1.3, 0.5],
            "singular_values": [3.2, 1.5],
            "components": [[0.6, 0.5, 0.6], [-0.7, 0.7, 0.1]],
            "mean": [0.0, 0.0, 0.0],
            "feature_names": ["betweenness", "clustering", "degree"],
        }
        filepath = tmp_path / "pca.png"
        plot_pca_scatter(X_pca, labels, pca_info, filepath)
        assert filepath.exists()
        assert filepath.stat().st_size > 0


class TestPlotDendrogram:
    def test_creates_file(self, tmp_path):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 3)
        model = AgglomerativeClustering(
            n_clusters=3, linkage="ward", compute_distances=True
        )
        model.fit(X)
        filepath = tmp_path / "dendro.png"
        plot_dendrogram(model, n_clusters=3, filepath=filepath)
        assert filepath.exists()
        assert filepath.stat().st_size > 0
