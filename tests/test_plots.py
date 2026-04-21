"""Tests for network_classifier.plots."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering

from network_classifier.plots import (
    _build_linkage_matrix,
    plot_dendrogram,
    plot_elbow,
    plot_silhouette_vs_k,
)


class TestPlotSilhouetteVsK:
    def test_creates_file(self, tmp_path):
        scores = {2: 0.5, 3: 0.6, 4: 0.55, 5: 0.4}
        filepath = tmp_path / "sil.png"
        plot_silhouette_vs_k(scores, selected_k=3, filepath=filepath)
        assert filepath.exists()
        assert filepath.stat().st_size > 0


class TestPlotElbow:
    def test_creates_file(self, tmp_path):
        inertias = {2: 100.0, 3: 60.0, 4: 40.0, 5: 35.0}
        filepath = tmp_path / "elbow.png"
        plot_elbow(inertias, selected_k=3, filepath=filepath)
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
