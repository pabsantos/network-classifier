"""Tests for network_classifier.export."""

from pathlib import Path

from network_classifier.export import export_txt


class TestExportTxt:
    def test_creates_file(self, tmp_path):
        filepath = tmp_path / "metrics.txt"
        export_txt(
            filepath,
            city="Test City",
            method="kmeans",
            n_clusters=3,
            model_metrics={
                "inertia": 123.456,
                "silhouette_score": 0.55,
                "n_iter": 10,
            },
        )
        assert filepath.exists()
        content = filepath.read_text()
        assert "Test City" in content
        assert "KMEANS" in content
        assert "3" in content
        assert "123.456" in content

    def test_handles_string_metric(self, tmp_path):
        filepath = tmp_path / "metrics.txt"
        export_txt(
            filepath,
            city="Test City",
            method="hc_al",
            n_clusters=4,
            model_metrics={
                "linkage": "average",
                "silhouette_score": 0.45,
                "n_leaves": 100,
            },
        )
        content = filepath.read_text()
        assert "average" in content
        assert "100" in content

    def test_int_keys_no_decimal(self, tmp_path):
        filepath = tmp_path / "metrics.txt"
        export_txt(
            filepath,
            city="Test",
            method="som",
            n_clusters=2,
            model_metrics={
                "grid_side": 10,
                "n_neurons": 100,
                "n_iter": 5,
            },
        )
        content = filepath.read_text()
        # int values should not have decimal formatting
        assert "10.0" not in content
        assert "100.0" not in content

    def test_pca_section_rendered(self, tmp_path):
        filepath = tmp_path / "metrics.txt"
        pca_info = {
            "explained_variance_ratio": [0.7, 0.2],
            "explained_variance": [1.4, 0.4],
            "singular_values": [3.5, 1.8],
            "components": [
                [0.6, 0.5, 0.6],
                [-0.7, 0.7, 0.1],
            ],
            "mean": [0.0, 0.0, 0.0],
            "feature_names": ["betweenness", "clustering", "degree"],
        }
        export_txt(
            filepath,
            city="Test",
            method="kmeans",
            n_clusters=3,
            model_metrics={"inertia": 1.0, "silhouette_score": 0.5},
            pca_info=pca_info,
        )
        content = filepath.read_text()
        assert "PCA Parameters" in content
        assert "Explained variance ratio (PC1)" in content
        assert "Explained variance ratio (PC2)" in content
        assert "Cumulative explained variance" in content
        assert "Loadings" in content
        assert "PC1" in content and "PC2" in content
        assert "betweenness" in content

    def test_no_pca_section_when_info_absent(self, tmp_path):
        filepath = tmp_path / "metrics.txt"
        export_txt(
            filepath,
            city="Test",
            method="kmeans",
            n_clusters=3,
            model_metrics={"inertia": 1.0},
        )
        content = filepath.read_text()
        assert "PCA Parameters" not in content
