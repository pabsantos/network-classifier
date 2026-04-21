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
