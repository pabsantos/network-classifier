"""Tests for network_classifier.cli (argument parsing)."""

import argparse
import sys

import pytest


class TestBboxParsing:
    """Verify the --bbox pre-processing handles negative coordinates."""

    def _parse(self, args: list[str]) -> argparse.Namespace:
        """Run the CLI parser on *args* without executing the pipeline."""
        from network_classifier import cli

        original_argv = sys.argv
        try:
            sys.argv = ["network-classifier"] + args
            parser = argparse.ArgumentParser()
            source = parser.add_mutually_exclusive_group(required=True)
            source.add_argument("city", nargs="?", default=None)
            source.add_argument("--bbox", type=str)
            parser.add_argument("-f", "--format", required=True, choices=["graphml", "gpkg"])
            parser.add_argument("-m", "--method", default=None)
            parser.add_argument("-k", "--n-clusters", type=int, default=None)

            argv = sys.argv[1:]
            for i, arg in enumerate(argv):
                if arg == "--bbox" and i + 1 < len(argv):
                    argv = argv[:i] + [f"--bbox={argv[i + 1]}"] + argv[i + 2:]
                    break
            return parser.parse_args(argv)
        finally:
            sys.argv = original_argv

    def test_negative_coords_parsed(self):
        ns = self._parse(["--bbox", "-46.6,-23.5,-46.6,-23.5", "-f", "gpkg"])
        assert ns.bbox == "-46.6,-23.5,-46.6,-23.5"

    def test_positive_coords_parsed(self):
        ns = self._parse(["--bbox", "10.0,20.0,11.0,21.0", "-f", "gpkg"])
        assert ns.bbox == "10.0,20.0,11.0,21.0"

    def test_city_parsed(self):
        ns = self._parse(["Curitiba, Brazil", "-f", "gpkg"])
        assert ns.city == "Curitiba, Brazil"
        assert ns.bbox is None
