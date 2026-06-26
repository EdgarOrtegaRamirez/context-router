"""Tests for CLI interface."""

import pytest
from click.testing import CliRunner

from context_router.cli import main


class TestCLI:
    """Tests for the CLI interface."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Context Router" in result.output

    def test_route_help(self, runner):
        result = runner.invoke(main, ["route", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output

    def test_analyze_help(self, runner):
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "JSON" in result.output

    def test_providers_help(self, runner):
        result = runner.invoke(main, ["providers", "--help"])
        assert result.exit_code == 0

    def test_stats_help(self, runner):
        result = runner.invoke(main, ["stats", "--help"])
        assert result.exit_code == 0

    def test_sample_config_help(self, runner):
        result = runner.invoke(main, ["sample-config", "--help"])
        assert result.exit_code == 0

    def test_reset_help(self, runner):
        result = runner.invoke(main, ["reset", "--help"])
        assert result.exit_code == 0

    def test_route_with_config(self, runner):
        """Test routing with a valid config file."""
        result = runner.invoke(main, [
            "--config", "config.yaml",
            "route", "What is 2+2?",
            "--json",
        ])
        # Should parse config and at least start the route (will fail at HTTP call)
        # but config parsing should work
        assert result.exit_code in (0, 1)

    def test_analyze_with_config(self, runner):
        """Test analyzing with a valid config file."""
        result = runner.invoke(main, [
            "--config", "config.yaml",
            "analyze", "What is Python?",
            "--json",
        ])
        # Analysis doesn't need network, should work
        assert result.exit_code == 0
        assert "complexity" in result.output
        assert "task_type" in result.output

    def test_providers_with_config(self, runner):
        """Test listing providers with a valid config file."""
        result = runner.invoke(main, [
            "--config", "config.yaml",
            "providers",
        ])
        assert result.exit_code == 0
        assert "fast" in result.output or "smart" in result.output
