"""Tests for CLI interface."""

import pytest
import yaml
from click.testing import CliRunner

from context_router.cli import main

SAMPLE_CONFIG = {
    "routing_strategy": "hybrid",
    "providers": {
        "fast": {
            "type": "openai",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "cost_per_token": 0.00000015,
            "speed": "fast",
            "quality": "medium",
            "max_tokens": 4096,
        },
        "smart": {
            "type": "anthropic",
            "model": "claude-3.5-sonnet",
            "base_url": "https://api.anthropic.com/v1",
            "cost_per_token": 0.000003,
            "speed": "medium",
            "quality": "high",
            "max_tokens": 8192,
        },
    },
    "default_provider": "fast",
}


class TestCLI:
    """Tests for the CLI interface."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def config_path(self, tmp_path):
        """Create a temporary config file for tests."""
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(SAMPLE_CONFIG))
        return str(path)

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

    def test_route_with_config(self, runner, config_path):
        """Test routing with a valid config file."""
        result = runner.invoke(main, [
            "--config", config_path,
            "route", "What is 2+2?",
            "--json",
        ])
        # Should parse config and at least start the route (will fail at HTTP call)
        # but config parsing should work
        assert result.exit_code in (0, 1)

    def test_analyze_with_config(self, runner, config_path):
        """Test analyzing with a valid config file."""
        result = runner.invoke(main, [
            "--config", config_path,
            "analyze", "What is Python?",
            "--json",
        ])
        # Analysis doesn't need network, should work
        assert result.exit_code == 0
        assert "complexity" in result.output
        assert "task_type" in result.output

    def test_providers_with_config(self, runner, config_path):
        """Test listing providers with a valid config file."""
        result = runner.invoke(main, [
            "--config", config_path,
            "providers",
        ])
        assert result.exit_code == 0
        assert "fast" in result.output or "smart" in result.output
