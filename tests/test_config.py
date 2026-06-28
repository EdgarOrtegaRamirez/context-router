"""Tests for configuration management."""

from pathlib import Path

import pytest
import yaml

from context_router.config import RouterConfig, load_config, save_config
from context_router.models import ProviderConfig, ProviderType, Quality, RoutingStrategy, Speed


class TestRouterConfig:
    """Tests for RouterConfig model."""

    def test_default_config(self):
        config = RouterConfig()
        assert config.routing_strategy == RoutingStrategy.HYBRID
        assert config.providers == {}
        assert config.default_provider == "fast"

    def test_add_provider(self):
        config = RouterConfig()
        provider = ProviderConfig(
            name="test",
            type=ProviderType.OPENAI,
            model="gpt-4",
            base_url="https://api.openai.com/v1",
        )
        config.add_provider("test", provider)
        assert config.get_provider("test") == provider
        assert "test" in config.list_providers()

    def test_get_nonexistent_provider(self):
        config = RouterConfig()
        assert config.get_provider("nonexistent") is None

    def test_get_default_provider(self):
        config = RouterConfig()
        assert config.get_default_provider() is None

    def test_save_and_load(self, tmp_path: Path):
        config_path = tmp_path / "test_config.yaml"

        config = RouterConfig(
            routing_strategy=RoutingStrategy.COST,
            default_provider="fast",
            providers={
                "fast": ProviderConfig(
                    name="fast",
                    type=ProviderType.OPENAI,
                    model="gpt-4o-mini",
                    base_url="https://api.openai.com/v1",
                    cost_per_token=0.00000015,
                    speed=Speed.FAST,
                    quality=Quality.MEDIUM,
                ),
            },
        )

        save_config(config, config_path)

        # Verify file was created
        assert config_path.exists()

        # Load and verify
        loaded = load_config(config_path)
        assert loaded.routing_strategy == RoutingStrategy.COST
        assert "fast" in loaded.providers
        assert loaded.providers["fast"].model == "gpt-4o-mini"
        assert loaded.providers["fast"].speed == Speed.FAST


class TestLoadConfig:
    """Tests for config loading."""

    def test_missing_config_returns_defaults(self, tmp_path: Path):
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.routing_strategy == RoutingStrategy.HYBRID
        assert config.providers == {}

    def test_invalid_config_raises(self, tmp_path: Path):
        config_path = tmp_path / "bad_config.yaml"
        # Valid YAML but with bad provider data
        config_path.write_text("providers: 'not_a_mapping'\n")

        with pytest.raises(ValueError, match="providers"):
            load_config(config_path)

    def test_empty_providers(self, tmp_path: Path):
        config_path = tmp_path / "empty_providers.yaml"
        config_path.write_text("providers: {}\nrouting_strategy: cost\n")

        config = load_config(config_path)
        assert config.routing_strategy == RoutingStrategy.COST
        assert config.providers == {}

    def test_invalid_strategy_defaults_to_hybrid(self, tmp_path: Path):
        config_path = tmp_path / "bad_strategy.yaml"
        config_path.write_text("routing_strategy: nonexistent_strategy\n")

        config = load_config(config_path)
        assert config.routing_strategy == RoutingStrategy.HYBRID


class TestSaveConfig:
    """Tests for config saving."""

    def test_save_creates_file(self, tmp_path: Path):
        config_path = tmp_path / "output.yaml"
        config = RouterConfig()
        save_config(config, config_path)
        assert config_path.exists()

    def test_save_preserves_providers(self, tmp_path: Path):
        config_path = tmp_path / "output.yaml"
        config = RouterConfig(
            providers={
                "test": ProviderConfig(
                    name="test",
                    type=ProviderType.OPENAI,
                    model="gpt-4",
                    base_url="https://api.openai.com/v1",
                ),
            },
        )
        save_config(config, config_path)

        # Verify the saved file
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert "test" in data["providers"]
        assert data["providers"]["test"]["model"] == "gpt-4"
