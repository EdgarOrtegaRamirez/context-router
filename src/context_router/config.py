"""Configuration management for Context Router."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from context_router.models import ProviderConfig, ProviderType, RoutingStrategy

logger = logging.getLogger(__name__)


class RouterConfig(BaseModel):
    """Full router configuration."""
    routing_strategy: RoutingStrategy = Field(
        default=RoutingStrategy.HYBRID
    )
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    default_provider: str = Field(default="fast")
    rate_limit: dict[str, int] = Field(
        default_factory=lambda: {"per_provider": 60, "per_minute": 60}
    )
    log_level: str = Field(default="INFO")

    def get_provider(self, name: str) -> ProviderConfig | None:
        """Get a provider by name."""
        return self.providers.get(name)

    def get_default_provider(self) -> ProviderConfig | None:
        """Get the default provider."""
        return self.providers.get(self.default_provider)

    def list_providers(self) -> list[str]:
        """List all configured provider names."""
        return list(self.providers.keys())

    def add_provider(self, name: str, config: ProviderConfig) -> None:
        """Add or update a provider configuration."""
        self.providers[name] = config
        logger.info("Added provider: %s (model: %s)", name, config.model)


def load_config(config_path: str | Path | None = None) -> RouterConfig:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the config file. Defaults to config.yaml in project root.

    Returns:
        RouterConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    if config_path is None:
        config_path = Path("config.yaml")

    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found at %s, using defaults", path)
        return RouterConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config file: expected mapping, got {type(raw).__name__}")

    # Parse providers
    providers: dict[str, ProviderConfig] = {}
    raw_providers = raw.get("providers", {})
    if not isinstance(raw_providers, dict):
        raise ValueError("Config 'providers' must be a mapping")

    for name, provider_data in raw_providers.items():
        if not isinstance(provider_data, dict):
            logger.warning("Skipping invalid provider config for '%s'", name)
            continue
        try:
            # Add the name (it's the YAML key, not in the value)
            provider_data_copy = dict(provider_data)
            provider_data_copy["name"] = name
            # Convert string enum values
            provider_data_copy["type"] = ProviderType(provider_data_copy["type"])
            if "speed" in provider_data_copy:
                provider_data_copy["speed"] = ProviderSpeedMap[provider_data_copy["speed"]]
            if "quality" in provider_data_copy:
                provider_data_copy["quality"] = ProviderQualityMap[provider_data_copy["quality"]]
            providers[name] = ProviderConfig.model_validate(provider_data_copy)
        except (ValueError, KeyError) as e:
            logger.warning("Failed to parse provider '%s': %s", name, e)

    # Parse routing strategy
    strategy_str = raw.get("routing_strategy", "hybrid")
    try:
        strategy = RoutingStrategy(strategy_str)
    except ValueError:
        logger.warning("Unknown strategy '%s', defaulting to hybrid", strategy_str)
        strategy = RoutingStrategy.HYBRID

    # Parse rate limit
    rate_limit = raw.get("rate_limit", {})
    if not isinstance(rate_limit, dict):
        rate_limit = {}

    return RouterConfig(
        routing_strategy=strategy,
        providers=providers,
        default_provider=raw.get("default_provider", "fast"),
        rate_limit=rate_limit,
        log_level=raw.get("log_level", "INFO"),
    )


# String-to-enum mappings for YAML parsing
ProviderSpeedMap = {
    "fast": "fast",
    "medium": "medium",
    "slow": "slow",
}

ProviderQualityMap = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}


def save_config(config: RouterConfig, config_path: str | Path) -> None:
    """Save configuration to a YAML file."""
    path = Path(config_path)

    # Convert enums to strings for serialization
    data: dict[str, Any] = {
        "routing_strategy": config.routing_strategy.value,
        "default_provider": config.default_provider,
        "rate_limit": config.rate_limit,
        "log_level": config.log_level,
        "providers": {},
    }

    for name, provider in config.providers.items():
        data["providers"][name] = {
            "type": provider.type.value,
            "model": provider.model,
            "base_url": provider.base_url,
            "cost_per_token": provider.cost_per_token,
            "speed": provider.speed.value,
            "quality": provider.quality.value,
            "max_tokens": provider.max_tokens,
            "temperature": provider.temperature,
            "timeout": provider.timeout,
        }
        if provider.headers:
            data["providers"][name]["headers"] = provider.headers

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info("Config saved to %s", path)
