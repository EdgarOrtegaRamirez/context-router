"""Tests for Router engine."""

import pytest

from context_router.config import RouterConfig
from context_router.models import ProviderConfig, ProviderType, RoutingStrategy
from context_router.router import Router


class TestRouter:
    """Tests for the Router class."""

    @pytest.fixture
    def config(self):
        return RouterConfig(
            routing_strategy=RoutingStrategy.HYBRID,
            providers={
                "fast": ProviderConfig(
                    name="fast",
                    type=ProviderType.OPENAI,
                    model="gpt-4o-mini",
                    base_url="https://api.openai.com/v1",
                    cost_per_token=0.00000015,
                    speed="fast",
                    quality="medium",
                ),
                "smart": ProviderConfig(
                    name="smart",
                    type=ProviderType.ANTHROPIC,
                    model="claude-3.5-sonnet",
                    base_url="https://api.anthropic.com/v1",
                    cost_per_token=0.000003,
                    speed="medium",
                    quality="high",
                ),
            },
        )

    def test_router_init(self, config):
        router = Router(config)
        assert router.strategy == RoutingStrategy.HYBRID
        assert len(router.providers) == 2

    def test_router_list_providers(self, config):
        router = Router(config)
        providers = router.list_providers()
        assert "fast" in providers
        assert "smart" in providers

    def test_router_get_provider_info(self, config):
        router = Router(config)
        info = router.get_provider_info("fast")
        assert info is not None
        assert info.model == "gpt-4o-mini"

    def test_router_get_nonexistent_provider(self, config):
        router = Router(config)
        info = router.get_provider_info("nonexistent")
        assert info is None

    def test_router_analyze_simple(self, config):
        router = Router(config)
        analysis = router.analyze("What is 2+2?")
        assert analysis.task_type == "factual"
        assert analysis.complexity < 0.5

    def test_router_analyze_coding(self, config):
        router = Router(config)
        analysis = router.analyze("Write a Python function to sort a list")
        assert analysis.task_type == "coding"

    def test_router_analyze_empty_raises(self, config):
        router = Router(config)
        with pytest.raises(ValueError, match="empty"):
            router.analyze("")

    def test_router_analyze_whitespace_raises(self, config):
        router = Router(config)
        with pytest.raises(ValueError, match="empty"):
            router.analyze("   ")

    def test_router_reset_stats(self, config):
        router = Router(config)
        router.reset_stats()
        stats = router.get_stats()
        assert stats.total_requests == 0

    def test_router_no_providers_raises_on_route(self):
        empty_config = RouterConfig()
        router = Router(empty_config)
        with pytest.raises(ValueError, match="No providers"):
            # This will fail at strategy.select_provider since no providers
            import asyncio
            asyncio.run(router.route("Hello"))

    def test_router_explicit_provider_not_found(self, config):
        router = Router(config)
        with pytest.raises(ValueError, match="not found"):
            import asyncio
            asyncio.run(router.route("Hello", provider_name="nonexistent"))
