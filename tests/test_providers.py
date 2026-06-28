"""Tests for provider adapters."""

import asyncio

import pytest

from context_router.models import ProviderConfig, ProviderType
from context_router.providers import (
    PROVIDER_REGISTRY,
    AnthropicAdapter,
    BaseProviderAdapter,
    LocalAdapter,
    OpenAIAdapter,
    ProviderError,
    get_adapter,
)


class TestGetAdapter:
    """Tests for get_adapter function."""

    def test_get_openai_adapter(self):
        adapter = get_adapter(ProviderType.OPENAI)
        assert isinstance(adapter, OpenAIAdapter)

    def test_get_anthropic_adapter(self):
        adapter = get_adapter(ProviderType.ANTHROPIC)
        assert isinstance(adapter, AnthropicAdapter)

    def test_get_local_adapter(self):
        adapter = get_adapter(ProviderType.LOCAL)
        assert isinstance(adapter, LocalAdapter)

    def test_get_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            get_adapter("unsupported")  # type: ignore


class TestOpenAIAdapter:
    """Tests for OpenAI adapter."""

    def test_adapter_implements_base(self):
        adapter = OpenAIAdapter()
        assert isinstance(adapter, BaseProviderAdapter)

    def test_health_check_returns_false_for_bad_url(self):
        adapter = OpenAIAdapter()
        config = ProviderConfig(
            name="test",
            type=ProviderType.OPENAI,
            model="gpt-4",
            base_url="http://localhost:99999/v1",
        )
        # Should return False, not crash
        result = asyncio.run(adapter.health_check(config))
        assert result is False

    def test_send_request_to_bad_url_raises(self):
        adapter = OpenAIAdapter()
        config = ProviderConfig(
            name="test",
            type=ProviderType.OPENAI,
            model="gpt-4",
            base_url="http://localhost:19999/v1",
        )
        with pytest.raises((ProviderError, OSError, ConnectionError)):
            asyncio.run(
                adapter.send_request(
                    config,
                    [{"role": "user", "content": "Hello"}],
                )
            )


class TestAnthropicAdapter:
    """Tests for Anthropic adapter."""

    def test_adapter_implements_base(self):
        adapter = AnthropicAdapter()
        assert isinstance(adapter, BaseProviderAdapter)

    def test_health_check_returns_false_for_bad_url(self):
        adapter = AnthropicAdapter()
        config = ProviderConfig(
            name="test",
            type=ProviderType.ANTHROPIC,
            model="claude-3",
            base_url="http://localhost:99999/v1",
        )
        result = asyncio.run(adapter.health_check(config))
        assert result is False


class TestLocalAdapter:
    """Tests for Local adapter."""

    def test_adapter_implements_base(self):
        adapter = LocalAdapter()
        assert isinstance(adapter, BaseProviderAdapter)

    def test_health_check_returns_false_for_bad_url(self):
        adapter = LocalAdapter()
        config = ProviderConfig(
            name="local",
            type=ProviderType.LOCAL,
            model="llama-3",
            base_url="http://localhost:99999",
        )
        result = asyncio.run(adapter.health_check(config))
        assert result is False


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_core_types_registered(self):
        for ptype in (ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.LOCAL):
            assert ptype in PROVIDER_REGISTRY

    def test_openai_is_registered(self):
        assert ProviderType.OPENAI in PROVIDER_REGISTRY

    def test_anthropic_is_registered(self):
        assert ProviderType.ANTHROPIC in PROVIDER_REGISTRY

    def test_local_is_registered(self):
        assert ProviderType.LOCAL in PROVIDER_REGISTRY
