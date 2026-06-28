"""Tests for Context Router models."""

import pytest
from pydantic import ValidationError

from context_router.models import (
    PromptAnalysis,
    ProviderConfig,
    ProviderType,
    RouterResult,
    RoutingStats,
)


class TestProviderConfig:
    """Tests for ProviderConfig model."""

    def test_basic_provider(self):
        config = ProviderConfig(
            name="test",
            type=ProviderType.OPENAI,
            model="gpt-4",
            base_url="https://api.openai.com/v1",
        )
        assert config.name == "test"
        assert config.type == ProviderType.OPENAI
        assert config.model == "gpt-4"
        assert config.is_local is False
        assert config.is_free is False

    def test_local_provider(self):
        config = ProviderConfig(
            name="local",
            type=ProviderType.LOCAL,
            model="llama-3",
            base_url="http://localhost:8080/v1",
        )
        assert config.is_local is True

    def test_free_provider(self):
        config = ProviderConfig(
            name="free",
            type=ProviderType.OPENAI,
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            cost_per_token=0.0,
        )
        assert config.is_free is True

    def test_invalid_cost(self):
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="test",
                type=ProviderType.OPENAI,
                model="gpt-4",
                base_url="https://api.openai.com/v1",
                cost_per_token=-1.0,
            )

    def test_invalid_temperature(self):
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="test",
                type=ProviderType.OPENAI,
                model="gpt-4",
                base_url="https://api.openai.com/v1",
                temperature=3.0,
            )

    def test_max_tokens_minimum(self):
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="test",
                type=ProviderType.OPENAI,
                model="gpt-4",
                base_url="https://api.openai.com/v1",
                max_tokens=0,
            )

    def test_custom_headers(self):
        config = ProviderConfig(
            name="test",
            type=ProviderType.OPENAI,
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            headers={"x-custom": "value"},
        )
        assert config.headers == {"x-custom": "value"}


class TestPromptAnalysis:
    """Tests for PromptAnalysis model."""

    def test_basic_analysis(self):
        analysis = PromptAnalysis(
            complexity=0.5,
            task_type="coding",
            estimated_tokens=100,
            requires_reasoning=True,
            requires_creativity=False,
            urgency="normal",
            confidence=0.85,
        )
        assert analysis.complexity == 0.5
        assert analysis.task_type == "coding"

    def test_complexity_bounds(self):
        with pytest.raises(ValidationError):
            PromptAnalysis(
                complexity=-0.1,
                task_type="general",
                estimated_tokens=10,
                requires_reasoning=False,
                requires_creativity=False,
                urgency="normal",
                confidence=0.5,
            )

    def test_zero_tokens(self):
        analysis = PromptAnalysis(
            complexity=0.0,
            task_type="general",
            estimated_tokens=0,
            requires_reasoning=False,
            requires_creativity=False,
            urgency="normal",
            confidence=0.5,
        )
        assert analysis.estimated_tokens == 0


class TestRoutingStats:
    """Tests for RoutingStats model."""

    def test_empty_stats(self):
        stats = RoutingStats()
        assert stats.total_requests == 0
        assert stats.total_cost == 0.0

    def test_add_request(self):
        stats = RoutingStats()
        result = RouterResult(
            provider_name="fast",
            provider_type="openai",
            model="gpt-4o-mini",
            reason="Test reason",
            estimated_cost=0.001,
            estimated_tokens=100,
            routing_strategy="hybrid",
            latency_estimate="fast",
        )
        stats.add_request(result)
        assert stats.total_requests == 1
        assert stats.total_tokens == 100
        assert stats.total_cost == 0.001
        assert stats.provider_counts["fast"] == 1
        assert stats.strategy_counts["hybrid"] == 1
        assert stats.average_cost_per_request == 0.001
        assert stats.average_tokens_per_request == 100.0

    def test_multiple_requests(self):
        stats = RoutingStats()
        for i in range(3):
            result = RouterResult(
                provider_name="fast" if i < 2 else "smart",
                provider_type="openai",
                model="gpt-4o-mini",
                reason="Test",
                estimated_cost=0.001 * (i + 1),
                estimated_tokens=100 * (i + 1),
                routing_strategy="hybrid",
                latency_estimate="fast",
            )
            stats.add_request(result)

        assert stats.total_requests == 3
        assert stats.provider_counts["fast"] == 2
        assert stats.provider_counts["smart"] == 1
        # Costs: 0.001, 0.002, 0.003 = 0.006 total, avg = 0.002
        assert stats.total_cost == 0.006
        assert abs(stats.average_cost_per_request - 0.002) < 0.0001


class TestRouterResult:
    """Tests for RouterResult model."""

    def test_basic_result(self):
        result = RouterResult(
            provider_name="fast",
            provider_type="openai",
            model="gpt-4o-mini",
            reason="Test",
            estimated_cost=0.001,
            estimated_tokens=100,
            routing_strategy="hybrid",
            latency_estimate="fast",
        )
        assert result.provider_name == "fast"
        assert result.timestamp is not None

    def test_result_with_error(self):
        result = RouterResult(
            provider_name="fast",
            provider_type="openai",
            model="gpt-4o-mini",
            reason="Test",
            estimated_cost=0.0,
            estimated_tokens=0,
            routing_strategy="hybrid",
            latency_estimate="unknown",
            error="Connection timeout",
        )
        assert result.error == "Connection timeout"
        assert result.response is None
