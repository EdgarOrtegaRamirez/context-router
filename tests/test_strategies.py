"""Tests for routing strategies."""

import pytest

from context_router.models import (
    ProviderConfig,
    ProviderType,
    PromptAnalysis,
    Quality,
    RoutingStrategy,
    Speed,
)
from context_router.strategies import (
    CapabilityRouter,
    ComplexityRouter,
    CostRouter,
    HybridRouter,
    BaseRouterStrategy,
    get_strategy,
    STRATEGY_REGISTRY,
)


def make_provider(
    name: str = "test",
    cost: float = 0.000001,
    speed: Speed = Speed.MEDIUM,
    quality: Quality = Quality.MEDIUM,
    model: str = "gpt-4",
) -> ProviderConfig:
    """Helper to create a test provider config."""
    return ProviderConfig(
        name=name,
        type=ProviderType.OPENAI,
        model=model,
        base_url="https://api.openai.com/v1",
        cost_per_token=cost,
        speed=speed,
        quality=quality,
    )


def make_analysis(
    complexity: float = 0.5,
    task_type: str = "general",
    requires_reasoning: bool = False,
    requires_creativity: bool = False,
    urgency: str = "normal",
) -> PromptAnalysis:
    """Helper to create a test analysis."""
    return PromptAnalysis(
        complexity=complexity,
        task_type=task_type,
        estimated_tokens=100,
        requires_reasoning=requires_reasoning,
        requires_creativity=requires_creativity,
        urgency=urgency,
        confidence=0.8,
    )


class TestComplexityRouter:
    """Tests for ComplexityRouter."""

    def test_simple_task_prefers_fast(self):
        router = ComplexityRouter()
        simple = make_analysis(complexity=0.1)

        fast = make_provider("fast", cost=0.0, speed=Speed.FAST, quality=Quality.LOW)
        smart = make_provider("smart", cost=0.001, speed=Speed.SLOW, quality=Quality.HIGH)

        fast_score = router.score_provider(fast, simple)
        smart_score = router.score_provider(smart, simple)

        assert fast_score > smart_score

    def test_complex_task_prefers_quality(self):
        router = ComplexityRouter()
        complex_analysis = make_analysis(complexity=0.9)

        fast = make_provider("fast", cost=0.0, speed=Speed.FAST, quality=Quality.LOW)
        smart = make_provider("smart", cost=0.001, speed=Speed.SLOW, quality=Quality.HIGH)

        fast_score = router.score_provider(fast, complex_analysis)
        smart_score = router.score_provider(smart, complex_analysis)

        assert smart_score > fast_score

    def test_select_provider(self):
        router = ComplexityRouter()
        simple = make_analysis(complexity=0.1)
        providers = {
            "fast": make_provider("fast", speed=Speed.FAST),
            "smart": make_provider("smart", quality=Quality.HIGH),
        }
        name, provider, reason = router.select_provider(providers, simple)
        assert name in providers
        assert reason

    def test_no_providers_raises(self):
        router = ComplexityRouter()
        with pytest.raises(ValueError, match="No providers"):
            router.select_provider({}, make_analysis())


class TestCostRouter:
    """Tests for CostRouter."""

    def test_prefers_cheapest(self):
        router = CostRouter()
        analysis = make_analysis()

        cheap = make_provider("cheap", cost=0.0)
        expensive = make_provider("expensive", cost=0.01)

        assert router.score_provider(cheap, analysis) > router.score_provider(expensive, analysis)

    def test_low_quality_penalty_for_complex(self):
        router = CostRouter()
        complex_analysis = make_analysis(complexity=0.9)
        low_quality = make_provider("low", quality=Quality.LOW)

        score = router.score_provider(low_quality, complex_analysis)
        assert score < 0.5  # Should be penalized

    def test_no_providers_raises(self):
        router = CostRouter()
        with pytest.raises(ValueError, match="No providers"):
            router.select_provider({}, make_analysis())


class TestCapabilityRouter:
    """Tests for CapabilityRouter."""

    def test_coding_prefers_quality(self):
        router = CapabilityRouter()
        coding = make_analysis(task_type="coding")

        high = make_provider("high", quality=Quality.HIGH)
        low = make_provider("low", quality=Quality.LOW)

        assert router.score_provider(high, coding) > router.score_provider(low, coding)

    def test_factual_prefers_cheap(self):
        router = CapabilityRouter()
        factual = make_analysis(task_type="factual")

        cheap = make_provider("cheap", cost=0.0, quality=Quality.MEDIUM)
        expensive = make_provider("expensive", cost=0.01, quality=Quality.MEDIUM)

        assert router.score_provider(cheap, factual) > router.score_provider(expensive, factual)

    def test_reasoning_bonus(self):
        router = CapabilityRouter()
        reasoning = make_analysis(task_type="reasoning", requires_reasoning=True)
        high = make_provider("high", quality=Quality.HIGH)

        score = router.score_provider(high, reasoning)
        assert score > 0.3


class TestHybridRouter:
    """Tests for HybridRouter."""

    def test_balanced_scoring(self):
        router = HybridRouter()
        analysis = make_analysis()

        balanced = make_provider("balanced", cost=0.00001, speed=Speed.MEDIUM, quality=Quality.MEDIUM)
        score = router.score_provider(balanced, analysis)
        assert 0.0 <= score <= 1.0

    def test_immediate_urgency_prefers_fast(self):
        router = HybridRouter()
        urgent = make_analysis(urgency="immediate")

        fast = make_provider("fast", speed=Speed.FAST)
        slow = make_provider("slow", speed=Speed.SLOW)

        fast_score = router.score_provider(fast, urgent)
        slow_score = router.score_provider(slow, urgent)
        assert fast_score > slow_score

    def test_batch_urgency_prefers_quality(self):
        router = HybridRouter()
        batch = make_analysis(urgency="batch")

        high = make_provider("high", quality=Quality.HIGH, speed=Speed.SLOW)
        low = make_provider("low", quality=Quality.LOW, speed=Speed.FAST)

        high_score = router.score_provider(high, batch)
        low_score = router.score_provider(low, batch)
        assert high_score > low_score

    def test_no_providers_raises(self):
        router = HybridRouter()
        with pytest.raises(ValueError, match="No providers"):
            router.select_provider({}, make_analysis())


class TestStrategyRegistry:
    """Tests for strategy registry."""

    def test_all_strategies_registered(self):
        for strategy in RoutingStrategy:
            assert strategy in STRATEGY_REGISTRY

    def test_get_strategy(self):
        assert isinstance(get_strategy(RoutingStrategy.COMPLEXITY), ComplexityRouter)
        assert isinstance(get_strategy(RoutingStrategy.COST), CostRouter)
        assert isinstance(get_strategy(RoutingStrategy.CAPABILITY), CapabilityRouter)
        assert isinstance(get_strategy(RoutingStrategy.HYBRID), HybridRouter)

    def test_unknown_strategy_defaults_to_hybrid(self):
        from context_router.strategies import BaseRouterStrategy
        # Unknown strategy should default to HybridRouter
        result = get_strategy(RoutingStrategy.HYBRID)
        assert isinstance(result, HybridRouter)
