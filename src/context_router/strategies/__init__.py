"""Base classes and strategy implementations for routing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from context_router.analyzer import PromptAnalyzer
from context_router.config import RouterConfig
from context_router.models import (
    PromptAnalysis,
    ProviderConfig,
    ProviderType,
    Quality,
    RouterResult,
    RoutingStrategy,
    Speed,
)

logger = logging.getLogger(__name__)


class BaseRouterStrategy(ABC):
    """Base class for routing strategies."""

    @abstractmethod
    def score_provider(
        self,
        provider: ProviderConfig,
        analysis: PromptAnalysis,
    ) -> float:
        """Score a provider for a given analysis.

        Higher score = better match.

        Args:
            provider: The provider to score.
            analysis: The prompt analysis.

        Returns:
            Score from 0.0 to 1.0.
        """
        ...

    def select_provider(
        self,
        providers: dict[str, ProviderConfig],
        analysis: PromptAnalysis,
    ) -> tuple[str, ProviderConfig, str]:
        """Select the best provider for the analysis.

        Args:
            providers: Available providers.
            analysis: The prompt analysis.

        Returns:
            Tuple of (provider_name, provider_config, reason).

        Raises:
            ValueError: If no providers are available.
        """
        if not providers:
            raise ValueError("No providers configured")

        scores: dict[str, float] = {}
        for name, provider in providers.items():
            scores[name] = self.score_provider(provider, analysis)

        if not scores:
            raise ValueError("No providers scored successfully")

        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]
        best_provider = providers[best_name]

        reason = self._generate_reason(best_name, best_score, analysis)
        logger.info(
            "Selected provider '%s' (score: %.2f) for %s task",
            best_name, best_score, analysis.task_type,
        )

        return best_name, best_provider, reason

    def _generate_reason(
        self,
        name: str,
        score: float,
        analysis: PromptAnalysis,
    ) -> str:
        """Generate a human-readable reason for the selection."""
        return (
            f"Scored {score:.2f} for {analysis.task_type} task "
            f"(complexity: {analysis.complexity:.2f})"
        )


class ComplexityRouter(BaseRouterStrategy):
    """Route based on task complexity.

    Simple tasks → cheaper/faster providers.
    Complex tasks → higher-quality providers.
    """

    def score_provider(
        self,
        provider: ProviderConfig,
        analysis: PromptAnalysis,
    ) -> float:
        """Score based on matching provider quality to task complexity."""
        score = 0.0

        # Match quality to complexity
        if analysis.complexity < 0.3:
            # Simple task — prefer speed and cost
            if provider.speed == Speed.FAST:
                score += 0.4
            if provider.cost_per_token < 0.000001:
                score += 0.3
            if provider.quality == Quality.MEDIUM:
                score += 0.2
        elif analysis.complexity < 0.7:
            # Medium complexity — balanced
            if provider.quality == Quality.MEDIUM:
                score += 0.3
            if provider.speed == Speed.MEDIUM:
                score += 0.2
            if provider.cost_per_token < 0.00001:
                score += 0.2
        else:
            # Complex task — prefer quality
            if provider.quality == Quality.HIGH:
                score += 0.5
            if provider.quality == Quality.MEDIUM:
                score += 0.2
            if provider.speed == Speed.SLOW:
                score += 0.1

        # Bonus for free/local providers on simple tasks
        if provider.is_free and analysis.complexity < 0.5:
            score += 0.2

        return min(score, 1.0)


class CostRouter(BaseRouterStrategy):
    """Route to minimize cost while maintaining adequate quality."""

    def score_provider(
        self,
        provider: ProviderConfig,
        analysis: PromptAnalysis,
    ) -> float:
        """Score based on cost efficiency."""
        score = 0.0

        # Lower cost = higher score
        if provider.cost_per_token == 0.0:
            score += 0.5
        elif provider.cost_per_token < 0.000001:
            score += 0.4
        elif provider.cost_per_token < 0.00001:
            score += 0.3
        elif provider.cost_per_token < 0.0001:
            score += 0.2
        else:
            score += 0.05

        # Ensure minimum quality for complex tasks
        if analysis.complexity > 0.7 and provider.quality == Quality.LOW:
            score -= 0.3
        elif analysis.complexity > 0.5 and provider.quality == Quality.LOW:
            score -= 0.15

        return max(score, 0.0)


class CapabilityRouter(BaseRouterStrategy):
    """Route based on matching task type to provider capabilities."""

    def score_provider(
        self,
        provider: ProviderConfig,
        analysis: PromptAnalysis,
    ) -> float:
        """Score based on capability match."""
        score = 0.0

        # Task-type matching
        task_type = analysis.task_type
        if task_type in ("coding",):
            # Coding tasks benefit from high quality
            if provider.quality == Quality.HIGH:
                score += 0.4
            elif provider.quality == Quality.MEDIUM:
                score += 0.2
        elif task_type in ("creative",):
            # Creative tasks need quality
            if provider.quality == Quality.HIGH:
                score += 0.4
            elif provider.quality == Quality.MEDIUM:
                score += 0.25
        elif task_type in ("factual",):
            # Factual tasks can use cheaper providers
            if provider.cost_per_token < 0.000001:
                score += 0.4
            elif provider.quality == Quality.MEDIUM:
                score += 0.3
        elif task_type in ("summary",):
            # Summary tasks are fast
            if provider.speed == Speed.FAST:
                score += 0.4
            elif provider.quality == Quality.MEDIUM:
                score += 0.2
        else:
            # General — balanced
            if provider.quality == Quality.MEDIUM:
                score += 0.3
            elif provider.quality == Quality.HIGH:
                score += 0.25

        # Reasoning bonus
        if analysis.requires_reasoning and provider.quality in (Quality.HIGH, Quality.MEDIUM):
            score += 0.15

        # Creativity bonus
        if analysis.requires_creativity and provider.quality in (Quality.HIGH, Quality.MEDIUM):
            score += 0.1

        return min(score, 1.0)


class HybridRouter(BaseRouterStrategy):
    """Balance cost, quality, and speed for optimal routing."""

    def score_provider(
        self,
        provider: ProviderConfig,
        analysis: PromptAnalysis,
    ) -> float:
        """Score based on a weighted combination of factors."""
        score = 0.0

        # Quality match (40% weight)
        quality_score = 0.0
        if analysis.complexity > 0.6:
            if provider.quality == Quality.HIGH:
                quality_score = 1.0
            elif provider.quality == Quality.MEDIUM:
                quality_score = 0.6
            else:
                quality_score = 0.2
        else:
            if provider.quality in (Quality.MEDIUM, Quality.HIGH):
                quality_score = 0.8
            else:
                quality_score = 0.5
        score += quality_score * 0.4

        # Cost efficiency (30% weight)
        cost_score = 0.0
        if provider.cost_per_token == 0.0:
            cost_score = 1.0
        elif provider.cost_per_token < 0.000001:
            cost_score = 0.9
        elif provider.cost_per_token < 0.00001:
            cost_score = 0.7
        elif provider.cost_per_token < 0.0001:
            cost_score = 0.4
        else:
            cost_score = 0.1
        score += cost_score * 0.3

        # Speed match (20% weight)
        speed_score = 0.0
        if analysis.urgency == "immediate":
            if provider.speed == Speed.FAST:
                speed_score = 1.0
            elif provider.speed == Speed.MEDIUM:
                speed_score = 0.6
            else:
                speed_score = 0.2
        elif analysis.urgency == "batch":
            if provider.quality == Quality.HIGH:
                speed_score = 0.8
            else:
                speed_score = 0.6
        else:
            speed_score = 0.5
        score += speed_score * 0.2

        # Capability match (10% weight)
        if analysis.task_type in ("coding", "reasoning") and provider.quality in (
            Quality.HIGH,
            Quality.MEDIUM,
        ):
            score += 0.1

        return min(score, 1.0)


# Strategy registry
STRATEGY_REGISTRY: dict[RoutingStrategy, type[BaseRouterStrategy]] = {
    RoutingStrategy.COMPLEXITY: ComplexityRouter,
    RoutingStrategy.COST: CostRouter,
    RoutingStrategy.CAPABILITY: CapabilityRouter,
    RoutingStrategy.HYBRID: HybridRouter,
}


def get_strategy(strategy: RoutingStrategy) -> BaseRouterStrategy:
    """Get a routing strategy by name."""
    cls = STRATEGY_REGISTRY.get(strategy)
    if cls is None:
        logger.warning("Unknown strategy '%s', using hybrid", strategy)
        return HybridRouter()
    return cls()
