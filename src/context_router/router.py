"""Router engine — coordinates analysis, strategy, and provider selection."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from context_router.analyzer import PromptAnalyzer
from context_router.config import RouterConfig, load_config
from context_router.models import (
    PromptAnalysis,
    ProviderConfig,
    ProviderType,
    RouterResult,
    RoutingStats,
)
from context_router.providers import ProviderError, get_adapter
from context_router.strategies import get_strategy

logger = logging.getLogger(__name__)


class Router:
    """Main router that coordinates prompt analysis and provider selection."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        """Initialize the router.

        Args:
            config: Router configuration. If None, loads from config.yaml.
        """
        self.config = config or load_config()
        self.analyzer = PromptAnalyzer()
        self.stats = RoutingStats()
        self._request_history: list[RouterResult] = []

        logger.info(
            "Router initialized with strategy: %s, providers: %s",
            self.config.routing_strategy.value,
            list(self.config.providers.keys()),
        )

    @property
    def strategy(self):
        """Get the current routing strategy."""
        return self.config.routing_strategy

    @property
    def providers(self) -> dict[str, ProviderConfig]:
        """Get configured providers."""
        return self.config.providers

    def analyze(self, prompt: str) -> PromptAnalysis:
        """Analyze a prompt.

        Args:
            prompt: The prompt text to analyze.

        Returns:
            PromptAnalysis with complexity, type, and other features.

        Raises:
            ValueError: If the prompt is invalid.
        """
        return self.analyzer.analyze(prompt)

    async def route(
        self,
        prompt: str,
        system_prompt: str | None = None,
        provider_name: str | None = None,
        **kwargs,
    ) -> RouterResult:
        """Route a prompt to the best provider.

        Args:
            prompt: The prompt text.
            system_prompt: Optional system prompt.
            provider_name: Optional specific provider to use (bypasses routing).
            **kwargs: Additional parameters passed to the provider.

        Returns:
            RouterResult with the response and routing metadata.

        Raises:
            ValueError: If no providers are configured.
            ProviderError: If the provider request fails.
        """
        # Analyze the prompt
        analysis = self.analyze(prompt)

        # If a specific provider is requested, use it directly
        if provider_name:
            provider = self.config.get_provider(provider_name)
            if provider is None:
                raise ValueError(
                    f"Provider '{provider_name}' not found. "
                    f"Available: {self.config.list_providers()}"
                )
            response = await self._send_to_provider(provider, prompt, system_prompt, **kwargs)
            result = RouterResult(
                provider_name=provider_name,
                provider_type=provider.type.value,
                model=provider.model,
                reason=f"Explicitly requested provider: {provider_name}",
                estimated_cost=provider.cost_per_token * analysis.estimated_tokens,
                estimated_tokens=analysis.estimated_tokens,
                routing_strategy="explicit",
                latency_estimate="unknown",
                response=response,
            )
        else:
            # Use the routing strategy
            strategy = get_strategy(self.config.routing_strategy)
            provider_name, provider, reason = strategy.select_provider(
                self.config.providers, analysis
            )
            response = await self._send_to_provider(
                provider, prompt, system_prompt, **kwargs
            )
            result = RouterResult(
                provider_name=provider_name,
                provider_type=provider.type.value,
                model=provider.model,
                reason=reason,
                estimated_cost=provider.cost_per_token * analysis.estimated_tokens,
                estimated_tokens=analysis.estimated_tokens,
                routing_strategy=self.config.routing_strategy.value,
                latency_estimate=provider.speed.value,
                response=response,
            )

        # Update stats
        self.stats.add_request(result)
        self._request_history.append(result)

        logger.info(
            "Routed to %s (%s) — cost: $%.6f, tokens: %d",
            result.provider_name,
            result.model,
            result.estimated_cost,
            result.estimated_tokens,
        )

        return result

    async def _send_to_provider(
        self,
        provider: ProviderConfig,
        prompt: str,
        system_prompt: str | None,
        **kwargs,
    ) -> str:
        """Send a request to a specific provider.

        Args:
            provider: The provider configuration.
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters.

        Returns:
            The provider's response text.

        Raises:
            ProviderError: If the request fails.
        """
        adapter = get_adapter(provider.type)

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await adapter.send_request(provider, messages, **kwargs)

    def get_stats(self) -> RoutingStats:
        """Get routing statistics."""
        return self.stats

    def list_providers(self) -> list[str]:
        """List all configured providers."""
        return self.config.list_providers()

    def get_provider_info(self, name: str) -> ProviderConfig | None:
        """Get info about a specific provider."""
        return self.config.get_provider(name)

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self.stats = RoutingStats()
        self._request_history.clear()
        logger.info("Routing stats reset")
