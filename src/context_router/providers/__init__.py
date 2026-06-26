"""Provider adapters for Context Router."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from context_router.models import ProviderConfig, ProviderType

logger = logging.getLogger(__name__)


class BaseProviderAdapter(ABC):
    """Base class for provider adapters."""

    @abstractmethod
    async def send_request(
        self,
        config: ProviderConfig,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Send a request to the provider and return the response text.

        Args:
            config: Provider configuration.
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The response text from the provider.

        Raises:
            ProviderError: If the request fails.
        """
        ...

    @abstractmethod
    async def health_check(self, config: ProviderConfig) -> bool:
        """Check if the provider is reachable.

        Args:
            config: Provider configuration.

        Returns:
            True if the provider is reachable.
        """
        ...


class ProviderError(Exception):
    """Error from a provider request."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenAIAdapter(BaseProviderAdapter):
    """Adapter for OpenAI-compatible APIs."""

    async def send_request(
        self,
        config: ProviderConfig,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Send request to OpenAI-compatible API."""
        url = f"{config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.headers.get('api_key', '')}",
        }
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        payload.update(kwargs)

        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"HTTP {e.response.status_code}: {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Unexpected response format: {e}") from e
        except httpx.ConnectError as e:
            raise ProviderError(f"Connection failed: {e}") from e

    async def health_check(self, config: ProviderConfig) -> bool:
        """Check OpenAI API health."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{config.base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {config.headers.get('api_key', '')}"},
                )
                return response.status_code == 200
        except Exception:
            return False


class AnthropicAdapter(BaseProviderAdapter):
    """Adapter for Anthropic Claude API."""

    async def send_request(
        self,
        config: ProviderConfig,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Send request to Anthropic API."""
        url = f"{config.base_url.rstrip('/')}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": config.headers.get("api_key", ""),
            "anthropic-version": "2023-06-01",
        }
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            role = "user" if msg["role"] != "assistant" else "assistant"
            anthropic_messages.append({
                "role": role,
                "content": msg["content"],
            })

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        payload.update(kwargs)

        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"]
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"HTTP {e.response.status_code}: {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Unexpected response format: {e}") from e
        except httpx.ConnectError as e:
            raise ProviderError(f"Connection failed: {e}") from e

    async def health_check(self, config: ProviderConfig) -> bool:
        """Check Anthropic API health."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{config.base_url.rstrip('/')}/models",
                    headers={
                        "x-api-key": config.headers.get("api_key", ""),
                        "anthropic-version": "2023-06-01",
                    },
                )
                return response.status_code == 200
        except Exception:
            return False


class LocalAdapter(BaseProviderAdapter):
    """Adapter for local/self-hosted providers (e.g., Ollama, LM Studio)."""

    async def send_request(
        self,
        config: ProviderConfig,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Send request to local provider."""
        url = f"{config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        payload.update(kwargs)

        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"HTTP {e.response.status_code}: {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Unexpected response format: {e}") from e
        except httpx.ConnectError as e:
            raise ProviderError(f"Connection to local provider failed: {e}") from e

    async def health_check(self, config: ProviderConfig) -> bool:
        """Check local provider health."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(config.base_url.rstrip("/"))
                return response.status_code in (200, 404)  # 404 is ok for some local servers
        except Exception:
            return False


# Provider adapter registry
PROVIDER_REGISTRY: dict[ProviderType, type[BaseProviderAdapter]] = {
    ProviderType.OPENAI: OpenAIAdapter,
    ProviderType.ANTHROPIC: AnthropicAdapter,
    ProviderType.LOCAL: LocalAdapter,
}


def get_adapter(provider_type: ProviderType) -> BaseProviderAdapter:
    """Get a provider adapter by type.

    Args:
        provider_type: The provider type enum.

    Returns:
        An adapter instance.

    Raises:
        ValueError: If the provider type is not supported.
    """
    adapter_cls = PROVIDER_REGISTRY.get(provider_type)
    if adapter_cls is None:
        raise ValueError(f"Unsupported provider type: {provider_type}")
    return adapter_cls()
