"""Pydantic data models for Context Router."""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ProviderType(str, enum.Enum):
    """Supported provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    LOCAL = "local"


class Speed(str, enum.Enum):
    """Provider speed categories."""
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"


class Quality(str, enum.Enum):
    """Provider quality categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RoutingStrategy(str, enum.Enum):
    """Available routing strategies."""
    COMPLEXITY = "complexity"
    COST = "cost"
    CAPABILITY = "capability"
    HYBRID = "hybrid"


class PromptAnalysis(BaseModel):
    """Result of analyzing a prompt."""
    complexity: float = Field(
        ge=0.0, le=1.0,
        description="Complexity score from 0.0 (trivial) to 1.0 (highly complex)"
    )
    task_type: str = Field(
        description="Detected task type: coding, creative, reasoning, factual, summary, etc."
    )
    estimated_tokens: int = Field(
        ge=0,
        description="Estimated token count for the prompt"
    )
    requires_reasoning: bool = Field(
        description="Whether the task requires deep reasoning"
    )
    requires_creativity: bool = Field(
        description="Whether the task requires creative generation"
    )
    urgency: str = Field(
        description="Urgency level: immediate, normal, batch"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the analysis"
    )


class ProviderConfig(BaseModel):
    """Configuration for an AI provider."""
    name: str
    type: ProviderType
    model: str
    base_url: str = Field(
        description="Base URL for the provider API"
    )
    cost_per_token: float = Field(
        ge=0.0,
        default=0.000001,
        description="Cost per token in USD (0.0 means free/local)",
    )
    speed: Speed = Field(default=Speed.MEDIUM)
    quality: Quality = Field(default=Quality.MEDIUM)
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: int = Field(default=60, ge=1, description="Request timeout in seconds")
    headers: dict[str, str] = Field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        """Check if this is a local/self-hosted provider."""
        return self.type == ProviderType.LOCAL

    @property
    def is_free(self) -> bool:
        """Check if this provider is free (explicitly zero cost)."""
        return self.cost_per_token == 0.0


class RouterResult(BaseModel):
    """Result of routing a prompt."""
    provider_name: str
    provider_type: str
    model: str
    reason: str
    estimated_cost: float
    estimated_tokens: int
    routing_strategy: str
    latency_estimate: str
    response: str | None = None
    error: str | None = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class RoutingStats(BaseModel):
    """Statistics about routing decisions."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    provider_counts: dict[str, int] = Field(default_factory=dict)
    strategy_counts: dict[str, int] = Field(default_factory=dict)
    average_cost_per_request: float = 0.0
    average_tokens_per_request: float = 0.0

    def add_request(self, result: RouterResult) -> None:
        """Record a routing decision."""
        self.total_requests += 1
        self.total_tokens += result.estimated_tokens
        self.total_cost += result.estimated_cost
        self.provider_counts[result.provider_name] = (
            self.provider_counts.get(result.provider_name, 0) + 1
        )
        self.strategy_counts[result.routing_strategy] = (
            self.strategy_counts.get(result.routing_strategy, 0) + 1
        )
        if self.total_requests > 0:
            self.average_cost_per_request = (
                self.total_cost / self.total_requests
            )
            self.average_tokens_per_request = (
                self.total_tokens / self.total_requests
            )
