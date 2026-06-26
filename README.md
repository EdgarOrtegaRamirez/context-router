# Context Router — Smart AI Request Router

## What It Does

Context Router analyzes incoming prompts and routes them to the best AI model/provider based on:
- **Complexity** — Simple vs. complex reasoning tasks
- **Cost** — Budget-conscious routing for routine tasks
- **Capability** — Matching task requirements to model strengths
- **Latency** — Time-sensitive vs. batch processing

It provides a unified interface that automatically selects the optimal model, saving costs while maintaining quality.

## Quick Start

```bash
# Install
pip install context-router

# Configure
cp .env.example .env
# Edit .env with your API keys

# Route a prompt
context-router route "What is 2+2?"

# Analyze without routing
context-router analyze "Write a Python function to sort a list"

# List configured providers
context-router providers list

# Get routing stats
context-router stats
```

## Example: Python API

```python
from context_router import Router, PromptAnalyzer

# Create a router with your providers
router = Router(
    providers={
        "fast": {"model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},
        "smart": {"model": "claude-3.5-sonnet", "base_url": "https://api.anthropic.com/v1"},
        "cheap": {"model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},
    },
    config="config.yaml",
)

# Route a prompt to the best provider
result = router.route("Explain quantum computing in simple terms")
print(f"Routed to: {result.provider}")
print(f"Reason: {result.reason}")
print(f"Estimated cost: ${result.estimated_cost}")
print(f"Result: {result.response}")
```

## Routing Strategies

| Strategy | Use Case | Example |
|----------|----------|---------|
| `complexity` | Route by task complexity | Simple math → cheap model, reasoning → smart model |
| `cost` | Minimize cost while maintaining quality | Always pick cheapest that can handle the task |
| `capability` | Match task type to model strength | Code → code-specialized, creative → creative model |
| `hybrid` | Balance cost, quality, and speed | Default strategy |

## Provider Configuration

Providers are configured in `config.yaml`:

```yaml
providers:
  fast:
    type: openai
    model: gpt-4o-mini
    base_url: https://api.openai.com/v1
    cost_per_token: 0.00000015
    speed: fast
    quality: medium
  smart:
    type: anthropic
    model: claude-3.5-sonnet
    base_url: https://api.anthropic.com/v1
    cost_per_token: 0.000003
    speed: medium
    quality: high
  local:
    type: openai
    model: llama-3.1-8b
    base_url: http://localhost:8080/v1
    cost_per_token: 0.0
    speed: fast
    quality: low
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Prompt In   │────▶│  Analyzer    │────▶│  Router     │
│  (text)      │     │  (complexity, │     │  (strategy) │
│              │     │   type, etc.) │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │    Provider Selection  │
                                    │  (best match chosen)  │
                                    └───────────┬───────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │    Request Forward     │
                                    │  (with headers, etc.)  │
                                    └───────────┬───────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │    Response Return     │
                                    │  (with routing info)   │
                                    └───────────────────────┘
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | Anthropic API key |
| `ROUTING_STRATEGY` | No | Routing strategy (complexity, cost, capability, hybrid) |
| `CONFIG_PATH` | No | Path to config file (default: config.yaml) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

## Security

- No API keys stored in code — all via environment variables
- Input validation on all prompts
- Rate limiting support via provider config
- No network calls without explicit user action

## License

MIT — see LICENSE file.
