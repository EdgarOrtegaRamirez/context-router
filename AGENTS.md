# AGENTS.md — Notes for AI Agents Working on This Project

## Project Overview
Context Router is a Python CLI tool and library that analyzes prompts and routes them to the best AI model/provider.

## Architecture
- `src/context_router/` — Core library
  - `analyzer.py` — Prompt analysis (complexity, type detection, token estimation)
  - `router.py` — Routing engine (strategy selection, provider scoring)
  - `providers/` — Provider adapters (OpenAI, Anthropic, OpenRouter, local)
  - `config.py` — Configuration management
  - `models.py` — Pydantic data models
  - `cli.py` — Click CLI interface
- `tests/` — Test suite
- `config.yaml.example` — Example configuration

## Key Design Decisions
1. **Strategy pattern** for routing — new strategies are added by implementing `BaseRouterStrategy`
2. **Provider adapter pattern** — each AI provider gets its own adapter class
3. **Pydantic models** for all data structures — validation and serialization built-in
4. **Click CLI** for the command-line interface
5. **No hardcoded API keys** — all via environment variables

## Adding a New Provider
1. Create `src/context_router/providers/<name>.py`
2. Implement `BaseProviderAdapter` interface
3. Add to `PROVIDER_REGISTRY` in `providers/__init__.py`
4. Add tests in `tests/providers/test_<name>.py`

## Adding a New Routing Strategy
1. Create `src/context_router/strategies/<name>.py`
2. Implement `BaseRouterStrategy` interface
3. Register in `strategies/__init__.py`
4. Add tests in `tests/test_strategies.py`

## Testing
Run: `pytest tests/ -v`
Coverage: `pytest tests/ --cov=context_router --cov-report=term-missing`

## Common Patterns
- Use `logging` module, not `print()`
- Use type hints everywhere
- Validate inputs with Pydantic
- Handle errors gracefully with specific exceptions
- Use `tenacity` for retry logic on transient failures
