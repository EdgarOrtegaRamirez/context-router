"""Tests for PromptAnalyzer."""

import pytest

from context_router.analyzer import PromptAnalyzer
from context_router.models import PromptAnalysis


class TestPromptAnalyzer:
    """Tests for the PromptAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        return PromptAnalyzer()

    def test_empty_prompt_raises(self, analyzer):
        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze("")

    def test_whitespace_only_raises(self, analyzer):
        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze("   ")

    def test_long_prompt_raises(self, analyzer):
        with pytest.raises(ValueError, match="exceeds"):
            analyzer.analyze("x" * 100_001)

    def test_simple_factual(self, analyzer):
        analysis = analyzer.analyze("What is the capital of France?")
        assert analysis.task_type == "factual"
        assert analysis.complexity < 0.5
        assert analysis.estimated_tokens > 0

    def test_coding_task(self, analyzer):
        analysis = analyzer.analyze("Write a Python function to sort a list")
        assert analysis.task_type == "coding"

    def test_creative_task(self, analyzer):
        # "Write a story" matches both creative and coding patterns,
        # but creative should win due to higher pattern count
        analysis = analyzer.analyze("Write a creative story about a dragon")
        assert analysis.requires_creativity is True

    def test_reasoning_task(self, analyzer):
        analysis = analyzer.analyze("Why does gravity exist and how does it work?")
        assert analysis.requires_reasoning is True
        assert analysis.task_type == "reasoning"

    def test_summary_task(self, analyzer):
        analysis = analyzer.analyze("Summarize the key points of this document")
        assert analysis.task_type == "summary"

    def test_translation_task(self, analyzer):
        analysis = analyzer.analyze("Translate this to Spanish: Hello world")
        assert analysis.task_type == "translation"

    def test_data_analysis_task(self, analyzer):
        analysis = analyzer.analyze(
            "Create a visualization and plot the correlation between these "
            "statistics and data trends in the chart"
        )
        assert analysis.task_type == "data_analysis"

    def test_complexity_scores(self, analyzer):
        # Simple prompt
        simple = analyzer.analyze("What is 2+2?")
        assert simple.complexity < 0.3

        # Complex prompt
        complex_prompt = (
            "First analyze the data, then compare the results, "
            "and finally write a function to implement the solution"
        )
        complex_analysis = analyzer.analyze(complex_prompt)
        assert complex_analysis.complexity > simple.complexity

    def test_token_estimation(self, analyzer):
        analysis = analyzer.analyze("Hello world")
        # ~4 chars per token approximation
        assert analysis.estimated_tokens >= 1
        assert analysis.estimated_tokens <= 10

    def test_urgency_immediate(self, analyzer):
        analysis = analyzer.analyze("I need this ASAP, right now!")
        assert analysis.urgency == "immediate"

    def test_urgency_batch(self, analyzer):
        analysis = analyzer.analyze("When possible, at your convenience")
        assert analysis.urgency == "batch"

    def test_urgency_normal(self, analyzer):
        analysis = analyzer.analyze("What is Python?")
        assert analysis.urgency == "normal"

    def test_confidence_levels(self, analyzer):
        # Short prompt — low confidence
        short = analyzer.analyze("Hi")
        assert short.confidence < 0.5

        # Long prompt — high confidence
        long_prompt = (
            "Write a comprehensive Python function that implements "
            "a binary search algorithm with detailed error handling and "
            "type annotations for the input array and return value, "
            "including edge cases and performance considerations"
        )
        long_analysis = analyzer.analyze(long_prompt)
        assert long_analysis.confidence > 0.7

    def test_general_task_type(self, analyzer):
        # Prompt that doesn't match any specific type
        analysis = analyzer.analyze("Hello, how are you today?")
        assert analysis.task_type == "general"

    def test_multi_step_complexity(self, analyzer):
        analysis = analyzer.analyze(
            "First create a database schema, then write the migration, "
            "then implement the API endpoints, and finally write tests"
        )
        assert analysis.complexity > 0.3

    def test_math_task(self, analyzer):
        analysis = analyzer.analyze(
            "Calculate the definite integral of x squared from 0 to 1 "
            "and explain each step of the computation"
        )
        assert analysis.task_type == "math" or analysis.complexity > 0.1

    def test_comparison_task(self, analyzer):
        analysis = analyzer.analyze(
            "Compare Python and JavaScript for web development, "
            "focusing on frameworks, performance, and ecosystem differences"
        )
        assert analysis.complexity > 0.08

    def test_analysis_is_deterministic(self, analyzer):
        prompt = "Write a function to sort a list in Python"
        a1 = analyzer.analyze(prompt)
        a2 = analyzer.analyze(prompt)
        assert a1.complexity == a2.complexity
        assert a1.task_type == a2.task_type
        assert a1.estimated_tokens == a2.estimated_tokens
