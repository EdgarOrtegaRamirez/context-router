"""Prompt analyzer — analyzes prompts for complexity, type, and other features."""

from __future__ import annotations

import logging
import re
from typing import Any

from context_router.models import PromptAnalysis

logger = logging.getLogger(__name__)

# Complexity indicators
COMPLEXITY_PATTERNS: dict[str, Any] = {
    "multi_step": {
        "pattern": r"(?:first|then|after|next|finally|step\s*\d+|also|and\s+then|subsequently)",
        "weight": 0.15,
    },
    "conditional": {
        "pattern": r"(?:if|otherwise|unless|in\s+case|depending|when\s+.*?then)",
        "weight": 0.12,
    },
    "comparison": {
        "pattern": r"(?:compare|versus|vs\.?|better\s+than|different\s+from|similar\s+to)",
        "weight": 0.1,
    },
    "code_related": {
        "pattern": r"(?:function|class|method|api|endpoint|database|query|algorithm|data\s*structure)",
        "weight": 0.2,
    },
    "math": {
        "pattern": r"(?:calculate|compute|solve|equation|integral|derivative|probability|statistic)",
        "weight": 0.15,
    },
    "creative": {
        "pattern": r"(?:write\s+(?:a\s+)?(?:story|poem|song|script|essay|article|blog|post|email|letter))",
        "weight": 0.05,
    },
    "long_context": {
        "pattern": r"(?:summarize|summarise|overview|brief|key\s*points|bullet\s*points)",
        "weight": 0.08,
    },
    "reasoning": {
        "pattern": r"(?:why|how\s+does|explain\s+why|reason|justify|prove|demonstrate)",
        "weight": 0.18,
    },
}

# Task type indicators
TASK_TYPE_PATTERNS: dict[str, Any] = {
    "coding": [
        r"(?:write\s+(?:a\s+)?(?:function|class|method|script|program|code|api|endpoint|service))",
        r"(?:debug|fix|refactor|optimize|implement|build|create\s+(?:a\s+)?(?:function|class|method))",
        r"(?:python|javascript|typescript|rust|go|java|c\+\+|sql)",
        r"(?:import|export|class|def |function |const |let |var )",
    ],
    "creative": [
        r"(?:write\s+(?:a\s+)?(?:story|poem|song|script|essay|article|blog|post|creative))",
        r"(?:brainstorm|idea|concept|creative|imagine|invent|design)",
    ],
    "reasoning": [
        r"(?:why|how\s+does|explain\s+why|reason|justify|prove|demonstrate|analyze)",
        r"(?:logical|deduct|induct|inference|conclusion|premise)",
    ],
    "factual": [
        r"(?:what\s+is|who\s+is|when\s+did|where\s+is|define|definition|means)",
        r"(?:fact|truth|reality|actual|real)",
    ],
    "summary": [
        r"(?:summarize|summarise|overview|brief|key\s*points|bullet\s*points|tl;dr)",
        r"(?:condense|compress|shorten|abridge)",
    ],
    "translation": [
        r"(?:translate|translation|convert\s+to\s+(?:english|spanish|french|german|chinese|japanese|korean))",
    ],
    "data_analysis": [
        r"(?:analyze|statistics|data\s*analysis|trend|pattern|correlation|regression)",
        r"(?:chart|graph|visualization|plot|table)",
    ],
}


class PromptAnalyzer:
    """Analyzes prompts to determine complexity, type, and other features."""

    def __init__(self) -> None:
        self._compiled_patterns: dict[str, re.Pattern] = {}
        for name, config in COMPLEXITY_PATTERNS.items():
            self._compiled_patterns[f"complexity_{name}"] = re.compile(
                config["pattern"], re.IGNORECASE
            )

    def analyze(self, prompt: str) -> PromptAnalysis:
        """Analyze a prompt and return analysis results.

        Args:
            prompt: The prompt text to analyze.

        Returns:
            PromptAnalysis with complexity, type, and other features.

        Raises:
            ValueError: If prompt is empty or too long.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if len(prompt) > 100_000:
            raise ValueError("Prompt exceeds maximum length of 100,000 characters")

        complexity = self._calculate_complexity(prompt)
        task_type = self._detect_task_type(prompt)
        estimated_tokens = self._estimate_tokens(prompt)
        requires_reasoning = self._check_reasoning(prompt)
        requires_creativity = self._check_creativity(prompt)
        urgency = self._assess_urgency(prompt)
        confidence = self._calculate_confidence(prompt, task_type)

        logger.debug(
            "Analyzed prompt: complexity=%.2f, type=%s, tokens=%d",
            complexity, task_type, estimated_tokens,
        )

        return PromptAnalysis(
            complexity=round(complexity, 2),
            task_type=task_type,
            estimated_tokens=estimated_tokens,
            requires_reasoning=requires_reasoning,
            requires_creativity=requires_creativity,
            urgency=urgency,
            confidence=round(confidence, 2),
        )

    def _calculate_complexity(self, prompt: str) -> float:
        """Calculate complexity score from 0.0 to 1.0."""
        score = 0.0
        prompt_lower = prompt.lower()

        for name, compiled in self._compiled_patterns.items():
            if compiled.search(prompt_lower):
                # Get the weight from COMPLEXITY_PATTERNS
                key = name.replace("complexity_", "")
                weight = COMPLEXITY_PATTERNS[key]["weight"]
                score += weight

        # Length factor — longer prompts tend to be more complex
        word_count = len(prompt.split())
        if word_count > 100:
            score += 0.15
        elif word_count > 50:
            score += 0.08
        elif word_count > 20:
            score += 0.03

        # Multiple questions increase complexity
        question_count = prompt.count("?")
        score += min(question_count * 0.05, 0.2)

        return min(score, 1.0)

    def _detect_task_type(self, prompt: str) -> str:
        """Detect the type of task from the prompt."""
        prompt_lower = prompt.lower()
        scores: dict[str, float] = {}

        for task_type, patterns in TASK_TYPE_PATTERNS.items():
            type_score = 0.0
            for pattern in patterns:
                if re.search(pattern, prompt_lower, re.IGNORECASE):
                    type_score += 1.0
            scores[task_type] = type_score

        if not scores or max(scores.values()) == 0:
            return "general"

        # Return the type with highest score
        return max(scores, key=scores.get)

    def _estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for the prompt.
        
        Approximate: ~4 chars per token for English text.
        """
        return max(1, len(prompt) // 4)

    def _check_reasoning(self, prompt: str) -> bool:
        """Check if the task requires deep reasoning."""
        reasoning_patterns = [
            r"(?:why|how\s+does|explain\s+why|reason|justify|prove|demonstrate|analyze|compare.*with|trade\s*off)",
            r"(?:complex|difficult|challenging|advanced|expert|sophisticated)",
            r"(?:algorithm|data\s*structure|architecture|design\s*pattern)",
        ]
        prompt_lower = prompt.lower()
        return any(re.search(p, prompt_lower) for p in reasoning_patterns)

    def _check_creativity(self, prompt: str) -> bool:
        """Check if the task requires creative generation."""
        creative_patterns = [
            r"(?:write\s+(?:a\s+)?(?:story|poem|song|script|essay|article|blog|post|creative))",
            r"(?:brainstorm|idea|concept|creative|imagine|invent|design|suggest)",
        ]
        prompt_lower = prompt.lower()
        return any(re.search(p, prompt_lower) for p in creative_patterns)

    def _assess_urgency(self, prompt: str) -> str:
        """Assess urgency level from the prompt."""
        prompt_lower = prompt.lower()
        urgent_patterns = [
            r"(?:asap|urgent|immediately|right\s*now|now|quickly|fast|quick)",
            r"(?:deadline|due|time\s*sensitive|need\s*this\s*now)",
        ]
        normal_patterns = [
            r"(?:when\s+.*?possible|at\s*your\s*convenience|no\s*rush|whenever)",
        ]

        if any(re.search(p, prompt_lower) for p in urgent_patterns):
            return "immediate"
        if any(re.search(p, prompt_lower) for p in normal_patterns):
            return "batch"
        return "normal"

    def _calculate_confidence(self, prompt: str, task_type: str) -> float:
        """Calculate confidence in the analysis."""
        # Higher confidence for longer, clearer prompts
        word_count = len(prompt.split())
        if word_count < 3:
            return 0.3
        if word_count < 10:
            return 0.5
        if word_count < 30:
            return 0.7
        if word_count < 100:
            return 0.85
        return 0.95
