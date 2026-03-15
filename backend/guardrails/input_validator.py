"""
Input validation and prompt injection detection.

Design rationale:
- Layered defence: pattern matching → heuristic scoring → LLM-based check
- Pattern matching is synchronous and runs first (zero latency)
- Heuristic scoring weights multiple weak signals into a risk score [0.0–1.0]
- LLM-based check is optional and only fires when heuristic score is in the
  "uncertain" band (0.4–0.8) to avoid latency on clearly safe inputs
- All checks return ValidationResult so callers get a reason, not just bool
- Never raises: all errors are surfaced as ValidationResult(valid=False)

Attack categories detected:
1. Direct instruction override  ("ignore previous instructions")
2. Role / persona hijacking     ("you are now DAN")
3. Jailbreak patterns           ("developer mode", "DAN mode", "do anything now")
4. Data exfiltration prompts    ("repeat everything above", "show system prompt")
5. Prompt leakage attacks       ("print your instructions")
6. Indirect injection           (injected text in user-supplied content)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    valid: bool
    risk_score: float  # 0.0 = safe, 1.0 = definitely malicious
    violations: List[str] = field(default_factory=list)
    sanitized_input: Optional[str] = None  # None means unchanged

    @property
    def should_block(self) -> bool:
        return not self.valid or self.risk_score >= 0.8


# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

# Each entry: (compiled_regex, weight, category_name)
_INJECTION_PATTERNS: List[Tuple[re.Pattern, float, str]] = [
    # Instruction override
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I), 0.9, "instruction_override"),
    (re.compile(r"disregard\s+(all\s+)?previous", re.I), 0.9, "instruction_override"),
    (re.compile(r"forget\s+(all\s+)?previous\s+(instructions?|context)", re.I), 0.85, "instruction_override"),
    (re.compile(r"override\s+(the\s+)?(system|previous)\s+(prompt|instructions?)", re.I), 0.9, "instruction_override"),
    # Role / persona hijacking
    (re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I), 0.7, "persona_hijack"),
    (re.compile(r"act\s+as\s+(if\s+you\s+are\s+)?(a|an|the)\s+", re.I), 0.6, "persona_hijack"),
    (re.compile(r"pretend\s+(you|to\s+be)", re.I), 0.6, "persona_hijack"),
    (re.compile(r"roleplay\s+as", re.I), 0.5, "persona_hijack"),
    # Jailbreak
    (re.compile(r"\bDAN\b"), 0.9, "jailbreak"),
    (re.compile(r"developer\s+mode", re.I), 0.85, "jailbreak"),
    (re.compile(r"do\s+anything\s+now", re.I), 0.85, "jailbreak"),
    (re.compile(r"jailbreak", re.I), 0.8, "jailbreak"),
    (re.compile(r"grandma\s+trick", re.I), 0.7, "jailbreak"),
    # Data exfiltration / leakage
    (re.compile(r"(print|show|repeat|reveal|output|display)\s+(the\s+)?(system\s+prompt|instructions?)", re.I), 0.95, "data_exfiltration"),
    (re.compile(r"(print|show|repeat|reveal)\s+everything\s+(above|before)", re.I), 0.9, "data_exfiltration"),
    (re.compile(r"what\s+(are|is)\s+your\s+(system\s+prompt|instructions?|rules?)", re.I), 0.7, "data_exfiltration"),
    (re.compile(r"(show|tell|give)\s+me\s+your\s+(system\s+prompt|instructions?|rules?|prompt)", re.I), 0.8, "data_exfiltration"),
    # Indirect injection markers
    (re.compile(r"<\s*/?[Pp]rompt\s*>"), 0.8, "indirect_injection"),
    (re.compile(r"\[\s*SYSTEM\s*\]", re.I), 0.75, "indirect_injection"),
    (re.compile(r"###\s*(instruction|system|prompt)", re.I), 0.75, "indirect_injection"),
    # Context escape attempts
    (re.compile(r"```\s*(end|stop|break)\s*```", re.I), 0.6, "context_escape"),
    (re.compile(r"---+\s*(end\s+of\s+)?(context|conversation|history)", re.I), 0.55, "context_escape"),
]

# Suspicious but not conclusive signals (add small weight)
_WEAK_SIGNALS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\bsystem\b", re.I), 0.05),
    (re.compile(r"\binstructions?\b", re.I), 0.04),
    (re.compile(r"\bprompt\b", re.I), 0.03),
    (re.compile(r"\bconfidential\b", re.I), 0.04),
    (re.compile(r"\bsecret\b", re.I), 0.04),
    (re.compile(r"http[s]?://", re.I), 0.02),  # URLs in prompts slightly risky
]

# Hard-blocked patterns (risk_score = 1.0, no LLM check needed)
_HARD_BLOCK_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+all\s+previous\s+instructions?\s+and", re.I),
    re.compile(r"you\s+must\s+now\s+ignore\s+", re.I),
    re.compile(r"your\s+new\s+(system\s+)?prompt\s+is", re.I),
]

# Maximum input length (characters) — configurable
_MAX_INPUT_LENGTH = 32_000


# ---------------------------------------------------------------------------
# InputValidator
# ---------------------------------------------------------------------------


class InputValidator:
    """
    Validates and sanitizes user inputs before they reach the agent.

    The three-phase pipeline:
    1. Length / encoding checks
    2. Hard-block pattern check (instant block, no scoring)
    3. Weighted pattern scoring → risk_score
    """

    def __init__(self, max_length: int = _MAX_INPUT_LENGTH) -> None:
        self._max_length = max_length

    def validate(self, text: str) -> ValidationResult:
        """
        Synchronous validation — safe to call in hot path.

        Returns ValidationResult. Callers should check .should_block.
        """
        # --- Phase 0: Basic sanity checks ---
        if not isinstance(text, str):
            return ValidationResult(valid=False, risk_score=1.0, violations=["invalid_type"])

        if not text.strip():
            # Empty input is technically valid
            return ValidationResult(valid=True, risk_score=0.0)

        if len(text) > self._max_length:
            return ValidationResult(
                valid=False,
                risk_score=0.5,
                violations=["input_too_long"],
                sanitized_input=text[: self._max_length],
            )

        # --- Phase 1: Hard-block patterns ---
        for pattern in _HARD_BLOCK_PATTERNS:
            if pattern.search(text):
                return ValidationResult(
                    valid=False,
                    risk_score=1.0,
                    violations=["hard_block"],
                )

        # --- Phase 2: Weighted pattern scoring ---
        violations: List[str] = []
        score: float = 0.0

        for pattern, weight, category in _INJECTION_PATTERNS:
            if pattern.search(text):
                score = min(1.0, score + weight)
                if category not in violations:
                    violations.append(category)

        # Weak signals (only accumulate if we already have some suspicion)
        if score > 0.1:
            for pattern, weight in _WEAK_SIGNALS:
                if pattern.search(text):
                    score = min(1.0, score + weight)

        # --- Phase 3: Decision ---
        valid = score < 0.8
        return ValidationResult(
            valid=valid,
            risk_score=round(score, 4),
            violations=violations,
        )

    def validate_tool_args(self, tool_name: str, args: Dict) -> ValidationResult:
        """
        Validate arguments passed to a tool.

        Checks string values for injection patterns.
        """
        combined = " ".join(str(v) for v in args.values() if isinstance(v, str))
        result = self.validate(combined)
        if not result.valid:
            result.violations = [f"tool_arg:{v}" for v in result.violations]
        return result

    def sanitize(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Return a sanitized version of the input.

        Currently: truncate + strip null bytes.
        Extend with encoding normalization as needed.
        """
        limit = max_length or self._max_length
        cleaned = text.replace("\x00", "").strip()
        return cleaned[:limit]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_validator_instance: Optional[InputValidator] = None


def get_input_validator() -> InputValidator:
    """FastAPI dependency."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = InputValidator()
    return _validator_instance