"""
Output validation — structured output enforcement via Pydantic.

Design rationale:
- LLM outputs are inherently unstructured; this layer enforces structure when
  agents are expected to return JSON (tool calls, planner output, etc.)
- OutputValidator is a thin Pydantic-based validator that:
    1. Tries to parse raw LLM text as JSON
    2. Validates the JSON against a provided Pydantic model
    3. Returns a typed result or a structured error — never raises
- Content safety check: detect if the model is leaking the system prompt or
  producing clearly harmful text before returning output to the user
- PII scrubbing: optional regex-based removal of emails, phone numbers, SSNs
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class OutputValidationResult(Generic[T]):
    valid: bool
    data: Optional[T] = None
    raw_output: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    was_sanitized: bool = False


# ---------------------------------------------------------------------------
# Content safety patterns
# ---------------------------------------------------------------------------

# Patterns that suggest the model is leaking internal context
_SYSTEM_LEAK_PATTERNS = [
    re.compile(r"<system>", re.I),
    re.compile(r"\[SYSTEM PROMPT\]", re.I),
    re.compile(r"my\s+system\s+prompt\s+(is|says|states)", re.I),
    re.compile(r"you\s+told\s+me\s+to\s+(be|act|pretend)", re.I),
]

# PII patterns for optional scrubbing
_PII_PATTERNS: List[tuple] = [
    # Email
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL_REDACTED]"),
    # US Phone
    (re.compile(r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE_REDACTED]"),
    # SSN
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    # Credit card (loose)
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[CC_REDACTED]"),
]

# Minimum / maximum lengths for agent responses
_MIN_RESPONSE_LENGTH = 1
_MAX_RESPONSE_LENGTH = 50_000


# ---------------------------------------------------------------------------
# OutputValidator
# ---------------------------------------------------------------------------


class OutputValidator:
    """
    Validates and optionally sanitizes LLM outputs.

    Usage:
        validator = OutputValidator()

        # Validate against a Pydantic schema
        result = validator.validate_json(raw_text, MyModel)
        if result.valid:
            use(result.data)

        # Check plain-text response for safety
        result = validator.validate_text(response_text)
    """

    def __init__(
        self,
        scrub_pii: bool = False,
        max_length: int = _MAX_RESPONSE_LENGTH,
    ) -> None:
        self._scrub_pii = scrub_pii
        self._max_length = max_length

    # ------------------------------------------------------------------
    # JSON / structured output validation
    # ------------------------------------------------------------------

    def validate_json(
        self,
        raw_text: str,
        model_class: Type[T],
        strip_markdown: bool = True,
    ) -> OutputValidationResult[T]:
        """
        Parse raw_text as JSON and validate against model_class.

        Handles common LLM quirks:
        - JSON wrapped in ```json ... ``` code fence
        - Trailing commas (stripped before parse)
        - Leading/trailing whitespace
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Strip markdown code fence
        text = raw_text.strip()
        if strip_markdown:
            text = self._strip_code_fence(text)

        # Try to parse JSON
        try:
            data_dict = json.loads(text)
        except json.JSONDecodeError as exc:
            # Second attempt: strip trailing commas
            cleaned = re.sub(r",\s*([}\]])", r"\1", text)
            try:
                data_dict = json.loads(cleaned)
                warnings.append("json_trailing_comma_fixed")
            except json.JSONDecodeError:
                return OutputValidationResult(
                    valid=False,
                    raw_output=raw_text,
                    errors=[f"json_parse_error: {exc.msg} at pos {exc.pos}"],
                )

        # Validate against Pydantic model
        try:
            instance = model_class.model_validate(data_dict)
        except ValidationError as exc:
            field_errors = [
                f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            ]
            return OutputValidationResult(
                valid=False,
                raw_output=raw_text,
                errors=field_errors,
            )

        return OutputValidationResult(
            valid=True,
            data=instance,
            raw_output=raw_text,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Plain text validation
    # ------------------------------------------------------------------

    def validate_text(
        self, text: str, check_safety: bool = True
    ) -> OutputValidationResult[None]:
        """
        Validate a plain-text LLM response.

        Checks:
        - Length bounds
        - System prompt leakage
        - Optional PII scrubbing
        """
        errors: List[str] = []
        warnings: List[str] = []
        was_sanitized = False

        if not text or len(text) < _MIN_RESPONSE_LENGTH:
            return OutputValidationResult(
                valid=False, raw_output=text, errors=["empty_response"]
            )

        if len(text) > self._max_length:
            text = text[: self._max_length]
            warnings.append("response_truncated")
            was_sanitized = True

        # Safety checks
        if check_safety:
            leak_found = self._check_system_leak(text)
            if leak_found:
                warnings.append("possible_system_prompt_leak")
                logger.warning("Output validator: possible system prompt leak detected")

        # PII scrubbing
        if self._scrub_pii:
            text, pii_types = self._scrub_pii_from_text(text)
            if pii_types:
                warnings.append(f"pii_scrubbed: {', '.join(pii_types)}")
                was_sanitized = True

        return OutputValidationResult(
            valid=True,
            raw_output=text,
            errors=errors,
            warnings=warnings,
            was_sanitized=was_sanitized,
        )

    # ------------------------------------------------------------------
    # Tool output validation
    # ------------------------------------------------------------------

    def validate_tool_output(self, tool_name: str, output: Any) -> OutputValidationResult[None]:
        """
        Validate a tool's output before injecting into agent context.

        Basic checks: not None, reasonable size.
        """
        if output is None:
            return OutputValidationResult(
                valid=False, errors=[f"tool '{tool_name}' returned None"]
            )

        text = output if isinstance(output, str) else json.dumps(output)

        if len(text) > self._max_length:
            text = text[: self._max_length]
            return OutputValidationResult(
                valid=True,
                raw_output=text,
                warnings=["tool_output_truncated"],
                was_sanitized=True,
            )

        return OutputValidationResult(valid=True, raw_output=text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` wrappers."""
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.I)
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()

    @staticmethod
    def _check_system_leak(text: str) -> bool:
        """Return True if the output appears to leak system internals."""
        return any(p.search(text) for p in _SYSTEM_LEAK_PATTERNS)

    @staticmethod
    def _scrub_pii_from_text(text: str):
        """Replace PII with redaction placeholders. Returns (cleaned_text, [types])."""
        found_types: List[str] = []
        for pattern, replacement in _PII_PATTERNS:
            if pattern.search(text):
                found_types.append(replacement)
                text = pattern.sub(replacement, text)
        return text, found_types


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_validator_instance: Optional[OutputValidator] = None


def get_output_validator(scrub_pii: bool = False) -> OutputValidator:
    """FastAPI dependency."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = OutputValidator(scrub_pii=scrub_pii)
    return _validator_instance