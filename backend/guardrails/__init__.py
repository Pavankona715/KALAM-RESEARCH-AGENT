"""
Guardrails package — input validation, output validation, permission enforcement.

Public API:
    InputValidator / get_input_validator()
    OutputValidator / get_output_validator()
    PermissionChecker / get_permission_checker()
    GuardrailsMiddleware — integrates all three into the chat pipeline
"""

from backend.guardrails.input_validator import (
    InputValidator,
    ValidationResult,
    get_input_validator,
)
from backend.guardrails.output_validator import (
    OutputValidationResult,
    OutputValidator,
    get_output_validator,
)
from backend.guardrails.permissions import (
    PermissionChecker,
    PermissionDecision,
    PermissionResult,
    UserRole,
    get_permission_checker,
)

__all__ = [
    "InputValidator",
    "ValidationResult",
    "get_input_validator",
    "OutputValidationResult",
    "OutputValidator",
    "get_output_validator",
    "PermissionChecker",
    "PermissionDecision",
    "PermissionResult",
    "UserRole",
    "get_permission_checker",
]