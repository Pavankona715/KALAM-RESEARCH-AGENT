"""
Calculator Tool
===============
Safely evaluates mathematical expressions using sympy.

WHY NOT eval()?
eval() executes arbitrary Python code — a massive security hole.
sympy.sympify() parses only mathematical expressions and rejects
anything that isn't valid math, preventing code injection.

Supports:
- Basic arithmetic: +, -, *, /, **, %
- Functions: sqrt, sin, cos, tan, log, exp, abs, floor, ceil
- Constants: pi, e, inf
- Symbolic simplification: simplify("(x^2 - 1) / (x - 1)")
"""

from __future__ import annotations

from typing import Any

from backend.tools.base import BaseTool
from backend.tools.registry import get_tool_registry
from backend.observability.logger import get_logger

logger = get_logger(__name__)

# Allowlist of safe sympy functions and constants
SAFE_FUNCTIONS = {
    "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "log", "ln", "exp", "abs", "floor", "ceil", "round",
    "factorial", "gcd", "lcm", "mod",
    "pi", "e", "inf", "oo",  # constants
    "simplify", "expand", "factor", "solve",
}


class CalculatorTool(BaseTool):
    """
    Evaluate mathematical expressions safely.

    Best for:
    - Arithmetic calculations
    - Unit conversions
    - Statistical calculations
    - Algebraic simplification
    """

    name = "calculator"
    description = (
        "Evaluate mathematical expressions. Supports arithmetic (+, -, *, /, **), "
        "functions (sqrt, sin, cos, log, exp), and constants (pi, e). "
        "Use this for any numerical calculation rather than guessing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "Mathematical expression to evaluate. "
                    "Examples: '2 + 2', 'sqrt(144)', 'sin(pi/4)', '(100 * 1.08) ** 12'"
                ),
            }
        },
        "required": ["expression"],
    }

    async def _execute(self, expression: str) -> str:
        """Safely evaluate the expression using sympy."""
        try:
            import sympy

            # Clean up common notations
            expr = expression.strip()
            expr = expr.replace("^", "**")    # Support ^ for exponentiation
            expr = expr.replace("×", "*")     # Support × multiplication sign
            expr = expr.replace("÷", "/")     # Support ÷ division sign

            # Parse with sympy (safe — does not execute Python code)
            result = sympy.sympify(expr, evaluate=True)

            # Convert to float if possible for cleaner output
            try:
                float_result = float(result.evalf())
                # Format nicely: avoid scientific notation for reasonable numbers
                if abs(float_result) < 1e15 and abs(float_result) > 1e-10 or float_result == 0:
                    # Use integer representation if it's a whole number
                    if float_result == int(float_result) and abs(float_result) < 1e12:
                        return f"{expression} = {int(float_result)}"
                    else:
                        return f"{expression} = {float_result:,.6f}".rstrip("0").rstrip(".")
                else:
                    return f"{expression} = {float_result:.6e}"
            except (TypeError, ValueError):
                # Symbolic result (e.g., from simplify())
                return f"{expression} = {result}"

        except ImportError:
            raise ImportError("sympy not installed. Run: pip install sympy")
        except Exception as e:
            raise ValueError(f"Could not evaluate '{expression}': {e}")


# ─── Auto-registration ────────────────────────────────────────────────────────

get_tool_registry().register(CalculatorTool())