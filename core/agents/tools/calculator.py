# core/agent/tools/calculator.py
"""
Calculator Tool
Performs mathematical calculations safely
"""

import logging
import re
import math
import operator
from typing import Union, Dict, Any
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)


class SafeCalculator:
    """Safe mathematical expression evaluator"""

    # Allowed operators and functions
    OPERATORS = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "//": operator.floordiv,
        "%": operator.mod,
        "**": operator.pow,
        "^": operator.xor,  # bitwise XOR, not power
    }

    FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sqrt": math.sqrt,
        "pow": pow,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
        "factorial": math.factorial,
    }

    def __init__(self):
        self.safe_dict = {
            "__builtins__": {},
            # Add safe functions and constants
            **self.FUNCTIONS,
        }

    def evaluate(self, expression: str) -> Union[float, int, str]:
        """
        Safely evaluate mathematical expression

        Args:
            expression: Mathematical expression as string

        Returns:
            Calculated result or error message
        """
        try:
            # Clean and validate expression
            cleaned_expr = self._clean_expression(expression)

            if not self._is_safe_expression(cleaned_expr):
                return "Error: Expression contains unsafe operations"

            # Replace power operator ^ with **
            cleaned_expr = cleaned_expr.replace("^", "**")

            # Evaluate using restricted namespace
            result = eval(cleaned_expr, self.safe_dict, {})

            # Handle special float values
            if isinstance(result, float):
                if math.isnan(result):
                    return "Error: Result is NaN"
                elif math.isinf(result):
                    return "Error: Result is infinite"

                # Round very small differences to zero
                if abs(result) < 1e-10:
                    result = 0.0

                # Round to reasonable precision
                if abs(result) < 1e6:
                    result = round(result, 10)

            return result

        except ZeroDivisionError:
            return "Error: Division by zero"
        except ValueError as e:
            return f"Error: Invalid value - {str(e)}"
        except OverflowError:
            return "Error: Number too large"
        except SyntaxError:
            return "Error: Invalid mathematical expression"
        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return f"Error: {str(e)}"

    def _clean_expression(self, expression: str) -> str:
        """Clean and normalize expression"""
        # Remove whitespace
        expr = expression.strip()

        # Remove invalid characters (keep only numbers, operators, parentheses, letters for functions)
        expr = re.sub(r"[^0-9+\-*/().,\s\w^]", "", expr)

        # Handle common text replacements
        replacements = {
            "x": "*",
            "X": "*",
            " to the power of ": "**",
            " power ": "**",
            " squared": "**2",
            " cubed": "**3",
        }

        for old, new in replacements.items():
            expr = expr.replace(old, new)

        return expr

    def _is_safe_expression(self, expression: str) -> bool:
        """Check if expression is safe to evaluate"""
        # Block dangerous keywords
        dangerous_keywords = [
            "import",
            "exec",
            "eval",
            "open",
            "file",
            "input",
            "raw_input",
            "compile",
            "reload",
            "__",
            "quit",
            "exit",
            "globals",
            "locals",
            "vars",
            "dir",
            "help",
            "copyright",
            "credits",
            "license",
        ]

        expr_lower = expression.lower()

        for keyword in dangerous_keywords:
            if keyword in expr_lower:
                return False

        # Check for excessive complexity (prevent DoS)
        if len(expression) > 1000:
            return False

        # Check for excessive nesting
        paren_depth = 0
        max_depth = 0
        for char in expression:
            if char == "(":
                paren_depth += 1
                max_depth = max(max_depth, paren_depth)
            elif char == ")":
                paren_depth -= 1

        if max_depth > 20:
            return False

        return True


def calculate(expression: str) -> Dict[str, Any]:
    """
    Calculate mathematical expression

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Dictionary with calculation result
    """
    calculator = SafeCalculator()

    logger.info(f"Calculating expression: {expression}")

    result = calculator.evaluate(expression)

    # Format response
    if isinstance(result, str) and result.startswith("Error:"):
        return {"success": False, "error": result, "expression": expression}
    else:
        return {
            "success": True,
            "result": result,
            "expression": expression,
            "formatted_result": f"{expression} = {result}",
        }


def basic_math(operation: str, a: float, b: float = None) -> Dict[str, Any]:
    """
    Perform basic mathematical operations

    Args:
        operation: Operation type (add, subtract, multiply, divide, power, sqrt, etc.)
        a: First number
        b: Second number (optional for unary operations)

    Returns:
        Dictionary with operation result
    """
    try:
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
            "power": lambda x, y: x**y,
            "sqrt": lambda x, y=None: math.sqrt(x),
            "square": lambda x, y=None: x**2,
            "cube": lambda x, y=None: x**3,
            "abs": lambda x, y=None: abs(x),
            "log": lambda x, y=None: math.log(x),
            "log10": lambda x, y=None: math.log10(x),
            "sin": lambda x, y=None: math.sin(x),
            "cos": lambda x, y=None: math.cos(x),
            "tan": lambda x, y=None: math.tan(x),
        }

        if operation not in operations:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}",
                "available_operations": list(operations.keys()),
            }

        func = operations[operation]

        # Check if operation requires second parameter
        needs_two_params = operation in [
            "add",
            "subtract",
            "multiply",
            "divide",
            "power",
        ]

        if needs_two_params and b is None:
            return {
                "success": False,
                "error": f"Operation '{operation}' requires two parameters",
            }

        # Perform calculation
        if needs_two_params:
            result = func(a, b)
        else:
            result = func(a)

        if isinstance(result, str) and result.startswith("Error:"):
            return {"success": False, "error": result}

        return {
            "success": True,
            "result": result,
            "operation": operation,
            "operands": [a, b] if b is not None else [a],
        }

    except Exception as e:
        return {"success": False, "error": f"Calculation error: {str(e)}"}


# Additional utility functions
def percentage(value: float, percent: float) -> Dict[str, Any]:
    """Calculate percentage of a value"""
    try:
        result = (value * percent) / 100
        return {
            "success": True,
            "result": result,
            "formatted": f"{percent}% of {value} = {result}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def unit_convert(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """Convert between common units"""
    conversions = {
        # Length conversions (to meters)
        "mm": 0.001,
        "cm": 0.01,
        "m": 1.0,
        "km": 1000.0,
        "inch": 0.0254,
        "ft": 0.3048,
        "yard": 0.9144,
        "mile": 1609.34,
        # Weight conversions (to grams)
        "mg": 0.001,
        "g": 1.0,
        "kg": 1000.0,
        "oz": 28.3495,
        "lb": 453.592,
        # Temperature handled separately
    }

    try:
        if from_unit == to_unit:
            return {
                "success": True,
                "result": value,
                "formatted": f"{value} {from_unit} = {value} {to_unit}",
            }

        # Handle temperature separately
        if from_unit in ["celsius", "fahrenheit", "kelvin"]:
            return _convert_temperature(value, from_unit, to_unit)

        # Check if units are in the same category
        from_category = None
        to_category = None

        for category, units in [
            ("length", ["mm", "cm", "m", "km", "inch", "ft", "yard", "mile"]),
            ("weight", ["mg", "g", "kg", "oz", "lb"]),
        ]:
            if from_unit in units:
                from_category = category
            if to_unit in units:
                to_category = category

        if from_category != to_category or from_category is None:
            return {
                "success": False,
                "error": f"Cannot convert from {from_unit} to {to_unit}",
            }

        # Convert using base unit
        base_value = value * conversions[from_unit]
        result = base_value / conversions[to_unit]

        return {
            "success": True,
            "result": result,
            "formatted": f"{value} {from_unit} = {result} {to_unit}",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin"""
    try:
        # Convert to Celsius first
        if from_unit == "fahrenheit":
            celsius = (value - 32) * 5 / 9
        elif from_unit == "kelvin":
            celsius = value - 273.15
        else:  # celsius
            celsius = value

        # Convert from Celsius to target
        if to_unit == "fahrenheit":
            result = celsius * 9 / 5 + 32
        elif to_unit == "kelvin":
            result = celsius + 273.15
        else:  # celsius
            result = celsius

        return {
            "success": True,
            "result": result,
            "formatted": f"{value}°{from_unit.title()} = {result:.2f}°{to_unit.title()}",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
