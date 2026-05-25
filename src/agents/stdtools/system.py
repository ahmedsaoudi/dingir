import ast
import operator
import datetime


def calculator(expression: str) -> str:
    """Safely evaluates a mathematical expression containing standard arithmetic operations (+, -, *, /, **, %). Only numeric literals and operators are allowed; no variables or function calls."""
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: lambda x: x,
    }

    def _eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            op = type(node.op)
            if op not in allowed_operators:
                raise TypeError(f"Unsupported operator: {op.__name__}")
            return allowed_operators[op](_eval_node(node.left), _eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            op = type(node.op)
            if op not in allowed_operators:
                raise TypeError(f"Unsupported unary operator: {op.__name__}")
            return allowed_operators[op](_eval_node(node.operand))
        else:
            raise TypeError(f"Unsupported expression element: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Math evaluation error: {str(e)}"


def current_datetime() -> str:
    """Returns the current local date, time, and timezone in ISO-like format (YYYY-MM-DD HH:MM:SS TZ)."""
    now = datetime.datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")
