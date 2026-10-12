"""calc_server.py — a tiny but REAL MCP server (P3 of our agent project).

WHAT IS THIS?
=============
A standalone process that exposes a few calculator tools over the Model
Context Protocol. Our agent (the MCP *client*) spawns this file as a
subprocess and talks JSON-RPC to it over stdin/stdout ("stdio transport").

It is deliberately simple — the point is to see the MECHANISM:
  - tools are declared with @server.tool() and a type-annotated function
  - the MCP framework auto-generates the JSON Schema from the type hints
    (that's why the agent sees {"a": {"type": "integer"}} without us writing it)
  - the server runs forever, answering requests, until the client kills it

RUN IT STANDALONE (optional debugging — no agent needed):
    python servers/calc_server.py
then in another terminal:
    npx @modelcontextprotocol/inspector python servers/calc_server.py
(That inspector UI is MCP's own tool for poking at servers.)

DEPENDENCY:  pip install mcp   (we use the v2 MCPServer API)
"""

import ast
import operator

from mcp.server.mcpserver import MCPServer

# Create the server. The name is what the agent sees in /mcp.
server = MCPServer("calc")


# ---------------------------------------------------------------------------
# A SAFE expression evaluator.
#
# Why not eval()/exec()? They run arbitrary Python — a model-generated
# expression could do "import os; os.system(...)" — catastrophic.
# Instead we parse the expression into an AST (Abstract Syntax Tree) and
# walk it ourselves, allowing ONLY numbers and math operators.
# Anything else (names, calls, attribute access) is rejected.
# This whitelist approach is how real calculators/ sandboxes are built.
# ---------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate one AST node. Raises ValueError on anything unsafe."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value  # a plain number — safe
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)  # recurse into both sides
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 64:
            raise ValueError("exponent too large (max 64)")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def _safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")  # parse to AST, don't execute
    return _eval_node(tree.body)


# ---------------------------------------------------------------------------
# The tools. Docstrings become the tool DESCRIPTIONS the model reads!
# ---------------------------------------------------------------------------
@server.tool()
def add(a: float, b: float) -> float:
    """Add two numbers: a + b."""
    return a + b


@server.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers: a * b."""
    return a * b


@server.tool()
def evaluate(expression: str) -> str:
    """Safely evaluate a math expression string, e.g. '(17 + 24) * 3 - 12 / 4'.

    Supports + - * / // % ** and parentheses. No functions or variables.
    Returns the result, or an error message if the expression is invalid.
    """
    try:
        return str(_safe_eval(expression))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        # Return errors as strings — the agent's contract is "text in, text out".
        return f"Error: {exc}"


if __name__ == "__main__":
    import asyncio

    # Block forever, serving requests over stdin/stdout. The client (our
    # agent) started this process and will terminate it when done.
    asyncio.run(server.run_stdio_async())
