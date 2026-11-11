"""Static (AST) checks applied to model-written code before it is executed.

This is defense in depth, not the security boundary. The boundary is the
sandbox process (scrubbed environment, temp working directory, timeouts and,
on Windows, a Job Object with memory/process limits). The guard exists to
reject the obvious escapes cheaply and to stop a genome from reaching the
evaluator, the archive or the network directly instead of through the broker.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

# Modules a genome (agent.py) may import. It talks to the outside world only
# through the `llm` and `tools` objects the supervisor passes in.
GENOME_IMPORTS = frozenset({
    "json", "re", "math", "itertools", "functools", "collections", "textwrap",
    "random", "string", "typing", "dataclasses", "difflib", "statistics",
    "heapq", "bisect", "ast", "enum", "copy", "operator", "time",
})

# Modules a task *solution* may import.
SOLUTION_IMPORTS = frozenset({
    "math", "re", "collections", "itertools", "functools", "heapq", "bisect",
    "string", "typing", "dataclasses", "fractions", "decimal", "statistics",
    "operator", "copy", "json", "enum", "abc", "array", "textwrap", "random",
    "unicodedata", "numbers", "graphlib", "datetime", "calendar", "__future__",
})

FORBIDDEN_CALLS = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "breakpoint", "input", "help", "exit", "quit",
})
GENOME_FORBIDDEN_CALLS = FORBIDDEN_CALLS | {"getattr", "setattr", "delattr"}
ALLOWED_DUNDERS = frozenset({"__name__", "__init__", "__repr__", "__str__", "__eq__",
                             "__lt__", "__le__", "__gt__", "__ge__", "__hash__", "__len__",
                             "__iter__", "__next__", "__contains__", "__getitem__",
                             "__setitem__", "__post_init__", "__future__", "__all__",
                             "__slots__", "__bool__", "__add__", "__sub__", "__mul__"})


@dataclass
class GuardResult:
    ok: bool
    problems: list[str]

    def __bool__(self) -> bool:
        return self.ok


def _check(source: str, allowed_imports: frozenset[str], forbidden_calls: frozenset[str]) -> GuardResult:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return GuardResult(False, [f"SyntaxError: {e.msg} (line {e.lineno})"])
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in allowed_imports:
                    problems.append(f"line {node.lineno}: import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in allowed_imports:
                problems.append(f"line {node.lineno}: import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in forbidden_calls and isinstance(node.ctx, ast.Load):
                problems.append(f"line {node.lineno}: use of '{node.id}' is not allowed")
            elif node.id.startswith("__") and node.id not in ALLOWED_DUNDERS:
                problems.append(f"line {node.lineno}: name '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS:
                problems.append(f"line {node.lineno}: attribute '{node.attr}' is not allowed")
            if node.attr.startswith("_") and isinstance(node.value, ast.Name) and node.value.id in ("llm", "tools"):
                problems.append(f"line {node.lineno}: private broker attribute '{node.attr}' is not allowed")
    return GuardResult(not problems, problems)


def check_genome_code(source: str) -> GuardResult:
    return _check(source, GENOME_IMPORTS, GENOME_FORBIDDEN_CALLS)


def check_solution_code(source: str) -> GuardResult:
    return _check(source, SOLUTION_IMPORTS, FORBIDDEN_CALLS)
