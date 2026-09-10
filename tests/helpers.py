"""Test helpers.

`code_only` exists because several architectural invariants are checked by
searching the source, and a naive text search matches the prose that *describes*
the invariant as readily as a violation of it. Stripping comments and docstrings
first means those checks test the code rather than the documentation.
"""

from __future__ import annotations

import ast
import io
import pathlib
import tokenize


def code_only(path: pathlib.Path) -> str:
    """Return the file's source with comments and docstrings removed."""
    src = path.read_text()

    # Drop comments via the tokeniser, preserving line structure.
    out_lines = src.splitlines()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            row = tok.start[0] - 1
            out_lines[row] = out_lines[row][: tok.start[1]]

    # Blank out docstring line ranges.
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(
            first.value.value, str
        ):
            for row in range(first.lineno - 1, (first.end_lineno or first.lineno)):
                out_lines[row] = ""

    return "\n".join(out_lines)
