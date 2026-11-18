"""Enforce file-size limits from AGENTS.md §2.

Fails (exit 1) if any ``src/**/*.py`` exceeds 300 lines or any ``notebooks/src/*.py`` exceeds
400 lines. Warns (no failure) for functions longer than 50 lines.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_MAX = 300
NOTEBOOK_MAX = 400
FUNC_MAX = 50


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _long_functions(path: Path) -> list[tuple[str, int, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            if length > FUNC_MAX:
                out.append((node.name, node.lineno, length))
    return out


def check(root: Path = ROOT) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for all checked files under ``root``."""
    errors: list[str] = []
    warnings: list[str] = []
    groups = [(sorted((root / "src").rglob("*.py")), MODULE_MAX)]
    groups.append((sorted((root / "notebooks" / "src").glob("*.py")), NOTEBOOK_MAX))
    for files, limit in groups:
        for f in files:
            rel = f.relative_to(root)
            n = _line_count(f)
            if n > limit:
                errors.append(f"{rel}: {n} lines > {limit}")
            for name, line, length in _long_functions(f):
                warnings.append(f"{rel}:{line} function {name}() is {length} lines > {FUNC_MAX}")
    return errors, warnings


def main() -> int:
    errors, warnings = check()
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print(f"size-limits: {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
