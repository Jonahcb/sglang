#!/usr/bin/env python3
"""Pre-commit hook: disallow direct triton imports across the repo.

All triton imports should go through `sglang.srt.triton_utils` so that the
codebase can be loaded in environments where triton is not installed.

Allowed files (the shim layer itself) are listed in ALLOWED below.
"""

import ast
import pathlib
import sys

ALLOWED = {
    "python/sglang/srt/triton_utils",
}

# Top-level directories to scan for .py files.
SCAN_ROOTS = [
    "python",
    "benchmark",
    "test",
    "examples",
    "sgl-kernel",
]


def main() -> int:
    errors: list[str] = []

    for root in SCAN_ROOTS:
        root_path = pathlib.Path(root)
        if not root_path.exists():
            continue
        for p in root_path.rglob("*.py"):
            if any(str(p).startswith(a) for a in ALLOWED):
                continue
            try:
                source = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            try:
                tree = ast.parse(source, filename=str(p))
            except SyntaxError:
                continue

            # Collect import node ids that are inside try/except blocks —
            # those are guarded and won't break Triton-free environments.
            guarded: set[int] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    for child in ast.walk(node):
                        if isinstance(child, (ast.Import, ast.ImportFrom)):
                            guarded.add(id(child))

            for node in ast.walk(tree):
                if id(node) in guarded:
                    continue
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "triton" or alias.name.startswith("triton."):
                            errors.append(
                                f"{p}:{node.lineno}: direct `import {alias.name}` found"
                            )
                elif isinstance(node, ast.ImportFrom) and node.module and (
                    node.module == "triton" or node.module.startswith("triton.")
                ):
                    errors.append(
                        f"{p}:{node.lineno}: direct `from {node.module} import ...` found"
                    )

    if errors:
        print(
            "❌ Direct triton imports detected. "
            "Import triton via `from sglang.srt.triton_utils import [triton, tl, ...]` instead.\n"
        )
        print("\n".join(errors))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
