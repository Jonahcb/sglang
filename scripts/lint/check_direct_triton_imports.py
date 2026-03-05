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


class TritonImportChecker(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.errors = []
        self._in_try_block = 0

    def visit_Try(self, node):
        self._in_try_block += 1
        self.generic_visit(node)
        self._in_try_block -= 1

    def _check_triton(self, name, lineno, is_from=False):
        if self._in_try_block > 0:
            return

        if name == "triton" or name.startswith("triton."):
            msg_type = f"from {name} import ..." if is_from else f"import {name}"
            self.errors.append(f"{self.filename}:{lineno}: direct `{msg_type}` found")

    def visit_Import(self, node):
        for alias in node.names:
            self._check_triton(alias.name, node.lineno)

    def visit_ImportFrom(self, node):
        if node.module:
            self._check_triton(node.module, node.lineno, is_from=True)


def main() -> int:
    errors: list[str] = []

    for root in SCAN_ROOTS:
        root_path = pathlib.Path(root)
        if not root_path.is_dir():
            raise NotADirectoryError(f"Required directory not found: {root_path}")
        for p in root_path.rglob("*.py"):
            if any(str(p).startswith(a) for a in ALLOWED):
                continue
            try:
                source = p.read_bytes()
                tree = ast.parse(source, filename=str(p))
            except (OSError, SyntaxError):
                # I think we should continue here because these other issues should be caught by other linting tools
                continue

            # Run the AST visitor
            checker = TritonImportChecker(str(p))
            checker.visit(tree)
            errors.extend(checker.errors)

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
