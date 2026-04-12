# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sync the Dynamo import acceptance test with a live Dynamo checkout.

Scans a Dynamo repository for all ``tensorrt_llm`` imports and compares
them against the DYNAMO_IMPORTS list in ``test_dynamo_imports.py``.

Usage:
    # Show what changed (exits non-zero if out of sync):
    python sync_dynamo_imports.py /path/to/dynamo

    # Update test_dynamo_imports.py in-place:
    python sync_dynamo_imports.py /path/to/dynamo --update
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

TEST_FILE = Path(__file__).parent / "test_dynamo_imports.py"

# Matches single-line: from tensorrt_llm.foo import bar, baz as qux
_FROM_IMPORT_RE = re.compile(
    r"^\s*from\s+(tensorrt_llm(?:\.\w+)*)\s+import\s+(.+)$")

# Matches bare: import tensorrt_llm  /  import tensorrt_llm.foo
_IMPORT_RE = re.compile(r"^\s*import\s+(tensorrt_llm(?:\.\w+)*)\s*$")


def _parse_symbol_list(raw: str) -> List[str]:
    """Parse 'A, B as C, D' into ['A', 'B', 'D'] (original names only)."""
    symbols = []
    for token in raw.split(","):
        token = token.strip().rstrip("\\").strip()
        if not token or token == "(":
            continue
        name = token.split()[0]  # drop " as alias"
        if name == "#":
            break
        symbols.append(name)
    return symbols


def scan_dynamo_imports(
        dynamo_root: Path) -> Set[Tuple[str, Optional[str]]]:
    """Walk *dynamo_root* and extract all tensorrt_llm imports."""
    imports: Set[Tuple[str, Optional[str]]] = set()

    for py_file in dynamo_root.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        tree = ast.parse(source, filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if not mod.startswith("tensorrt_llm"):
                    continue
                for alias in node.names:
                    imports.add((mod, alias.name))

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("tensorrt_llm"):
                        if alias.name == "tensorrt_llm":
                            continue
                        imports.add((alias.name, None))

    return imports


def read_current_imports() -> Set[Tuple[str, Optional[str]]]:
    """Parse the DYNAMO_IMPORTS list from test_dynamo_imports.py using AST."""
    source = TEST_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_FILE))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DYNAMO_IMPORTS":
                    return _extract_list_of_tuples(node.value)

    raise RuntimeError(
        f"Could not find DYNAMO_IMPORTS assignment in {TEST_FILE}")


def _extract_list_of_tuples(
        node: ast.expr) -> Set[Tuple[str, "str | None"]]:
    """Extract set of (str, str|None) from an AST List of Tuple nodes."""
    assert isinstance(node, ast.List)
    result: Set[Tuple[str, "str | None"]] = set()
    for elt in node.elts:
        assert isinstance(elt, ast.Tuple) and len(elt.elts) == 2
        mod = ast.literal_eval(elt.elts[0])
        sym = ast.literal_eval(elt.elts[1])
        result.add((mod, sym))
    return result


def _format_imports_list(imports: List[Tuple[str, "str | None"]]) -> str:
    """Format a sorted list of (module, symbol) into the DYNAMO_IMPORTS source."""
    lines = ["DYNAMO_IMPORTS = ["]
    current_group = None

    for mod, sym in imports:
        top = mod.split(".")[1] if "." in mod else ""
        group_key = top if top else "top-level"
        if group_key != current_group:
            if current_group is not None:
                lines.append("")
            current_group = group_key

        if sym is None:
            entry = f'    ("{mod}", None),'
        else:
            entry = f'    ("{mod}", "{sym}"),'
        lines.append(entry)

    lines.append("]")
    return "\n".join(lines)


def update_test_file(imports: Set[Tuple[str, str]]) -> None:
    """Rewrite the DYNAMO_IMPORTS list in test_dynamo_imports.py."""
    source = TEST_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_FILE))

    assign_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DYNAMO_IMPORTS":
                    assign_node = node
                    break

    if assign_node is None:
        raise RuntimeError(
            f"Could not find DYNAMO_IMPORTS assignment in {TEST_FILE}")

    sorted_imports = sorted(imports)
    new_block = _format_imports_list(sorted_imports)

    source_lines = source.splitlines(keepends=True)

    start_line = assign_node.lineno - 1
    end_line = assign_node.end_lineno

    # Find the comment lines immediately before the assignment
    comment_start = start_line
    while comment_start > 0 and source_lines[comment_start -
                                              1].strip().startswith("#"):
        comment_start -= 1

    before = source_lines[:comment_start]
    after = source_lines[end_line:]

    new_source = "".join(before) + new_block + "\n" + "".join(after)
    TEST_FILE.write_text(new_source, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Sync Dynamo import acceptance tests with a Dynamo checkout."
    )
    parser.add_argument("dynamo_root",
                        type=Path,
                        help="Path to the Dynamo repository root.")
    parser.add_argument(
        "--update",
        action="store_true",
        help=
        "Update test_dynamo_imports.py in-place (default: diff mode, exits non-zero if stale)."
    )
    args = parser.parse_args()

    if not args.dynamo_root.is_dir():
        print(f"Error: {args.dynamo_root} is not a directory.", file=sys.stderr)
        sys.exit(2)

    live_imports = scan_dynamo_imports(args.dynamo_root)
    current_imports = read_current_imports()

    added = live_imports - current_imports
    removed = current_imports - live_imports

    if not added and not removed:
        print("DYNAMO_IMPORTS is up to date.")
        sys.exit(0)

    if added:
        print(f"\n{len(added)} new import(s) in Dynamo not in DYNAMO_IMPORTS:")
        for mod, sym in sorted(added):
            print(f"  + ({mod}, {sym})")

    if removed:
        print(
            f"\n{len(removed)} import(s) in DYNAMO_IMPORTS no longer used by Dynamo:"
        )
        for mod, sym in sorted(removed):
            print(f"  - ({mod}, {sym})")

    if args.update:
        merged = live_imports
        update_test_file(merged)
        print(f"\nUpdated {TEST_FILE}")
        sys.exit(0)
    else:
        print(
            "\nRun with --update to apply changes, or manually edit test_dynamo_imports.py."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
