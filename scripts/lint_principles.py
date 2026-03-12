#!/usr/bin/env python3
"""Lint source files for golden principle violations.

Scans all Python source files for anti-patterns. Each violation includes
file:line and a fix instruction written for agents.

Usage:
    python scripts/lint_principles.py           # report violations
    python scripts/lint_principles.py --strict   # exit 1 if any violations
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = [
    ROOT / "scripts" / "classification",
    ROOT / "scripts" / "labeling",
]
SKIP_FILES = {"sparsh_vit.py"}  # vendored code

# --- GP1: One Fact, One Place ---
# Magic numbers that should be in config
MAGIC_NUMBER_PATTERNS = [
    # ImageNet normalization hardcoded outside dataset.py
    (r"\[0\.485,\s*0\.456,\s*0\.406\]", "ImageNet mean", "dataset.py"),
    (r"\[0\.229,\s*0\.224,\s*0\.225\]", "ImageNet std", "dataset.py"),
    # Force threshold hardcoded outside labeling/config.py
    (r"(?<!\w)12\.0(?!\d)", "force threshold (12.0)", "labeling/config.py"),
    # Crop size hardcoded outside labeling/config.py
    (r"(?<!\w)1080(?!\d)", "crop size (1080)", "labeling/config.py"),
    # Boundary margin hardcoded
    (r"0\.003(?!\d)", "boundary margin (0.003)", "config or constants"),
]

# Files where certain magic numbers are allowed (their source of truth)
MAGIC_ALLOWED = {
    "ImageNet mean": {"dataset.py", "config.py"},
    "ImageNet std": {"dataset.py", "config.py"},
    "force threshold (12.0)": {"config.py"},
    "crop size (1080)": {"config.py", "generate_gt.py"},
    "boundary margin (0.003)": {"dataset.py", "config.py"},
}

# --- GP4: Fail Loud, Not Silent ---
SILENT_FAIL_PATTERNS = [
    (r"except\s*:", "bare except (swallows all errors)"),
    (r"except\s+Exception\s*:", "broad except Exception (may hide bugs)"),
    (r"pass\s*$", None),  # only flag if inside except block (checked via AST)
]


def is_noqa(line: str) -> bool:
    """Check if line has a # noqa comment."""
    return "# noqa" in line


def is_in_comment_or_docstring(line: str, stripped: str) -> bool:
    """Check if the match is in a comment."""
    return stripped.startswith("#")


def scan_magic_numbers(filepath: Path) -> list:
    """GP1: Find hardcoded magic numbers that belong in config."""
    violations = []
    fname = filepath.name

    try:
        lines = filepath.read_text().splitlines()
    except (UnicodeDecodeError, PermissionError):
        return violations

    for lineno, line in enumerate(lines, 1):
        if is_noqa(line):
            continue
        stripped = line.strip()
        if is_in_comment_or_docstring(line, stripped):
            continue
        # Skip docstrings (rough heuristic — triple quotes)
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        for pattern, name, source in MAGIC_NUMBER_PATTERNS:
            allowed_files = MAGIC_ALLOWED.get(name, set())
            if fname in allowed_files:
                continue
            if re.search(pattern, line):
                violations.append({
                    "file": str(filepath.relative_to(ROOT)),
                    "line": lineno,
                    "principle": "GP1",
                    "message": f"Magic number: {name}",
                    "fix": f"Import from {source} instead of hardcoding",
                })

    return violations


def scan_silent_failures(filepath: Path) -> list:
    """GP4: Find bare except blocks and silent error swallowing."""
    violations = []

    try:
        source = filepath.read_text()
        lines = source.splitlines()
    except (UnicodeDecodeError, PermissionError):
        return violations

    # AST-based check for except blocks with only pass
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            # Check for bare except (no type specified)
            if node.type is None:
                lineno = node.lineno
                if lineno <= len(lines) and not is_noqa(lines[lineno - 1]):
                    violations.append({
                        "file": str(filepath.relative_to(ROOT)),
                        "line": lineno,
                        "principle": "GP4",
                        "message": "Bare except: catches all exceptions silently",
                        "fix": "Specify the exception type (e.g., except ValueError:) and log the error",
                    })

            # Check for except with only pass body
            if (len(node.body) == 1
                    and isinstance(node.body[0], ast.Pass)):
                lineno = node.body[0].lineno
                if lineno <= len(lines) and not is_noqa(lines[lineno - 1]):
                    violations.append({
                        "file": str(filepath.relative_to(ROOT)),
                        "line": lineno,
                        "principle": "GP4",
                        "message": "Silent except: catches exception and does nothing (pass)",
                        "fix": "Add logging (e.g., logger.warning) or re-raise the exception",
                    })

    return violations


def scan_unused_imports(filepath: Path) -> list:
    """GP2: Find unused imports."""
    violations = []

    try:
        source = filepath.read_text()
    except (UnicodeDecodeError, PermissionError):
        return violations

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return violations

    # Collect imported names
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                imports[name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname if alias.asname else alias.name
                imports[name] = node.lineno

    # Check if each import is used in the source (simple text search)
    for name, lineno in imports.items():
        # Count occurrences (excluding the import line itself)
        lines = source.splitlines()
        if lineno <= len(lines) and is_noqa(lines[lineno - 1]):
            continue

        # Count uses of the name in non-import lines
        use_count = 0
        for i, line in enumerate(lines, 1):
            if i == lineno:
                continue
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue
            if re.search(rf"\b{re.escape(name)}\b", line):
                use_count += 1

        if use_count == 0:
            violations.append({
                "file": str(filepath.relative_to(ROOT)),
                "line": lineno,
                "principle": "GP2",
                "message": f"Unused import: {name}",
                "fix": f"Remove the unused import of '{name}'",
            })

    return violations


def scan_hardcoded_config_values(filepath: Path) -> list:
    """GP5: Find hardcoded values that should be in YAML config."""
    violations = []
    fname = filepath.name

    # Only check classification source files (not configs, tests, or labeling)
    if fname in {"config.py", "sparsh_vit.py"} or fname.endswith(".yaml"):
        return violations

    try:
        lines = filepath.read_text().splitlines()
    except (UnicodeDecodeError, PermissionError):
        return violations

    for lineno, line in enumerate(lines, 1):
        if is_noqa(line):
            continue
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        # Check for hardcoded num_classes outside config
        if re.search(r"num_classes\s*=\s*4\b", line) and fname not in {"config.py", "test_config.py", "test_models.py"}:
            violations.append({
                "file": str(filepath.relative_to(ROOT)),
                "line": lineno,
                "principle": "GP5",
                "message": "Hardcoded num_classes=4",
                "fix": "Read num_classes from config instead of hardcoding",
            })

    return violations


def collect_source_files() -> list:
    """Collect all Python source files to scan."""
    files = []
    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            continue
        for py_file in src_dir.glob("*.py"):
            if py_file.name in SKIP_FILES:
                continue
            files.append(py_file)
    # Also scan tests
    test_dir = ROOT / "tests"
    if test_dir.exists():
        for py_file in test_dir.glob("*.py"):
            files.append(py_file)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Lint golden principle violations")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if violations found")
    args = parser.parse_args()

    files = collect_source_files()
    all_violations = []

    for f in files:
        all_violations.extend(scan_magic_numbers(f))
        all_violations.extend(scan_silent_failures(f))
        all_violations.extend(scan_unused_imports(f))
        all_violations.extend(scan_hardcoded_config_values(f))

    # Group by principle
    by_principle = defaultdict(list)
    for v in all_violations:
        by_principle[v["principle"]].append(v)

    # Report
    total = len(all_violations)
    print(f"{'='*60}")
    print(f"Golden Principles Lint — {total} violation(s) found")
    print(f"Scanned {len(files)} files")
    print(f"{'='*60}\n")

    if total == 0:
        print("All clean!")
        return

    for principle in sorted(by_principle.keys()):
        violations = by_principle[principle]
        principle_names = {
            "GP1": "One Fact, One Place",
            "GP2": "Centralize, Don't Duplicate",
            "GP3": "Validate at Boundaries",
            "GP4": "Fail Loud, Not Silent",
            "GP5": "Config Drives Experiments",
        }
        name = principle_names.get(principle, principle)
        print(f"## {principle}: {name} ({len(violations)} violations)\n")

        for v in violations:
            print(f"  - {v['file']}:{v['line']} — {v['message']}")
            print(f"    FIX: {v['fix']}")
        print()

    if args.strict and total > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
