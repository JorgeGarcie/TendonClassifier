#!/usr/bin/env python3
"""Check documentation health: broken references, staleness, accuracy.

Usage:
    python scripts/check_docs_freshness.py           # report issues
    python scripts/check_docs_freshness.py --strict   # exit 1 if issues found
"""

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Library/framework prefixes to skip (not file paths)
SKIP_PREFIXES = [
    "torch.", "torchvision.", "np.", "numpy.", "scipy.", "sklearn.",
    "cv2.", "matplotlib.", "plt.", "wandb.", "yaml.", "json.",
    "os.", "sys.", "re.", "ast.", "math.", "pathlib.", "collections.",
    "dataclasses.", "typing.", "abc.", "io.", "functools.", "itertools.",
    "pandas.", "pd.", "PIL.", "safetensors.", "trimesh.", "openai.",
]


def find_markdown_files() -> list:
    """Find all markdown files in the repo."""
    files = []
    for pattern in ["*.md", "**/*.md"]:
        files.extend(ROOT.glob(pattern))
    return sorted(set(files))


def check_backtick_paths(filepath: Path) -> list:
    """Find backtick-wrapped paths that reference non-existent files."""
    issues = []
    try:
        content = filepath.read_text()
    except (UnicodeDecodeError, PermissionError):
        return issues

    lines = content.splitlines()
    for lineno, line in enumerate(lines, 1):
        # Find backtick-wrapped strings containing /
        matches = re.findall(r"`([^`]+)`", line)
        for match in matches:
            # Must contain a directory separator to be a path
            if "/" not in match:
                continue

            # Skip library references
            if any(match.startswith(prefix) for prefix in SKIP_PREFIXES):
                continue

            # Skip URLs
            if match.startswith("http") or match.startswith("//"):
                continue

            # Skip command-line examples
            if match.startswith("-") or match.startswith("$"):
                continue

            # Skip glob patterns with wildcards
            if "*" in match:
                continue

            # Skip regex patterns
            if match.startswith("^") or match.endswith("$"):
                continue

            # Skip template placeholders (contain < > angle brackets)
            if "<" in match and ">" in match:
                continue

            # Skip paths with spaces (likely prose or commands)
            if " " in match:
                continue

            # Skip home-relative paths and ROS topics
            if match.startswith("~") or match.startswith("/"):
                continue

            # Skip bare directory names (no extension, ends with /)
            if match.endswith("/") and match.count("/") == 1:
                continue

            # Skip if inside a code block (``` fenced) — rough check
            if "```" in line:
                continue

            # Skip file:line references (e.g., file.py:42)
            if re.search(r":\d+", match):
                continue

            # Try to resolve the path
            # Handle relative paths from the markdown file's directory
            ref_path = filepath.parent / match
            if not ref_path.exists():
                # Also try from repo root
                ref_path = ROOT / match
                if not ref_path.exists():
                    # Also try from scripts/ directory
                    ref_path = ROOT / "scripts" / match
                    if not ref_path.exists():
                        issues.append({
                            "file": str(filepath.relative_to(ROOT)),
                            "line": lineno,
                            "type": "broken_reference",
                            "message": f"Referenced path does not exist: `{match}`",
                        })

    return issues


def check_staleness(filepath: Path) -> list:
    """Flag docs with old 'Last updated' dates or pending markers."""
    issues = []
    try:
        content = filepath.read_text()
    except (UnicodeDecodeError, PermissionError):
        return issues

    lines = content.splitlines()
    cutoff = datetime.now() - timedelta(days=30)

    for lineno, line in enumerate(lines, 1):
        # Check for "Last updated: YYYY-MM-DD" patterns
        date_match = re.search(r"[Ll]ast\s+updated[:\s]+(\d{4}-\d{2}-\d{2})", line)
        if date_match:
            try:
                updated = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                if updated < cutoff:
                    days_old = (datetime.now() - updated).days
                    issues.append({
                        "file": str(filepath.relative_to(ROOT)),
                        "line": lineno,
                        "type": "stale",
                        "message": f"Last updated {days_old} days ago ({date_match.group(1)})",
                    })
            except ValueError:
                pass

        # Check for pending/awaiting markers
        lower = line.lower()
        if any(marker in lower for marker in ["pending", "awaiting", "todo", "tbd"]):
            # Skip if in a completed checklist item
            if line.strip().startswith("- [x]"):
                continue
            # Skip if it's a section header about TODOs (like "Next Steps / TODOs")
            if line.strip().startswith("#"):
                continue
            issues.append({
                "file": str(filepath.relative_to(ROOT)),
                "line": lineno,
                "type": "pending",
                "message": f"Pending/TODO marker found: {line.strip()[:80]}",
            })

    return issues


def check_claude_md_references() -> list:
    """Verify CLAUDE.md Deeper Context table references existing files."""
    issues = []
    claude_md = ROOT / "CLAUDE.md"
    if not claude_md.exists():
        issues.append({
            "file": "CLAUDE.md",
            "line": 0,
            "type": "missing",
            "message": "CLAUDE.md does not exist",
        })
        return issues

    content = claude_md.read_text()
    lines = content.splitlines()
    in_table = False

    for lineno, line in enumerate(lines, 1):
        # Detect table rows (pipe-separated)
        if "Deeper Context" in line:
            in_table = True
            continue
        if in_table and line.strip().startswith("|"):
            # Extract path from table row
            cells = [c.strip() for c in line.split("|")]
            for cell in cells:
                # Look for backtick-wrapped paths
                path_match = re.search(r"`([^`]+)`", cell)
                if path_match:
                    ref = path_match.group(1)
                    if "/" in ref:
                        full_path = ROOT / ref
                        if not full_path.exists():
                            issues.append({
                                "file": "CLAUDE.md",
                                "line": lineno,
                                "type": "broken_reference",
                                "message": f"Deeper Context references missing file: `{ref}`",
                            })
        elif in_table and not line.strip().startswith("|") and line.strip():
            in_table = False

    return issues


def main():
    parser = argparse.ArgumentParser(description="Check documentation freshness")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if issues found")
    args = parser.parse_args()

    md_files = find_markdown_files()
    all_issues = []

    for f in md_files:
        all_issues.extend(check_backtick_paths(f))
        all_issues.extend(check_staleness(f))

    all_issues.extend(check_claude_md_references())

    # Group by type
    by_type = {}
    for issue in all_issues:
        t = issue["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(issue)

    total = len(all_issues)
    print(f"{'='*60}")
    print(f"Documentation Freshness Check — {total} issue(s) found")
    print(f"Scanned {len(md_files)} markdown files")
    print(f"{'='*60}\n")

    if total == 0:
        print("All docs are fresh and references are valid!")
        return

    type_labels = {
        "broken_reference": "Broken References",
        "stale": "Stale Documentation",
        "pending": "Pending/TODO Markers",
        "missing": "Missing Files",
    }

    for issue_type in sorted(by_type.keys()):
        issues = by_type[issue_type]
        label = type_labels.get(issue_type, issue_type)
        print(f"## {label} ({len(issues)} issues)\n")
        for issue in issues:
            loc = f"{issue['file']}:{issue['line']}" if issue["line"] else issue["file"]
            print(f"  - {loc} — {issue['message']}")
        print()

    if args.strict and total > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
