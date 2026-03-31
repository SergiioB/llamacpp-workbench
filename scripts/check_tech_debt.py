#!/usr/bin/env python3
"""
Tech debt tracking enforcement script.

Validates that TODO/FIXME/HACK/XXX comments link to GitHub issues.
Accepted formats:
- TODO(#123) - links to issue by number in current repo
- TODO(https://github.com/owner/repo/issues/123) - full URL
- FIXME(#456) - same patterns apply
- HACK(#789) - same patterns apply
- XXX(#012) - same patterns apply

Usage:
    python scripts/check_tech_debt.py [files...]

If no files specified, scans all Python and JavaScript files in src/ and static/.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Pattern to match TODO/FIXME/HACK/XXX comments that DON'T link to issues
# Valid formats: TODO(#123), TODO(https://github.com/...)
TECH_DEBT_PATTERN = re.compile(
    r"""
    (
        \#\s*(TODO|FIXME|HACK|XXX)  # Python comment style
        |
        \/\/\s*(TODO|FIXME|HACK|XXX)  # JavaScript comment style
        |
        \*\s*(TODO|FIXME|HACK|XXX)  # JS block comment style
    )
    \s*
    (?!\(  # Must NOT be followed by opening paren for issue link
        (
            \#\d+  # Issue number like #123
            |
            https://github\.com/[\w\-]+/[\w\-]+/issues/\d+  # Full GitHub URL
        )
        \)  # Closing paren
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Pattern to validate properly formatted tech debt comments
VALID_TECH_DEBT_PATTERN = re.compile(
    r"""
    (
        \#\s*|\/\/\s*|\*\s*
    )
    (TODO|FIXME|HACK|XXX)
    \s*
    \(
        (
            \#\d+
            |
            https://github\.com/[\w\-]+/[\w\-]+/issues/\d+
        )
    \)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def check_file(filepath: Path) -> list[tuple[int, str]]:
    """Check a file for tech debt comments without issue links.

    Args:
        filepath: Path to file to check

    Returns:
        List of (line_number, comment_text) for violations
    """
    violations: list[tuple[int, str]] = []

    try:
        content = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return violations

    for line_num, line in enumerate(content.splitlines(), start=1):
        # Skip lines that have properly formatted tech debt comments
        if VALID_TECH_DEBT_PATTERN.search(line):
            continue

        # Check for unlinked tech debt comments
        match = TECH_DEBT_PATTERN.search(line)
        if match:
            violations.append((line_num, line.strip()))

    return violations


def main() -> int:
    """Main entry point.

    Returns:
        0 if all checks pass, 1 if violations found
    """
    # Determine files to check
    if len(sys.argv) > 1:
        files = [Path(f) for f in sys.argv[1:]]
    else:
        # Default: scan src/ and static/ for Python and JavaScript files
        project_root = Path(__file__).parent.parent
        files = list(project_root.glob("src/**/*.py")) + list(
            project_root.glob("static/**/*.js")
        )
        # Also include scripts directory
        files.extend(project_root.glob("scripts/*.py"))

    all_violations: list[tuple[Path, int, str]] = []

    for filepath in files:
        if not filepath.exists():
            continue

        violations = check_file(filepath)
        for line_num, comment in violations:
            all_violations.append((filepath, line_num, comment))

    if all_violations:
        print("Tech debt comments must link to GitHub issues:\n")
        print("Valid formats:")
        print("  TODO(#123)              - link to issue by number")
        print("  TODO(https://github.com/owner/repo/issues/123) - full URL\n")
        print("Violations found:\n")

        for filepath, line_num, comment in all_violations:
            rel_path = filepath.relative_to(Path(__file__).parent.parent)
            print(f"  {rel_path}:{line_num}: {comment}")

        print(f"\nTotal: {len(all_violations)} violations")
        return 1

    print("All tech debt comments properly link to issues.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
