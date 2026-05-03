"""Assert every third-party top-level import in the source tree is declared in requirements.txt.

Catches the "shipped a new dep without adding it to requirements" failure mode that
breaks Streamlit Cloud deploys.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCAL_PACKAGES = {"app", "charts", "planner", "scripts", "tests"}

# Runtime-only sources — what Streamlit Cloud actually loads. Tests and the
# offline `scripts/build_historical_csv.py` use dev-only deps (pytest, xlrd)
# and are intentionally out of scope for the runtime requirements check.
SOURCES = [
    REPO / "app.py",
    REPO / "charts.py",
    *(REPO / "planner").glob("*.py"),
]


def collect_top_level_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — local
                continue
            if node.module:
                out.add(node.module.split(".")[0])
    return out


def parse_requirement_names(req_text: str) -> set[str]:
    names: set[str] = set()
    for line in req_text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"[A-Za-z0-9_.\-]+", line)
        if m:
            names.add(m.group(0).lower().replace("-", "_"))
    return names


def main() -> int:
    used: set[str] = set()
    for src in SOURCES:
        if src.exists():
            used |= collect_top_level_modules(src)

    third_party = {
        m for m in used
        if m not in sys.stdlib_module_names
        and m not in LOCAL_PACKAGES
        and not m.startswith("_")
    }

    declared = parse_requirement_names((REPO / "requirements.txt").read_text())

    missing = sorted(m for m in third_party if m.lower().replace("-", "_") not in declared)
    if missing:
        print("Missing from requirements.txt:", ", ".join(missing), file=sys.stderr)
        return 1
    print(f"OK: {len(third_party)} third-party imports all declared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
