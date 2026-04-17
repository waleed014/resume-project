"""Print a per-category file count summary across the configured raw sources."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import summarize  # noqa: E402


def main() -> int:
    summary = summarize()
    total = sum(summary.values())
    print(f"{'Category':<35} {'Count':>8}")
    print("-" * 45)
    for cat in sorted(summary):
        print(f"{cat:<35} {summary[cat]:>8}")
    print("-" * 45)
    print(f"{'TOTAL':<35} {total:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
