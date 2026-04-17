"""Train the domain classifier on the prebuilt embeddings.

Usage:
    python -m scripts.train_classifier
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classifier import train  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    report = train()
    print(f"\nAccuracy: {report['accuracy']:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    out = ROOT / "data" / "models" / "classification_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
