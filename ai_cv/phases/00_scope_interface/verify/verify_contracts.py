"""Verify all Phase 00 contracts, examples and cross-field invariants."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = ROOT / "shared" / "contracts"
sys.path.insert(0, str(CONTRACTS))

from validate_contracts import validate_examples  # noqa: E402


def main() -> int:
    validate_examples()
    print("Phase 00 contract schema and semantic verification: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
