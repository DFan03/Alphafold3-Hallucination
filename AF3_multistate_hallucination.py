#!/usr/bin/env python3
"""Top-level AFhall-style entry point delegating to `af3_hall.cli:main`."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from af3_hall.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
