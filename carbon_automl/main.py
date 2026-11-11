"""Package entry point delegating to the root CLI module."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Run the Carbon-Aware AutoML CLI."""
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from main import main as root_main

    root_main()


if __name__ == "__main__":
    main()
