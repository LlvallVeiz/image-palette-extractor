#!/usr/bin/env python3
"""Entry point: extract dominant colours from an image.

    python3 palette.py sample --out examples/sample.png
    python3 palette.py extract examples/sample.png --k 5
    python3 palette.py extract examples/sample.png --k 5 --space lab

The implementation lives in ``src/palette/``; this shim just puts it on
the import path.  Requires numpy and Pillow (see requirements.txt).
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
)

from palette.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
