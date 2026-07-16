"""Test checkout sources rather than a previously installed wheel."""

from pathlib import Path
import sys


ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
