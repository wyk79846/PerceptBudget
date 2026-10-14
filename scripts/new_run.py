#!/usr/bin/env python3
"""Create an external PerceptBudget run directory without starting an experiment."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perceptbudget.experiment import main


if __name__ == "__main__":
    raise SystemExit(main())
