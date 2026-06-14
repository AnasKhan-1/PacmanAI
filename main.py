"""
Pacman AI Agents Platform — root entry point.
Delegates to pacman_ai.main.main().
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pacman_ai.main import main

if __name__ == "__main__":
    main()
