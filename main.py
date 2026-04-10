import sys
import os

# Ensure the root directory is in sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import the main app from app.py
from app import main

if __name__ == "__main__":
    main()
