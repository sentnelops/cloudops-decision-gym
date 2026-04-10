import sys
import os

# Ensure src directory is in sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Import the main app from the src/app.py which is now found correctly 
# as this file is no longer named app.py
from app import main

if __name__ == "__main__":
    main()
