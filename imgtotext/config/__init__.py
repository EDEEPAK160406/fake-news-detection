"""
Configuration package initialization.
"""

import sys
from pathlib import Path

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import default config
from config import *
