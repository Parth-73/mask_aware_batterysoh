"""Isolated research package; optional dependencies live in a workspace-local vendor folder."""
from pathlib import Path
import sys
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
vendor=Path(__file__).parent/'vendor'
if vendor.exists():sys.path.insert(0,str(vendor))
