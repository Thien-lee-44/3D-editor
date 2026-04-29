"""
Application Entry Point.
Configures the environment and bootstraps the main application process.
"""

import os
import sys
from pathlib import Path

def setup_environment() -> None:
    """
    Resolves the project root and anchors the working directory to ensure
    consistent module resolution across different execution contexts.
    """
    root_dir = Path(__file__).resolve().parent
    os.chdir(str(root_dir))
    
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

if __name__ == "__main__":
    setup_environment()
    
    from src.app.main import run_app
    run_app()