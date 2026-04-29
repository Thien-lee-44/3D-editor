"""
Application Entry Point.

Configures the working directory and launches the application using the 'src' namespace.
Ensures consistent module resolution across different execution environments.
"""

import os
import sys
from pathlib import Path


def setup_environment() -> None:
    """
    Anchors the working directory to the project root and injects it into sys.path.
    """
    root_dir = Path(__file__).resolve().parent
    os.chdir(root_dir)
    
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))


if __name__ == "__main__":
    setup_environment()
    from src.app.main import run_app
    run_app()