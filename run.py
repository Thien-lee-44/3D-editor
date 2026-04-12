"""
Application Entry Point.
Configures the working directory and launches the application using the 'src' namespace.
"""
import os
import sys
from pathlib import Path

def setup_environment() -> None:
    """
    Anchors the working directory to the project root and ensures the root 
    is in the Python path to resolve module imports consistently.
    """
    root_dir = Path(__file__).resolve().parent
    os.chdir(str(root_dir))
    
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

if __name__ == "__main__":
    setup_environment()
    
    # Standard Python 3 absolute import to prevent Module Aliasing
    from src.app.main import run_app
    
    run_app()