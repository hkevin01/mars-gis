#!/usr/bin/env python3
"""Setup script for MARS-GIS development environment."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, cwd: Path = None) -> bool:
    """Run a shell command and return success status."""
    try:
        subprocess.run(
            command.split(),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}: {e.stderr}")
        return False


def main():
    """Set up the development environment."""
    project_root = Path(__file__).parent.parent
    
    print("Setting up MARS-GIS development environment...")
    
    # Create virtual environment if it doesn't exist
    venv_path = project_root / "venv"
    if not venv_path.exists():
        print("Creating virtual environment...")
        if not run_command(f"{sys.executable} -m venv venv", project_root):
            sys.exit(1)
    
    # Determine pip path
    if os.name == 'nt':  # Windows
        pip_path = venv_path / "Scripts" / "pip"
    else:  # Unix-like
        pip_path = venv_path / "bin" / "pip"
    
    # Upgrade pip
    if not run_command(f"{pip_path} install --upgrade pip", project_root):
        sys.exit(1)
    
    # Install dependencies
    requirements_file = project_root / "requirements.txt"
    if requirements_file.exists():
        if not run_command(
            f"{pip_path} install -r requirements.txt",
            project_root
        ):
            print("Warning: Some dependencies failed to install")
    
    # Install package in development mode
    if not run_command(f"{pip_path} install -e .", project_root):
        print("Warning: Package installation failed")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "logs",
        "assets/maps",
        "assets/images",
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("\n🚀 Environment setup complete!")
    print("To activate the virtual environment:")
    if os.name == 'nt':
        print("  venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")


if __name__ == "__main__":
    main()
