#!/usr/bin/env python3
"""
Mars GIS Test Runner
Comprehensive test execution script with multiple test configurations.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Mars GIS Test Runner")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--api", action="store_true", help="Run API tests only")
    parser.add_argument("--ml", action="store_true", help="Run ML tests only")
    parser.add_argument("--geospatial", action="store_true", help="Run geospatial tests only")
    parser.add_argument("--data", action="store_true", help="Run data processing tests only")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only (exclude slow)")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies first")
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(Path.cwd())
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        if not run_command("pip install -r requirements.txt", "Installing requirements"):
            print("Failed to install dependencies")
            return 1
    
    # Build base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        pytest_cmd.append("-v")
    
    # Add parallel execution
    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])
    
    # Add fail-fast
    if args.fail_fast:
        pytest_cmd.append("-x")
    
    # Add coverage if requested
    if args.coverage or args.all:
        pytest_cmd.extend([
            "--cov=mars_gis",
            "--cov-branch",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    # Determine which tests to run
    test_markers = []
    test_paths = []
    
    if args.unit:
        test_markers.append("unit")
    elif args.integration:
        test_markers.append("integration")
    elif args.api:
        test_markers.append("api")
    elif args.ml:
        test_markers.append("ml")
    elif args.geospatial:
        test_markers.append("geospatial")
    elif args.data:
        test_markers.append("data_processing")
    elif args.fast:
        test_markers.append("not slow")
    elif args.all:
        # Run all tests
        pass
    else:
        # Default: run unit and fast integration tests
        test_markers.append("not slow")
    
    # Add marker filters
    if test_markers:
        pytest_cmd.extend(["-m", " and ".join(test_markers)])
    
    # Add test paths
    pytest_cmd.append("tests/")
    
    # Run the tests
    cmd_str = " ".join(pytest_cmd)
    print(f"Executing: {cmd_str}")
    
    success = run_command(cmd_str, "Running pytest")
    
    if not success:
        print("\n❌ Tests failed!")
        return 1
    
    print("\n✅ All tests passed!")
    
    # Generate additional reports if coverage was run
    if args.coverage or args.all:
        print("\n📊 Coverage report generated in htmlcov/index.html")
        
        # Try to open coverage report (if on a system with a browser)
        try:
            import webbrowser
            webbrowser.open('htmlcov/index.html')
            print("Coverage report opened in browser")
        except:
            pass
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
