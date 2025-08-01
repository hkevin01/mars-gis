#!/usr/bin/env python3
"""
MARS-GIS Project Validation Script
Validates the reorganized project structure and functionality.
"""

import sys
from pathlib import Path


def validate_project_structure():
    """Validate that essential files and directories exist."""
    print("🔍 Validating project structure...")

    essential_files = [
        "src/mars_gis/__init__.py",
        "src/mars_gis/main.py",
        "src/mars_gis/core/config.py",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
        "docker/docker-helper.sh",
        "docker/compose/docker-compose.yml"
    ]

    missing = []
    for file_path in essential_files:
        if not Path(file_path).exists():
            missing.append(file_path)
        else:
            print(f"  ✅ {file_path}")

    if missing:
        print(f"  ❌ Missing files: {missing}")
        return False

    print("  ✅ All essential files present")
    return True

def validate_imports():
    """Validate that core imports work."""
    print("\n🐍 Validating Python imports...")

    try:
        sys.path.insert(0, 'src')

        # Test core config import
        from mars_gis.core.config import Settings
        print("  ✅ Core config imports successfully")

        # Test main package import
        import mars_gis
        print(f"  ✅ Main package imports successfully (v{mars_gis.__version__})")

        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False

def validate_docker_structure():
    """Validate Docker organization."""
    print("\n🐳 Validating Docker structure...")

    docker_files = [
        "docker/backend/Dockerfile",
        "docker/frontend/Dockerfile",
        "docker/compose/docker-compose.yml",
        "docker/docker-helper.sh"
    ]

    missing = []
    for file_path in docker_files:
        if not Path(file_path).exists():
            missing.append(file_path)
        else:
            print(f"  ✅ {file_path}")

    if missing:
        print(f"  ❌ Missing Docker files: {missing}")
        return False

    print("  ✅ Docker structure is properly organized")
    return True

def count_cleanup_files():
    """Count any remaining cleanup files."""
    print("\n🧹 Checking for cleanup files...")

    patterns = ["*cleanup*", "*backup*", "*final*", "*deprecated*", "*migration*"]
    found = []

    for pattern in patterns:
        for file_path in Path(".").rglob(pattern):
            if file_path.is_file() and not file_path.name.startswith('.'):
                found.append(str(file_path))

    if found:
        print(f"  ⚠️  Found {len(found)} cleanup files still present:")
        for f in found[:5]:  # Show first 5
            print(f"    - {f}")
        if len(found) > 5:
            print(f"    ... and {len(found) - 5} more")
        return False
    else:
        print("  ✅ No cleanup files found - project is clean")
        return True

def main():
    """Run complete project validation."""
    print("🚀 MARS-GIS Project Validation")
    print("=" * 50)

    checks = [
        ("Project Structure", validate_project_structure),
        ("Python Imports", validate_imports),
        ("Docker Organization", validate_docker_structure),
        ("Cleanup Status", count_cleanup_files)
    ]

    passed = 0
    total = len(checks)

    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {check_name} validation failed: {e}")

    print(f"\n📊 Validation Results: {passed}/{total} checks passed")

    if passed == total:
        print("✅ Project reorganization validation PASSED")
        print("🎉 The MARS-GIS project is properly organized and ready for development!")
        return True
    else:
        print("❌ Some validation checks failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
