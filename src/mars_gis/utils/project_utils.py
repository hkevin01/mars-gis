"""
Mars GIS Utilities

Common utility functions for data processing, coordinate transformations,
and file management.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def ensure_directory_structure():
    """Ensure proper Mars GIS directory structure exists."""
    base_path = Path(__file__).parent.parent.parent.parent
    
    required_dirs = [
        "src/mars_gis/models",
        "src/mars_gis/visualization", 
        "src/mars_gis/utils",
        "src/mars_gis/data",
        "src/mars_gis/database",
        "tests",
        "scripts",
        "docs",
        "data/mars",
        "data/earth",
        "logs"
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py for Python packages
        if dir_path.startswith("src/mars_gis") and not dir_path.endswith(("data", "logs")):
            init_file = full_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Mars GIS module."""\n')


def validate_import_paths():
    """Validate that all import paths are correct after reorganization."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        # Test foundation model imports
        from mars_gis.models.comparative import ComparativePlanetaryAnalyzer
        from mars_gis.models.foundation import MarsEarthTransferModel
        from mars_gis.models.multimodal import MultiModalMarsProcessor
        from mars_gis.models.optimization import MarsLandingSiteOptimizer
        from mars_gis.models.planetary_scale import PlanetaryScaleEmbeddingGenerator
        from mars_gis.models.self_supervised import SelfSupervisedMarsLearning
        from mars_gis.visualization.analysis_dashboard import MarsAnalysisDashboard
        from mars_gis.visualization.interactive_mapping import InteractiveMarsMap

        # Test visualization imports
        from mars_gis.visualization.mars_3d_globe import Mars3DGlobeGenerator
        
        print("✅ All import paths validated successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import validation failed: {e}")
        return False


def get_project_statistics():
    """Get comprehensive project statistics."""
    base_path = Path(__file__).parent.parent.parent.parent
    
    stats = {
        'foundation_models': 0,
        'visualization_modules': 0,
        'test_files': 0,
        'total_python_files': 0,
        'total_lines_of_code': 0,
        'documentation_files': 0
    }
    
    # Count foundation models
    models_path = base_path / "src/mars_gis/models"
    if models_path.exists():
        stats['foundation_models'] = len([f for f in models_path.glob("*.py") 
                                        if f.name != "__init__.py"])
    
    # Count visualization modules
    viz_path = base_path / "src/mars_gis/visualization"
    if viz_path.exists():
        stats['visualization_modules'] = len([f for f in viz_path.glob("*.py") 
                                            if f.name != "__init__.py"])
    
    # Count test files
    tests_path = base_path / "tests"
    if tests_path.exists():
        stats['test_files'] = len(list(tests_path.glob("test_*.py")))
    
    # Count all Python files and lines
    for py_file in base_path.rglob("*.py"):
        if not any(skip in str(py_file) for skip in ['.venv', '__pycache__', '.git']):
            stats['total_python_files'] += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    stats['total_lines_of_code'] += sum(1 for _ in f)
            except:
                pass
    
    # Count documentation files
    for doc_file in base_path.rglob("*.md"):
        if not any(skip in str(doc_file) for skip in ['.git', 'node_modules']):
            stats['documentation_files'] += 1
    
    return stats


def create_missing_init_files():
    """Create missing __init__.py files in Python packages."""
    base_path = Path(__file__).parent.parent.parent.parent
    src_path = base_path / "src"
    
    created_files = []
    
    for dir_path in src_path.rglob("*"):
        if dir_path.is_dir() and dir_path.name != "__pycache__":
            # Check if it's a Python package (contains .py files)
            has_py_files = any(dir_path.glob("*.py"))
            init_file = dir_path / "__init__.py"
            
            if has_py_files and not init_file.exists():
                # Create appropriate __init__.py content
                module_name = dir_path.name.replace("_", " ").title()
                content = f'"""{module_name} module."""\n'
                
                init_file.write_text(content)
                created_files.append(str(init_file.relative_to(base_path)))
    
    return created_files


def cleanup_unused_files():
    """Identify and report unused files (without actually deleting them)."""
    base_path = Path(__file__).parent.parent.parent.parent
    
    potentially_unused = []
    
    # Check for empty directories
    for dir_path in base_path.rglob("*"):
        if (dir_path.is_dir() and 
            dir_path.name not in ['.git', '__pycache__', '.pytest_cache'] and
            not any(dir_path.iterdir())):
            potentially_unused.append(f"Empty directory: {dir_path.relative_to(base_path)}")
    
    # Check for duplicate script files
    scripts_path = base_path / "scripts"
    if scripts_path.exists():
        script_files = list(scripts_path.rglob("*.py"))
        script_names = [f.stem for f in script_files]
        
        # Find potential duplicates
        from collections import Counter
        name_counts = Counter(script_names)
        for name, count in name_counts.items():
            if count > 1:
                matching_files = [f for f in script_files if f.stem == name]
                potentially_unused.append(f"Potential duplicate scripts: {[str(f.relative_to(base_path)) for f in matching_files]}")
    
    return potentially_unused


if __name__ == "__main__":
    print("🔧 Mars GIS Utility Functions")
    print("=" * 40)
    
    # Ensure directory structure
    print("📁 Ensuring directory structure...")
    ensure_directory_structure()
    
    # Create missing init files
    print("📝 Creating missing __init__.py files...")
    created_files = create_missing_init_files()
    if created_files:
        print(f"   Created {len(created_files)} __init__.py files")
        for file in created_files:
            print(f"   - {file}")
    else:
        print("   All __init__.py files already exist")
    
    # Validate imports
    print("🔍 Validating import paths...")
    validate_import_paths()
    
    # Get project statistics
    print("📊 Project Statistics:")
    stats = get_project_statistics()
    for key, value in stats.items():
        formatted_key = key.replace('_', ' ').title()
        print(f"   {formatted_key}: {value}")
    
    # Check for unused files
    print("🧹 Checking for potentially unused files...")
    unused = cleanup_unused_files()
    if unused:
        print(f"   Found {len(unused)} potential issues:")
        for issue in unused:
            print(f"   - {issue}")
    else:
        print("   No obvious unused files found")
    
    print("\n✅ Mars GIS utility check completed!")
