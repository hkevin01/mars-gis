"""
Mars GIS Project Validation Report
Generated after comprehensive code reorganization
"""

import sys
from pathlib import Path


def validate_project_structure():
    """Validate the reorganized project structure."""
    base_path = Path(__file__).parent.parent
    
    print("🔍 Mars GIS Project Structure Validation")
    print("=" * 50)
    
    # Check foundation models structure
    models_path = base_path / "src/mars_gis/models"
    expected_models = [
        "foundation.py",
        "multimodal.py", 
        "comparative.py",
        "optimization.py",
        "self_supervised.py",
        "planetary_scale.py"
    ]
    
    print("\n📦 Foundation Models Structure:")
    missing_models = []
    for model in expected_models:
        model_file = models_path / model
        if model_file.exists():
            line_count = len(model_file.read_text().splitlines())
            print(f"  ✅ {model} ({line_count} lines)")
        else:
            print(f"  ❌ {model} (MISSING)")
            missing_models.append(model)
    
    # Check visualization structure
    viz_path = base_path / "src/mars_gis/visualization"
    expected_viz = [
        "mars_3d_globe.py",
        "interactive_mapping.py",
        "analysis_dashboard.py"
    ]
    
    print("\n🎨 Visualization Components:")
    missing_viz = []
    for viz in expected_viz:
        viz_file = viz_path / viz
        if viz_file.exists():
            line_count = len(viz_file.read_text().splitlines())
            print(f"  ✅ {viz} ({line_count} lines)")
        else:
            print(f"  ❌ {viz} (MISSING)")
            missing_viz.append(viz)
    
    # Check for removed duplicate files
    print("\n🧹 Cleanup Verification:")
    old_files = [
        "src/mars_gis/visualization/mars_3d.py",
        "src/mars_gis/visualization/interactive_map.py",
        "src/mars_gis/ml/foundation_models"
    ]
    
    for old_file in old_files:
        old_path = base_path / old_file
        if old_path.exists():
            print(f"  ⚠️  {old_file} (Should be removed)")
        else:
            print(f"  ✅ {old_file} (Successfully removed)")
    
    # Check __init__.py files
    print("\n📋 Package Init Files:")
    init_files = [
        "src/mars_gis/__init__.py",
        "src/mars_gis/models/__init__.py", 
        "src/mars_gis/visualization/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = base_path / init_file
        if init_path.exists():
            print(f"  ✅ {init_file}")
        else:
            print(f"  ❌ {init_file} (MISSING)")
    
    # Summary
    total_issues = len(missing_models) + len(missing_viz)
    if total_issues == 0:
        print("\n🎉 PROJECT STRUCTURE VALIDATION: PASSED")
        print("   All components properly organized and accessible")
    else:
        print(f"\n⚠️  PROJECT STRUCTURE VALIDATION: {total_issues} ISSUES FOUND")
    
    return total_issues == 0

if __name__ == "__main__":
    success = validate_project_structure()
    sys.exit(0 if success else 1)
