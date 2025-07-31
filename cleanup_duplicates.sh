#!/bin/bash

# Mars GIS Cleanup Script - Remove duplicate and unused files

echo "🧹 Starting Mars GIS cleanup process..."

# Remove duplicate visualization files (keep the newer comprehensive ones)
echo "Removing duplicate visualization files..."
rm -f /home/kevin/Projects/mars-gis/src/mars_gis/visualization/mars_3d.py
rm -f /home/kevin/Projects/mars-gis/src/mars_gis/visualization/interactive_map.py

# Remove old foundation model files in ml directory
echo "Removing old foundation model files..."
rm -rf /home/kevin/Projects/mars-gis/src/mars_gis/ml/foundation_models/

# Remove excessive testing scripts
echo "Cleaning up excessive testing scripts..."
cd /home/kevin/Projects/mars-gis/scripts/testing/
# Keep only the essential ones
mv run_tests.py essential_run_tests.py 2>/dev/null || true
mv run_comprehensive_tests.py essential_comprehensive_tests.py 2>/dev/null || true
rm -f execute_tests.py
rm -f final_comprehensive_test.py
rm -f quick_test_runner.py
mv essential_run_tests.py run_tests.py 2>/dev/null || true
mv essential_comprehensive_tests.py run_comprehensive_tests.py 2>/dev/null || true

# Clean up excessive validation scripts
echo "Cleaning up excessive validation scripts..."
cd /home/kevin/Projects/mars-gis/scripts/validation/
# Keep only the essential ones
mv run_validation.py essential_validation.py 2>/dev/null || true
rm -f direct_audit_execution.py
rm -f direct_validation.py
rm -f execute_completion_audit.py
rm -f execute_final_validation.py
rm -f run_completion_audit.py
mv essential_validation.py run_validation.py 2>/dev/null || true

# Remove duplicate test files
echo "Cleaning up duplicate test files..."
cd /home/kevin/Projects/mars-gis/tests/
rm -f final_comprehensive_test.py
rm -f test_integration_comprehensive.py
rm -f test_core_config_comprehensive.py

echo "✅ Mars GIS cleanup completed!"
echo "📊 Kept essential files:"
echo "  - Foundation models: src/mars_gis/models/"
echo "  - Visualizations: mars_3d_globe.py, interactive_mapping.py, analysis_dashboard.py"
echo "  - Essential tests: test_integration.py, comprehensive_test_suite.py"
echo "  - Core scripts: run_tests.py, run_comprehensive_tests.py, run_validation.py"
