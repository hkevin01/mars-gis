#!/bin/bash

# Mars-GIS Root Directory Cleanup Script
# This script organizes files from the root directory into appropriate subfolders

echo "🧹 Starting Mars-GIS root directory cleanup..."

# Move project reports to docs/reports/
echo "📋 Moving project reports..."
mv PROJECT_COMPLETION_REPORT.md docs/reports/
mv PROJECT_EXECUTIVE_SUMMARY.md docs/reports/
mv PROJECT_PROGRESS_TRACKER.md docs/reports/
mv COMPLETION_AUDIT_SUMMARY.md docs/reports/
mv PROJECT_COMPLETE.md docs/reports/
mv PROGRESS_TRACKER_COMPLETE.md docs/reports/

# Move documentation files to docs/
echo "📚 Moving documentation files..."
mv README_COMPLETE.md docs/
mv TESTING.md docs/
mv TESTING_COMPLETE.md docs/

# Move validation scripts to scripts/validation/
echo "🔍 Moving validation scripts..."
mv direct_validation.py scripts/validation/
mv run_validation.py scripts/validation/
mv execute_final_validation.py scripts/validation/
mv validate_documentation.py scripts/validation/
mv direct_audit_execution.py scripts/validation/
mv execute_completion_audit.py scripts/validation/
mv run_completion_audit.py scripts/validation/

# Move test files to tests/
echo "🧪 Moving test files..."
mv test_core_config_comprehensive.py tests/
mv test_documentation_compliance.py tests/
mv test_end_to_end_workflows.py tests/
mv test_integration_comprehensive.py tests/
mv test_project_completion.py tests/
mv comprehensive_test_suite.py tests/

# Move test runner scripts to scripts/testing/
echo "🏃 Moving test runner scripts..."
mv execute_tests.py scripts/testing/
mv final_comprehensive_test.py scripts/testing/
mv run_comprehensive_tests.py scripts/testing/
mv run_tests.py scripts/testing/
mv quick_test_runner.py scripts/testing/

# Move notebook to notebooks/
echo "📓 Moving notebook files..."
mv comprehensive_test_generation.ipynb notebooks/

# Move deployment scripts to scripts/
echo "🚀 Moving deployment scripts..."
mv deploy.sh scripts/
mv setup_dev.sh scripts/

echo "✅ Root directory cleanup complete!"
echo "📁 Files have been organized into:"
echo "   📋 docs/reports/ - Project reports and summaries"
echo "   📚 docs/ - Additional documentation"
echo "   🔍 scripts/validation/ - Validation and audit scripts"
echo "   🧪 tests/ - Test files"
echo "   🏃 scripts/testing/ - Test runner scripts"
echo "   📓 notebooks/ - Jupyter notebooks"
echo "   🚀 scripts/ - Deployment and setup scripts"
