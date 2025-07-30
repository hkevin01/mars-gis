#!/bin/bash
# Remove test files from root that were already moved to tests/
rm -f test_project_completion.py
rm -f test_core_config_comprehensive.py
rm -f test_documentation_compliance.py
rm -f test_end_to_end_workflows.py  
rm -f test_integration_comprehensive.py

# Move remaining test files to tests/
mv comprehensive_test_suite.py tests/
mv final_comprehensive_test.py tests/ 

# Move validation scripts to scripts/validation/
mkdir -p scripts/validation
mv direct_validation.py scripts/validation/
mv run_validation.py scripts/validation/
mv execute_final_validation.py scripts/validation/
mv validate_documentation.py scripts/validation/
mv direct_audit_execution.py scripts/validation/
mv execute_completion_audit.py scripts/validation/
mv run_completion_audit.py scripts/validation/

# Move test runner scripts to scripts/testing/
mkdir -p scripts/testing
mv execute_tests.py scripts/testing/
mv run_comprehensive_tests.py scripts/testing/
mv run_tests.py scripts/testing/
mv quick_test_runner.py scripts/testing/

# Move notebook to notebooks/ 
mv comprehensive_test_generation.ipynb notebooks/

echo "File cleanup and reorganization complete!"
