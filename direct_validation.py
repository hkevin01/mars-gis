import os
import subprocess
import sys

# Set environment
os.chdir('/home/kevin/Projects/mars-gis')

# Execute validation directly
print("🚀 Executing MARS-GIS Documentation Compliance Validation")
print("=" * 70)

try:
    # Import and run validation directly
    sys.path.insert(0, '/home/kevin/Projects/mars-gis')
    from validate_documentation import DocumentationValidator
    
    validator = DocumentationValidator()
    results = validator.run_all_validations()
    report = validator.generate_report()
    
    print(report)
    
    # Calculate success metrics
    total_tests = sum(r['total'] for r in results)
    total_passed = sum(r['passed'] for r in results)
    success_rate = (total_passed / total_tests) if total_tests > 0 else 0
    
    print(f"\n🎯 FINAL VALIDATION RESULTS:")
    print(f"   Documentation Compliance: {success_rate*100:.1f}%")
    print(f"   Validation Points: {total_passed}/{total_tests}")
    
    if success_rate >= 0.8:
        print("🎉 SUCCESS: Documentation validation PASSED!")
        exit_code = 0
    else:
        print("⚠️  WARNING: Documentation validation needs attention")
        exit_code = 1
        
except Exception as e:
    print(f"❌ ERROR: Validation failed with error: {e}")
    import traceback
    traceback.print_exc()
    exit_code = 1

print(f"\nValidation completed with exit code: {exit_code}")
