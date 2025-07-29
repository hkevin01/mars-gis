#!/usr/bin/env python3
"""Execute the documentation validation and display results."""

import os
import subprocess
import sys


def main():
    # Change to project directory
    os.chdir('/home/kevin/Projects/mars-gis')
    
    print("🚀 Executing MARS-GIS Documentation Compliance Validation...")
    print("=" * 70)
    
    try:
        # Run the validation script
        result = subprocess.run([
            sys.executable, 'validate_documentation.py'
        ], capture_output=False, text=True)
        
        print(f"\n{'=' * 70}")
        print(f"Validation completed with exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("🎉 SUCCESS: Documentation compliance validation passed!")
        else:
            print("⚠️  WARNING: Some documentation compliance issues found")
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ ERROR: Failed to run validation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
