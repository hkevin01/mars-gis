import os
import subprocess
import sys

# Change to the project directory
os.chdir('/home/kevin/Projects/mars-gis')

# Execute the comprehensive test runner
result = subprocess.run([sys.executable, 'run_comprehensive_tests.py'], 
                       capture_output=True, text=True)

print("=== COMPREHENSIVE TEST EXECUTION RESULTS ===")
print(f"Return code: {result.returncode}")
print(f"\nSTDOUT:\n{result.stdout}")
print(f"\nSTDERR:\n{result.stderr}")
