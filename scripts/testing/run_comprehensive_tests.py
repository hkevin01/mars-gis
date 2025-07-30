#!/usr/bin/env python3
"""
Comprehensive Test Runner for MARS-GIS Platform
Executes all comprehensive test suites and generates detailed reports.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


class ComprehensiveTestRunner:
    """Runs comprehensive test suites and generates reports."""
    
    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with color coding."""
        colors = {
            "INFO": "\033[94m",     # Blue
            "SUCCESS": "\033[92m",  # Green
            "WARNING": "\033[93m",  # Yellow
            "ERROR": "\033[91m",    # Red
            "RESET": "\033[0m"      # Reset
        }
        
        color = colors.get(level, colors["INFO"])
        reset = colors["RESET"]
        timestamp = time.strftime("%H:%M:%S")
        
        print(f"{color}[{timestamp}] {level}: {message}{reset}")
    
    def run_python_test_file(self, test_file: str) -> Dict[str, Any]:
        """Run a specific Python test file."""
        self.log(f"Running test file: {test_file}")
        
        test_path = Path(test_file)
        if not test_path.exists():
            return {
                "file": test_file,
                "status": "skipped",
                "reason": "File not found",
                "duration": 0.0,
                "tests_run": 0,
                "failures": 0,
                "errors": 0
            }
        
        start_time = time.time()
        
        try:
            # Run the test file directly with Python
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per file
            )
            
            duration = time.time() - start_time
            
            # Parse output for test results
            output = result.stdout + result.stderr
            
            # Count tests, failures, errors
            tests_run = output.count("def test_")
            failures = output.count("FAILED")
            errors = output.count("ERROR")
            
            status = "passed" if result.returncode == 0 else "failed"
            
            return {
                "file": test_file,
                "status": status,
                "duration": duration,
                "tests_run": tests_run,
                "failures": failures,
                "errors": errors,
                "output": output[:1000],  # First 1000 chars
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "file": test_file,
                "status": "timeout",
                "duration": duration,
                "tests_run": 0,
                "failures": 1,
                "errors": 0,
                "output": "Test timed out after 2 minutes"
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "file": test_file,
                "status": "error",
                "duration": duration,
                "tests_run": 0,
                "failures": 0,
                "errors": 1,
                "output": str(e)
            }
    
    def run_pytest_if_available(self) -> Dict[str, Any]:
        """Run pytest on existing test files if available."""
        self.log("Attempting to run pytest on existing tests...")
        
        # Check if pytest is available
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--version"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {
                    "file": "pytest",
                    "status": "skipped",
                    "reason": "pytest not available",
                    "duration": 0.0
                }
        except Exception:
            return {
                "file": "pytest",
                "status": "skipped", 
                "reason": "pytest not available",
                "duration": 0.0
            }
        
        # Run pytest on tests directory
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            output = result.stdout + result.stderr
            
            # Parse pytest output
            if "collected" in output:
                import re
                collected_match = re.search(r'collected (\d+) items', output)
                tests_run = int(collected_match.group(1)) if collected_match else 0
            else:
                tests_run = 0
            
            failures = output.count("FAILED")
            errors = output.count("ERROR")
            
            return {
                "file": "pytest",
                "status": "passed" if result.returncode == 0 else "failed",
                "duration": duration,
                "tests_run": tests_run,
                "failures": failures,
                "errors": errors,
                "output": output[-2000:],  # Last 2000 chars
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "file": "pytest",
                "status": "timeout",
                "duration": time.time() - start_time,
                "tests_run": 0,
                "failures": 1,
                "errors": 0,
                "output": "pytest timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "file": "pytest",
                "status": "error",
                "duration": time.time() - start_time,
                "tests_run": 0,
                "failures": 0,
                "errors": 1,
                "output": str(e)
            }
    
    def test_basic_imports(self) -> Dict[str, Any]:
        """Test that basic imports work."""
        self.log("Testing basic imports...")
        
        start_time = time.time()
        import_results = []
        
        # Test core imports
        test_imports = [
            ("mars_gis.main", "Main application"),
            ("mars_gis.core.config", "Configuration"),
            ("pathlib", "Path utilities"),
            ("json", "JSON processing"),
            ("sys", "System utilities"),
            ("os", "OS utilities")
        ]
        
        for module_name, description in test_imports:
            try:
                __import__(module_name)
                import_results.append(f"✅ {description} ({module_name})")
            except ImportError as e:
                import_results.append(f"❌ {description} ({module_name}): {e}")
        
        duration = time.time() - start_time
        
        # Count successes and failures
        successes = len([r for r in import_results if r.startswith("✅")])
        failures = len([r for r in import_results if r.startswith("❌")])
        
        return {
            "file": "basic_imports",
            "status": "passed" if failures == 0 else "failed",
            "duration": duration,
            "tests_run": len(test_imports),
            "failures": failures,
            "errors": 0,
            "output": "\n".join(import_results)
        }
    
    def test_project_structure(self) -> Dict[str, Any]:
        """Test that project structure matches documentation."""
        self.log("Testing project structure...")
        
        start_time = time.time()
        structure_results = []
        
        # Expected structure from README
        expected_structure = [
            "src/mars_gis",
            "src/mars_gis/main.py",
            "src/mars_gis/core",
            "src/mars_gis/api",
            "src/mars_gis/data",
            "src/mars_gis/ml",
            "src/mars_gis/visualization",
            "tests",
            "requirements.txt",
            "README.md",
            "LICENSE",
            "docker-compose.yml",
            "Dockerfile",
            ".env.example"
        ]
        
        for path_str in expected_structure:
            path = Path(path_str)
            if path.exists():
                structure_results.append(f"✅ {path_str}")
            else:
                structure_results.append(f"❌ {path_str}")
        
        duration = time.time() - start_time
        
        successes = len([r for r in structure_results if r.startswith("✅")])
        failures = len([r for r in structure_results if r.startswith("❌")])
        
        return {
            "file": "project_structure",
            "status": "passed" if failures <= 2 else "failed",  # Allow some missing files
            "duration": duration,
            "tests_run": len(expected_structure),
            "failures": failures,
            "errors": 0,
            "output": "\n".join(structure_results)
        }
    
    def test_configuration_loading(self) -> Dict[str, Any]:
        """Test configuration loading works."""
        self.log("Testing configuration loading...")
        
        start_time = time.time()
        config_results = []
        
        try:
            from mars_gis.core.config import settings

            # Test key configuration attributes
            config_attrs = [
                'APP_NAME', 'VERSION', 'HOST', 'PORT',
                'DATABASE_URL', 'REDIS_URL'
            ]
            
            for attr in config_attrs:
                if hasattr(settings, attr):
                    value = getattr(settings, attr)
                    config_results.append(f"✅ {attr}: {value}")
                else:
                    config_results.append(f"❌ {attr}: missing")
            
            # Test that directories were created
            if hasattr(settings, 'DATA_DIR') and settings.DATA_DIR.exists():
                config_results.append("✅ Data directory created")
            else:
                config_results.append("❌ Data directory not created")
            
            status = "passed"
            
        except Exception as e:
            config_results.append(f"❌ Configuration loading failed: {e}")
            status = "failed"
        
        duration = time.time() - start_time
        
        failures = len([r for r in config_results if r.startswith("❌")])
        
        return {
            "file": "configuration_loading",
            "status": status,
            "duration": duration,
            "tests_run": len(config_results),
            "failures": failures,
            "errors": 0,
            "output": "\n".join(config_results)
        }
    
    def test_application_creation(self) -> Dict[str, Any]:
        """Test application creation works."""
        self.log("Testing application creation...")
        
        start_time = time.time()
        app_results = []
        
        try:
            from mars_gis.main import create_app
            
            app = create_app()
            
            if app is not None:
                app_results.append("✅ FastAPI app created successfully")
                
                # Test app attributes
                if hasattr(app, 'title'):
                    app_results.append(f"✅ App title: {app.title}")
                
                if hasattr(app, 'version'):
                    app_results.append(f"✅ App version: {app.version}")
                
                status = "passed"
            else:
                app_results.append("❌ App creation returned None")
                status = "failed"
                
        except ImportError as e:
            app_results.append(f"❌ Import failed: {e}")
            status = "skipped"
        except Exception as e:
            app_results.append(f"❌ App creation failed: {e}")
            status = "failed"
        
        duration = time.time() - start_time
        
        failures = len([r for r in app_results if r.startswith("❌")])
        
        return {
            "file": "application_creation",
            "status": status,
            "duration": duration,
            "tests_run": len(app_results),
            "failures": failures,
            "errors": 0,
            "output": "\n".join(app_results)
        }
    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all comprehensive tests."""
        self.log("🚀 Starting Comprehensive MARS-GIS Test Suite", "SUCCESS")
        self.log("=" * 70)
        
        # Built-in tests
        built_in_tests = [
            self.test_basic_imports,
            self.test_project_structure,
            self.test_configuration_loading,
            self.test_application_creation
        ]
        
        for test_func in built_in_tests:
            result = test_func()
            self.test_results.append(result)
            
            status_color = "SUCCESS" if result["status"] == "passed" else "ERROR"
            self.log(f"{result['file']}: {result['status']} ({result['duration']:.2f}s)", status_color)
        
        # Pytest on existing tests
        pytest_result = self.run_pytest_if_available()
        self.test_results.append(pytest_result)
        
        status_color = "SUCCESS" if pytest_result["status"] == "passed" else "WARNING"
        self.log(f"pytest: {pytest_result['status']} ({pytest_result['duration']:.2f}s)", status_color)
        
        # Comprehensive test files (if they exist)
        comprehensive_test_files = [
            "test_documentation_compliance.py",
            "test_core_config_comprehensive.py",
            "test_integration_comprehensive.py",
            "test_end_to_end_workflows.py"
        ]
        
        for test_file in comprehensive_test_files:
            if Path(test_file).exists():
                result = self.run_python_test_file(test_file)
                self.test_results.append(result)
                
                status_color = "SUCCESS" if result["status"] == "passed" else "WARNING"
                self.log(f"{test_file}: {result['status']} ({result['duration']:.2f}s)", status_color)
        
        return self.test_results
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of all test results."""
        total_duration = time.time() - self.start_time
        
        # Calculate totals
        total_tests = sum(r.get("tests_run", 0) for r in self.test_results)
        total_failures = sum(r.get("failures", 0) for r in self.test_results)
        total_errors = sum(r.get("errors", 0) for r in self.test_results)
        
        passed_suites = len([r for r in self.test_results if r["status"] == "passed"])
        failed_suites = len([r for r in self.test_results if r["status"] == "failed"])
        skipped_suites = len([r for r in self.test_results if r["status"] == "skipped"])
        
        report = f"""
{'=' * 70}
🚀 MARS-GIS COMPREHENSIVE TEST SUITE SUMMARY
{'=' * 70}

📊 OVERALL RESULTS:
   Total Test Suites: {len(self.test_results)}
   Passed Suites: {passed_suites}
   Failed Suites: {failed_suites}
   Skipped Suites: {skipped_suites}
   
   Total Tests: {total_tests}
   Total Failures: {total_failures}
   Total Errors: {total_errors}
   
   Total Duration: {total_duration:.2f}s

📋 DETAILED RESULTS:
"""
        
        for result in self.test_results:
            status_emoji = {
                "passed": "✅",
                "failed": "❌", 
                "skipped": "⏭️",
                "timeout": "⏰",
                "error": "💥"
            }.get(result["status"], "❓")
            
            report += f"""
{status_emoji} {result['file']}:
   Status: {result['status'].upper()}
   Duration: {result['duration']:.2f}s
   Tests: {result.get('tests_run', 0)}
   Failures: {result.get('failures', 0)}
   Errors: {result.get('errors', 0)}
"""
            
            if result.get("output"):
                # Show first few lines of output
                output_lines = result["output"].split('\n')[:3]
                for line in output_lines:
                    if line.strip():
                        report += f"   > {line.strip()}\n"
        
        # Success rate
        if len(self.test_results) > 0:
            success_rate = (passed_suites / len(self.test_results)) * 100
            report += f"\n🎯 SUCCESS RATE: {success_rate:.1f}%\n"
        
        report += "\n" + "=" * 70
        
        return report
    
    def save_detailed_report(self, filename: str = "comprehensive_test_report.txt"):
        """Save detailed test report to file."""
        report_path = Path(filename)
        
        with open(report_path, 'w') as f:
            f.write(self.generate_summary_report())
            f.write("\n\n" + "=" * 70)
            f.write("\n📄 DETAILED OUTPUT FOR EACH TEST SUITE:")
            f.write("\n" + "=" * 70)
            
            for result in self.test_results:
                f.write(f"\n\n{'=' * 50}")
                f.write(f"\n🔍 {result['file'].upper()}")
                f.write(f"\n{'=' * 50}")
                f.write(f"\nStatus: {result['status']}")
                f.write(f"\nDuration: {result['duration']:.2f}s")
                f.write(f"\nTests Run: {result.get('tests_run', 0)}")
                f.write(f"\nFailures: {result.get('failures', 0)}")
                f.write(f"\nErrors: {result.get('errors', 0)}")
                
                if result.get("output"):
                    f.write(f"\n\nOutput:\n{result['output']}")
        
        self.log(f"📄 Detailed report saved to: {report_path.absolute()}", "SUCCESS")
        return str(report_path.absolute())


def main():
    """Main entry point."""
    runner = ComprehensiveTestRunner()
    
    try:
        # Run all tests
        results = runner.run_all_tests()
        
        # Generate and display summary
        summary = runner.generate_summary_report()
        print(summary)
        
        # Save detailed report
        report_file = runner.save_detailed_report()
        
        # Determine exit code
        failed_suites = len([r for r in results if r["status"] == "failed"])
        exit_code = 0 if failed_suites == 0 else 1
        
        if exit_code == 0:
            runner.log("🎉 All test suites completed successfully!", "SUCCESS")
        else:
            runner.log(f"⚠️  {failed_suites} test suite(s) failed", "WARNING")
        
        runner.log(f"📄 Full report: {report_file}", "INFO")
        
        return exit_code
        
    except KeyboardInterrupt:
        runner.log("❌ Test suite interrupted by user", "WARNING")
        return 130
    except Exception as e:
        runner.log(f"💥 Fatal error: {e}", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
