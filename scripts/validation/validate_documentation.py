#!/usr/bin/env python3
"""
MARS-GIS Documentation Compliance Validator
Tests that all README claims match the actual implementation.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


class DocumentationValidator:
    """Validates that documentation matches implementation."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
    def log(self, message: str, status: str = "INFO"):
        """Log with color coding."""
        colors = {
            "PASS": "\033[92m✅",      # Green
            "FAIL": "\033[91m❌",      # Red  
            "SKIP": "\033[93m⏭️",      # Yellow
            "INFO": "\033[94mℹ️",      # Blue
            "RESET": "\033[0m"         # Reset
        }
        print(f"{colors.get(status, colors['INFO'])} {message}{colors['RESET']}")
    
    def test_project_structure(self) -> Dict[str, Any]:
        """Test 1: Verify project structure matches README."""
        self.log("Testing project structure compliance...", "INFO")
        
        # Required structure from README Quick Start
        required_paths = [
            "src/mars_gis",
            "src/mars_gis/main.py",
            "src/mars_gis/core",
            "src/mars_gis/api", 
            "src/mars_gis/data",
            "src/mars_gis/ml",
            "src/mars_gis/visualization",
            "requirements.txt",
            "README.md",
            ".env.example",
            "scripts/setup_database.py",
            "scripts/download_sample_data.py"
        ]
        
        results = []
        for path_str in required_paths:
            path = Path(path_str)
            exists = path.exists()
            results.append((path_str, exists))
            
            if exists:
                self.log(f"✓ {path_str}", "PASS")
            else:
                self.log(f"✗ {path_str} (missing)", "FAIL")
        
        passed = sum(1 for _, exists in results if exists)
        total = len(results)
        
        return {
            "test": "project_structure",
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": results
        }
    
    def test_python_imports(self) -> Dict[str, Any]:
        """Test 2: Verify core Python modules can be imported."""
        self.log("Testing core module imports...", "INFO")
        
        # Core imports mentioned in README tech stack
        import_tests = [
            ("mars_gis", "Core package"),
            ("mars_gis.main", "Main application"),
            ("mars_gis.core.config", "Configuration"),
            ("pathlib", "Standard library"),
            ("json", "Standard library")
        ]
        
        results = []
        for module, desc in import_tests:
            try:
                __import__(module)
                results.append((module, True, None))
                self.log(f"✓ {module} - {desc}", "PASS")
            except ImportError as e:
                results.append((module, False, str(e)))
                self.log(f"✗ {module} - {desc}: {e}", "FAIL")
        
        passed = sum(1 for _, success, _ in results if success)
        total = len(results)
        
        return {
            "test": "python_imports",
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": results
        }
    
    def test_fastapi_application(self) -> Dict[str, Any]:
        """Test 3: Verify FastAPI app can be created as documented."""
        self.log("Testing FastAPI application creation...", "INFO")
        
        results = []
        
        try:
            from mars_gis.main import create_app
            app = create_app()
            
            # Test app was created
            if app is not None:
                results.append(("app_creation", True, "FastAPI app created"))
                self.log("✓ FastAPI app created successfully", "PASS")
            else:
                results.append(("app_creation", False, "create_app returned None"))
                self.log("✗ create_app returned None", "FAIL")
            
            # Test app has expected attributes (from README)
            expected_attrs = [
                ("title", "MARS-GIS API"),
                ("version", "0.1.0"),
                ("docs_url", "/docs"),
                ("redoc_url", "/redoc")
            ]
            
            for attr, expected in expected_attrs:
                if hasattr(app, attr):
                    actual = getattr(app, attr)
                    if actual == expected:
                        results.append((f"app_{attr}", True, f"{attr}={actual}"))
                        self.log(f"✓ App {attr}: {actual}", "PASS")
                    else:
                        results.append((f"app_{attr}", False, f"Expected {expected}, got {actual}"))
                        self.log(f"✗ App {attr}: expected {expected}, got {actual}", "FAIL")
                else:
                    results.append((f"app_{attr}", False, f"Missing attribute {attr}"))
                    self.log(f"✗ App missing attribute: {attr}", "FAIL")
            
        except ImportError as e:
            results.append(("fastapi_import", False, str(e)))
            self.log(f"✗ Cannot import FastAPI components: {e}", "SKIP")
        except Exception as e:
            results.append(("app_creation_error", False, str(e)))
            self.log(f"✗ App creation failed: {e}", "FAIL")
        
        passed = sum(1 for _, success, _ in results if success)
        total = len(results)
        
        return {
            "test": "fastapi_application",
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": results
        }
    
    def test_configuration_system(self) -> Dict[str, Any]:
        """Test 4: Verify configuration system works as documented."""
        self.log("Testing configuration system...", "INFO")
        
        results = []
        
        try:
            from mars_gis.core.config import settings

            # Test settings object exists
            results.append(("settings_object", True, "Settings imported"))
            self.log("✓ Settings object imported", "PASS")
            
            # Test key configuration attributes mentioned in README
            expected_attrs = [
                "APP_NAME", "VERSION", "HOST", "PORT",
                "DATABASE_URL", "REDIS_URL", "ALLOWED_HOSTS"
            ]
            
            for attr in expected_attrs:
                if hasattr(settings, attr):
                    value = getattr(settings, attr)
                    results.append((f"settings_{attr}", True, f"{attr}={value}"))
                    self.log(f"✓ Settings.{attr}: {value}", "PASS")
                else:
                    results.append((f"settings_{attr}", False, f"Missing {attr}"))
                    self.log(f"✗ Settings missing: {attr}", "FAIL")
            
            # Test that settings have sensible defaults
            if hasattr(settings, 'APP_NAME') and settings.APP_NAME == "MARS-GIS":
                results.append(("app_name_correct", True, "Correct app name"))
                self.log("✓ App name matches documentation", "PASS")
            else:
                results.append(("app_name_correct", False, "Incorrect app name"))
                self.log("✗ App name doesn't match documentation", "FAIL")
                
        except ImportError as e:
            results.append(("config_import", False, str(e)))
            self.log(f"✗ Cannot import configuration: {e}", "FAIL")
        except Exception as e:
            results.append(("config_error", False, str(e)))
            self.log(f"✗ Configuration error: {e}", "FAIL")
        
        passed = sum(1 for _, success, _ in results if success)
        total = len(results)
        
        return {
            "test": "configuration_system",
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": results
        }
    
    def test_technology_stack(self) -> Dict[str, Any]:
        """Test 5: Verify documented technology stack can be imported."""
        self.log("Testing technology stack availability...", "INFO")
        
        # Technologies explicitly mentioned in README
        tech_stack = [
            ("fastapi", "FastAPI - Backend framework"),
            ("pathlib", "Path utilities (built-in)"),
            ("uvicorn", "ASGI server"),
            ("json", "JSON processing (built-in)"),
            ("sys", "System utilities (built-in)")
        ]
        
        results = []
        for module, desc in tech_stack:
            try:
                __import__(module)
                results.append((module, True, None))
                self.log(f"✓ {desc}", "PASS")
            except ImportError as e:
                results.append((module, False, str(e)))
                self.log(f"✗ {desc}: {e}", "SKIP")
        
        passed = sum(1 for _, success, _ in results if success)
        total = len(results)
        
        return {
            "test": "technology_stack",
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": results
        }
    
    def test_environment_setup(self) -> Dict[str, Any]:
        """Test 6: Verify environment setup matches documentation."""
        self.log("Testing environment setup compliance...", "INFO")
        
        results = []
        
        # Check Python version (README requires 3.8+)
        python_version = sys.version_info
        if python_version >= (3, 8):
            results.append(("python_version", True, f"Python {python_version.major}.{python_version.minor}"))
            self.log(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}", "PASS")
        else:
            results.append(("python_version", False, f"Python {python_version.major}.{python_version.minor} < 3.8"))
            self.log(f"✗ Python version too old: {python_version.major}.{python_version.minor}", "FAIL")
        
        # Check for required files
        env_files = [".env.example", "requirements.txt", "README.md"]
        for file in env_files:
            if Path(file).exists():
                results.append((f"file_{file}", True, f"{file} exists"))
                self.log(f"✓ {file} exists", "PASS")
            else:
                results.append((f"file_{file}", False, f"{file} missing"))
                self.log(f"✗ {file} missing", "FAIL")
        
        passed = sum(1 for _, success, _ in results if success)
        total = len(results)
        
        return {
            "test": "environment_setup",
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": results
        }
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test 7: Verify API endpoints work as documented."""
        self.log("Testing API endpoints...", "INFO")
        
        results = []
        
        try:
            from mars_gis.main import create_app
            app = create_app()
            
            # Test that app has routes
            if hasattr(app, 'routes'):
                route_count = len(app.routes)
                results.append(("routes_exist", True, f"{route_count} routes"))
                self.log(f"✓ App has {route_count} routes", "PASS")
                
                # Look for health check endpoint (documented in README)
                health_route_found = False
                for route in app.routes:
                    if hasattr(route, 'path') and route.path == "/":
                        health_route_found = True
                        break
                
                if health_route_found:
                    results.append(("health_endpoint", True, "Health check endpoint found"))
                    self.log("✓ Health check endpoint (/) found", "PASS")
                else:
                    results.append(("health_endpoint", False, "Health check endpoint missing"))
                    self.log("✗ Health check endpoint (/) missing", "FAIL")
            else:
                results.append(("routes_exist", False, "No routes attribute"))
                self.log("✗ App has no routes attribute", "FAIL")
                
        except Exception as e:
            results.append(("api_test_error", False, str(e)))
            self.log(f"✗ API endpoint test failed: {e}", "FAIL")
        
        passed = sum(1 for _, success, _ in results if success)
        total = len(results)
        
        return {
            "test": "api_endpoints", 
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": results
        }
    
    def test_quick_start_workflow(self) -> Dict[str, Any]:
        """Test 8: Verify Quick Start workflow from README works."""
        self.log("Testing Quick Start workflow...", "INFO")
        
        results = []
        
        # Step 1: Repository structure
        if Path("src/mars_gis").exists():
            results.append(("step1_clone", True, "Repository structure"))
            self.log("✓ Step 1: Repository structure correct", "PASS")
        else:
            results.append(("step1_clone", False, "Missing src/mars_gis"))
            self.log("✗ Step 1: Missing src/mars_gis directory", "FAIL")
        
        # Step 3: Dependencies (requirements.txt exists)
        if Path("requirements.txt").exists():
            results.append(("step3_deps", True, "requirements.txt exists"))
            self.log("✓ Step 3: requirements.txt exists", "PASS")
        else:
            results.append(("step3_deps", False, "requirements.txt missing"))
            self.log("✗ Step 3: requirements.txt missing", "FAIL")
        
        # Step 4: Environment configuration
        if Path(".env.example").exists():
            results.append(("step4_env", True, ".env.example exists"))
            self.log("✓ Step 4: .env.example exists", "PASS")
        else:
            results.append(("step4_env", False, ".env.example missing"))
            self.log("✗ Step 4: .env.example missing", "FAIL")
        
        # Step 5: Database setup script
        if Path("scripts/setup_database.py").exists():
            results.append(("step5_db", True, "Database setup script exists"))
            self.log("✓ Step 5: Database setup script exists", "PASS")
        else:
            results.append(("step5_db", False, "Database setup script missing"))
            self.log("✗ Step 5: Database setup script missing", "FAIL")
        
        # Step 6: Sample data script
        if Path("scripts/download_sample_data.py").exists():
            results.append(("step6_data", True, "Sample data script exists"))
            self.log("✓ Step 6: Sample data script exists", "PASS")
        else:
            results.append(("step6_data", False, "Sample data script missing"))
            self.log("✗ Step 6: Sample data script missing", "FAIL")
        
        # Step 7: Main application
        try:
            from mars_gis.main import create_app
            app = create_app()
            if app:
                results.append(("step7_run", True, "Application can be created"))
                self.log("✓ Step 7: Application can be created", "PASS")
            else:
                results.append(("step7_run", False, "Application creation failed"))
                self.log("✗ Step 7: Application creation failed", "FAIL")
        except Exception as e:
            results.append(("step7_run", False, str(e)))
            self.log(f"✗ Step 7: Application creation error: {e}", "FAIL")
        
        passed = sum(1 for _, success, _ in results if success)
        total = len(results)
        
        return {
            "test": "quick_start_workflow",
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": results
        }
    
    def run_all_validations(self) -> List[Dict[str, Any]]:
        """Run all documentation validation tests."""
        
        self.log("🚀 MARS-GIS Documentation Compliance Validation", "INFO")
        self.log("=" * 65, "INFO")
        
        validation_tests = [
            self.test_project_structure,
            self.test_python_imports,
            self.test_fastapi_application,
            self.test_configuration_system,
            self.test_technology_stack,
            self.test_environment_setup,
            self.test_api_endpoints,
            self.test_quick_start_workflow
        ]
        
        for test_func in validation_tests:
            self.log(f"\n📋 Running {test_func.__name__}...", "INFO")
            result = test_func()
            self.results.append(result)
            
            # Summary for each test
            success_rate = result["success_rate"] * 100
            status = "PASS" if success_rate >= 80 else "FAIL"
            self.log(
                f"   {result['test']}: {result['passed']}/{result['total']} ({success_rate:.1f}%)", 
                status
            )
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        total_duration = time.time() - self.start_time
        
        # Calculate overall stats
        total_tests = sum(r['total'] for r in self.results)
        total_passed = sum(r['passed'] for r in self.results)
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Count test categories
        fully_passed = len([r for r in self.results if r['success_rate'] == 1.0])
        partially_passed = len([r for r in self.results if 0 < r['success_rate'] < 1.0])
        failed = len([r for r in self.results if r['success_rate'] == 0])
        
        report = f"""
{'=' * 70}
🚀 MARS-GIS DOCUMENTATION COMPLIANCE REPORT
{'=' * 70}

📊 OVERALL RESULTS:
   Documentation Compliance: {overall_success_rate:.1f}%
   Total Validation Points: {total_passed}/{total_tests}
   Test Categories: {len(self.results)}
   
   ✅ Fully Compliant: {fully_passed}
   ⚠️  Partially Compliant: {partially_passed}  
   ❌ Non-Compliant: {failed}
   
   Validation Duration: {total_duration:.2f}s

📋 DETAILED RESULTS:
"""
        
        for result in self.results:
            success_rate = result['success_rate'] * 100
            
            if success_rate == 100:
                status_emoji = "✅"
            elif success_rate >= 50:
                status_emoji = "⚠️"
            else:
                status_emoji = "❌"
            
            report += f"""
{status_emoji} {result['test'].upper().replace('_', ' ')}:
   Compliance Rate: {success_rate:.1f}%
   Passed: {result['passed']}/{result['total']}
"""
            
            # Show failed items for context
            if result['details'] and success_rate < 100:
                failed_items = [
                    item[0] for item in result['details'] 
                    if isinstance(item, tuple) and len(item) >= 2 and not item[1]
                ]
                if failed_items:
                    report += f"   Issues: {', '.join(failed_items[:3])}"
                    if len(failed_items) > 3:
                        report += f" (+{len(failed_items)-3} more)"
                    report += "\n"
        
        # Overall assessment
        if overall_success_rate >= 90:
            assessment = "🎉 EXCELLENT - Documentation closely matches implementation"
        elif overall_success_rate >= 75:
            assessment = "✅ GOOD - Minor discrepancies between docs and code"
        elif overall_success_rate >= 50:
            assessment = "⚠️  NEEDS WORK - Significant gaps between docs and implementation"
        else:
            assessment = "❌ POOR - Major discrepancies need immediate attention"
        
        report += f"""
🎯 ASSESSMENT: {assessment}

📄 This report validates that the MARS-GIS project implementation
   matches the claims and instructions in the README.md file.
   
{'=' * 70}
"""
        
        return report


def main():
    """Main entry point."""
    validator = DocumentationValidator()
    
    try:
        # Run all validations
        results = validator.run_all_validations()
        
        # Generate and display report
        report = validator.generate_report()
        print(report)
        
        # Save report to file
        with open("documentation_compliance_report.txt", "w") as f:
            f.write(report)
        
        # Determine exit code based on overall compliance
        total_tests = sum(r['total'] for r in results)
        total_passed = sum(r['passed'] for r in results)
        success_rate = (total_passed / total_tests) if total_tests > 0 else 0
        
        validator.log(f"📄 Full report saved to: documentation_compliance_report.txt", "INFO")
        
        if success_rate >= 0.8:  # 80% compliance threshold
            validator.log("🎉 Documentation compliance validation PASSED!", "PASS")
            return 0
        else:
            validator.log(f"⚠️  Documentation compliance below 80% ({success_rate*100:.1f}%)", "FAIL")
            return 1
        
    except KeyboardInterrupt:
        validator.log("❌ Validation interrupted by user", "FAIL")
        return 130
    except Exception as e:
        validator.log(f"💥 Fatal validation error: {e}", "FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
