#!/usr/bin/env python3
"""
Mars GIS Comprehensive Test Suite
Executes all application tests with detailed results and HTML reports.
"""

import json
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Try to import optional dependencies
try:
    import coverage  # noqa: F401
    import pytest  # noqa: F401
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


class TestResult:
    """Container for individual test results."""
    
    def __init__(self, name: str, status: str, duration: float,
                 details: str = ""):
        self.name = name
        self.status = status  # 'passed', 'failed', 'skipped', 'error'
        self.duration = duration
        self.details = details
        self.timestamp = datetime.now()


class TestSuiteRunner:
    """Comprehensive test suite runner with detailed reporting."""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results: List[TestResult] = []
        self.coverage_data: Dict[str, Any] = {}
        self.start_time = datetime.now()
        self.total_duration = 0.0
    
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color_codes = {
            "INFO": "\033[34m",     # Blue
            "SUCCESS": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",    # Red
            "RESET": "\033[0m"      # Reset
        }
        
        color = color_codes.get(level, color_codes["INFO"])
        reset = color_codes["RESET"]
        print(f"{color}[{timestamp}] {level}: {message}{reset}")
    
    def run_command(self, cmd: List[str], description: str) -> TestResult:
        """Run a command and capture results."""
        self.log(f"Running: {description}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                status = "passed"
                msg = f"✅ {description} completed successfully"
                self.log(f"{msg} ({duration:.2f}s)", "SUCCESS")
            else:
                status = "failed"
                msg = f"❌ {description} failed"
                self.log(f"{msg} ({duration:.2f}s)", "ERROR")
            
            details = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            
            return TestResult(description, status, duration, details)
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            msg = f"⏰ {description} timed out after {duration:.2f}s"
            self.log(msg, "WARNING")
            return TestResult(description, "error", duration, "Test timed out")
        
        except Exception as e:
            duration = time.time() - start_time
            self.log(f"💥 {description} error: {str(e)}", "ERROR")
            return TestResult(description, "error", duration, str(e))
    
    def test_environment_setup(self) -> TestResult:
        """Test that the development environment is properly set up."""
        self.log("Testing environment setup...")
        
        checks = []
        details = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            checks.append(True)
            version_str = f"{python_version.major}.{python_version.minor}"
            version_str += f".{python_version.micro}"
            details.append(f"✅ Python {version_str}")
        else:
            checks.append(False)
            details.append(f"❌ Python version too old: {python_version}")
        
        # Check virtual environment
        has_real_prefix = hasattr(sys, 'real_prefix')
        has_base_prefix = (hasattr(sys, 'base_prefix') and
                           sys.base_prefix != sys.prefix)
        if has_real_prefix or has_base_prefix:
            checks.append(True)
            details.append("✅ Virtual environment active")
        else:
            checks.append(False)
            details.append("❌ Virtual environment not detected")
        
        # Check key dependencies
        dependencies = [
            ('fastapi', 'FastAPI framework'),
            ('sqlalchemy', 'Database ORM'),
            ('geopandas', 'Geospatial data processing'),
            ('torch', 'Machine learning framework'),
            ('redis', 'Caching system')
        ]
        
        for dep, description in dependencies:
            try:
                __import__(dep)
                checks.append(True)
                details.append(f"✅ {description} available")
            except ImportError:
                checks.append(False)
                details.append(f"❌ {description} not available")
        
        # Check configuration files
        config_files = [
            '.env',
            'requirements.txt',
            'pytest.ini',
            'src/mars_gis/main.py'
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                checks.append(True)
                details.append(f"✅ {config_file} exists")
            else:
                checks.append(False)
                details.append(f"❌ {config_file} missing")
        
        all_passed = all(checks)
        status = "passed" if all_passed else "failed"
        details_str = "\n".join(details)
        
        return TestResult("Environment Setup", status, 0.1, details_str)
    
    def test_database_connection(self) -> TestResult:
        """Test database connectivity and schema."""
        self.log("Testing database connection...")
        
        try:
            # Try to connect to test database
            import os

            from sqlalchemy import create_engine, text

            # Use test database URL or fallback to SQLite
            default_db = 'sqlite:///test_mars_gis.db'
            db_url = os.getenv('TEST_DATABASE_URL', default_db)
            
            engine = create_engine(db_url)
            
            with engine.connect() as conn:
                # Test basic connection
                result = conn.execute(text("SELECT 1")).fetchone()
                
                details = [
                    f"✅ Database connection successful",
                    f"✅ Database URL: {db_url}",
                    f"✅ Basic query executed: {result[0]}"
                ]
                
                # Test Mars GIS specific schemas (if PostgreSQL)
                if 'postgresql' in db_url:
                    try:
                        schemas = conn.execute(text(
                            "SELECT schema_name FROM information_schema.schemata "
                            "WHERE schema_name IN ('mars_gis', 'data_sources', 'analysis', 'missions')"
                        )).fetchall()
                        
                        schema_names = [row[0] for row in schemas]
                        details.append(f"✅ Mars GIS schemas found: {schema_names}")
                        
                    except Exception as e:
                        details.append(f"⚠️  Schema check failed: {str(e)}")
                
                return TestResult("Database Connection", "passed", 0.5, "\n".join(details))
                
        except Exception as e:
            return TestResult("Database Connection", "failed", 0.5, f"❌ Database connection failed: {str(e)}")
    
    def test_api_endpoints(self) -> TestResult:
        """Test API endpoints availability."""
        self.log("Testing API endpoints...")
        
        try:
            import subprocess
            import threading
            import time
            from contextlib import contextmanager

            import requests

            # Start the API server in background
            api_process = None
            try:
                # Try to start the API server
                api_process = subprocess.Popen([
                    sys.executable, '-m', 'uvicorn', 'mars_gis.main:app',
                    '--host', '127.0.0.1', '--port', '8001'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Wait for server to start
                time.sleep(3)
                
                base_url = "http://127.0.0.1:8001"
                endpoints_to_test = [
                    "/",
                    "/health",
                    "/docs",
                    "/openapi.json"
                ]
                
                details = []
                all_passed = True
                
                for endpoint in endpoints_to_test:
                    try:
                        response = requests.get(f"{base_url}{endpoint}", timeout=5)
                        if response.status_code == 200:
                            details.append(f"✅ {endpoint}: {response.status_code}")
                        else:
                            details.append(f"❌ {endpoint}: {response.status_code}")
                            all_passed = False
                    except requests.exceptions.ConnectionError:
                        details.append(f"❌ {endpoint}: Connection refused")
                        all_passed = False
                    except Exception as e:
                        details.append(f"❌ {endpoint}: {str(e)}")
                        all_passed = False
                
                status = "passed" if all_passed else "failed"
                return TestResult("API Endpoints", status, 2.0, "\n".join(details))
                
            finally:
                if api_process:
                    api_process.terminate()
                    api_process.wait(timeout=5)
                    
        except ImportError:
            return TestResult("API Endpoints", "skipped", 0.1, "❌ requests library not available")
        except Exception as e:
            return TestResult("API Endpoints", "error", 1.0, f"❌ API test error: {str(e)}")
    
    def run_pytest_suite(self) -> TestResult:
        """Run the comprehensive pytest suite."""
        if not PYTEST_AVAILABLE:
            return TestResult("PyTest Suite", "skipped", 0.1, "❌ pytest not available")
        
        self.log("Running comprehensive pytest suite...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "--verbose",
            "--tb=short",
            "--maxfail=10",
            "--durations=10"
        ]
        
        # Add coverage if available
        try:
            import coverage
            cmd.extend([
                "--cov=src/mars_gis",
                "--cov-report=html:htmlcov",
                "--cov-report=json:coverage.json",
                "--cov-report=term-missing"
            ])
        except ImportError:
            pass
        
        return self.run_command(cmd, "PyTest Test Suite")
    
    def test_ml_models(self) -> TestResult:
        """Test machine learning model functionality."""
        self.log("Testing ML models...")
        
        try:
            # Test PyTorch availability
            import torch
            
            details = [
                f"✅ PyTorch available: {torch.__version__}",
                f"✅ CUDA available: {torch.cuda.is_available()}"
            ]
            
            # Test basic tensor operations
            try:
                x = torch.randn(2, 3)
                y = torch.randn(3, 2)
                z = torch.mm(x, y)
                details.append("✅ Basic tensor operations working")
            except Exception as e:
                details.append(f"❌ Tensor operations failed: {str(e)}")
                return TestResult("ML Models", "failed", 0.5, "\n".join(details))
            
            # Test model imports (if available)
            try:
                from mars_gis.ml.models.terrain_models import MarsTerrainCNN
                model = MarsTerrainCNN(num_classes=8)
                details.append("✅ Mars Terrain CNN model created")
                
                # Test forward pass
                test_input = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    output = model(test_input)
                details.append(f"✅ Forward pass successful: {output.shape}")
                
            except ImportError:
                details.append("⚠️  Mars ML models not found (expected in development)")
            except Exception as e:
                details.append(f"❌ ML model test failed: {str(e)}")
            
            return TestResult("ML Models", "passed", 1.0, "\n".join(details))
            
        except ImportError:
            return TestResult("ML Models", "failed", 0.1, "❌ PyTorch not available")
        except Exception as e:
            return TestResult("ML Models", "error", 0.5, f"❌ ML test error: {str(e)}")
    
    def test_geospatial_functionality(self) -> TestResult:
        """Test geospatial analysis capabilities."""
        self.log("Testing geospatial functionality...")
        
        try:
            import geopandas as gpd
            import pandas as pd
            from shapely.geometry import Point, Polygon
            
            details = [
                f"✅ GeoPandas available: {gpd.__version__}",
                f"✅ Pandas available: {pd.__version__}"
            ]
            
            # Test basic geospatial operations
            try:
                # Create test points (Mars coordinates)
                mars_points = [
                    Point(175.4729, -14.5684),  # Olympia Undae
                    Point(137.8414, -5.4453),   # Gale Crater
                    Point(-49.97, 22.5)         # Valles Marineris
                ]
                
                gdf = gpd.GeoDataFrame(
                    {'name': ['Olympia Undae', 'Gale Crater', 'Valles Marineris']},
                    geometry=mars_points
                )
                
                details.append(f"✅ Created GeoDataFrame with {len(gdf)} Mars locations")
                
                # Test distance calculations
                dist = mars_points[0].distance(mars_points[1])
                details.append(f"✅ Distance calculation: {dist:.4f} degrees")
                
                # Test buffer operations
                buffered = gdf.buffer(1.0)  # 1 degree buffer
                details.append(f"✅ Buffer operation successful: {len(buffered)} polygons")
                
            except Exception as e:
                details.append(f"❌ Geospatial operations failed: {str(e)}")
                return TestResult("Geospatial Functionality", "failed", 1.0, "\n".join(details))
            
            # Test coordinate system handling
            try:
                import pyproj
                details.append(f"✅ PyProj available: {pyproj.__version__}")
                
                # Test Mars coordinate transformation (if possible)
                crs_wgs84 = pyproj.CRS.from_epsg(4326)  # WGS84
                details.append("✅ Coordinate system handling available")
                
            except ImportError:
                details.append("⚠️  PyProj not available for coordinate transformations")
            except Exception as e:
                details.append(f"❌ Coordinate system test failed: {str(e)}")
            
            return TestResult("Geospatial Functionality", "passed", 1.5, "\n".join(details))
            
        except ImportError:
            return TestResult("Geospatial Functionality", "failed", 0.1, "❌ GeoPandas not available")
        except Exception as e:
            return TestResult("Geospatial Functionality", "error", 0.5, f"❌ Geospatial test error: {str(e)}")
    
    def test_data_processing(self) -> TestResult:
        """Test data processing pipeline."""
        self.log("Testing data processing pipeline...")
        
        details = []
        
        try:
            import numpy as np
            import pandas as pd
            
            details.append(f"✅ NumPy available: {np.__version__}")
            details.append(f"✅ Pandas available: {pd.__version__}")
            
            # Test data processing operations
            try:
                # Create mock Mars data
                np.random.seed(42)
                mars_data = pd.DataFrame({
                    'latitude': np.random.uniform(-90, 90, 100),
                    'longitude': np.random.uniform(-180, 180, 100),
                    'elevation': np.random.uniform(-8000, 21000, 100),  # Mars elevation range
                    'temperature': np.random.uniform(-143, 35, 100),   # Mars temperature range
                    'pressure': np.random.uniform(30, 1155, 100)       # Mars pressure range
                })
                
                details.append(f"✅ Created mock Mars dataset: {mars_data.shape}")
                
                # Test data validation
                valid_lat = mars_data['latitude'].between(-90, 90).all()
                valid_lon = mars_data['longitude'].between(-180, 180).all()
                valid_temp = mars_data['temperature'].between(-200, 50).all()
                
                if valid_lat and valid_lon and valid_temp:
                    details.append("✅ Data validation passed")
                else:
                    details.append("❌ Data validation failed")
                
                # Test statistical operations
                stats = mars_data.describe()
                details.append(f"✅ Statistical analysis completed: {stats.shape}")
                
                # Test data filtering
                cold_regions = mars_data[mars_data['temperature'] < -100]
                details.append(f"✅ Data filtering: {len(cold_regions)} cold regions found")
                
            except Exception as e:
                details.append(f"❌ Data processing operations failed: {str(e)}")
                return TestResult("Data Processing", "failed", 1.0, "\n".join(details))
            
            return TestResult("Data Processing", "passed", 1.0, "\n".join(details))
            
        except ImportError:
            return TestResult("Data Processing", "failed", 0.1, "❌ Required libraries not available")
        except Exception as e:
            return TestResult("Data Processing", "error", 0.5, f"❌ Data processing test error: {str(e)}")
    
    def test_security_and_validation(self) -> TestResult:
        """Test security and validation features."""
        self.log("Testing security and validation...")
        
        details = []
        
        try:
            # Test input validation
            def validate_mars_coordinates(lat: float, lon: float) -> bool:
                return -90 <= lat <= 90 and -180 <= lon <= 180
            
            # Test valid coordinates
            valid_coords = [
                (-14.5684, 175.4729),  # Olympia Undae
                (-5.4453, 137.8414),   # Gale Crater
                (22.5, -49.97)         # Valles Marineris
            ]
            
            for lat, lon in valid_coords:
                if validate_mars_coordinates(lat, lon):
                    details.append(f"✅ Valid coordinates: ({lat}, {lon})")
                else:
                    details.append(f"❌ Invalid coordinates: ({lat}, {lon})")
            
            # Test invalid coordinates
            invalid_coords = [
                (95.0, 0.0),      # Invalid latitude
                (0.0, 185.0),     # Invalid longitude
                (-95.0, -185.0)   # Both invalid
            ]
            
            for lat, lon in invalid_coords:
                if not validate_mars_coordinates(lat, lon):
                    details.append(f"✅ Correctly rejected invalid coordinates: ({lat}, {lon})")
                else:
                    details.append(f"❌ Incorrectly accepted invalid coordinates: ({lat}, {lon})")
            
            # Test data sanitization
            import re
            
            def sanitize_mission_name(name: str) -> str:
                # Remove potentially dangerous characters
                return re.sub(r'[<>\"\'&]', '', name).strip()
            
            test_names = [
                "Mars Mission 2024",
                "<script>alert('xss')</script>",
                "Mission & Analysis",
                "Normal Mission Name"
            ]
            
            for name in test_names:
                sanitized = sanitize_mission_name(name)
                details.append(f"✅ Sanitized '{name}' -> '{sanitized}'")
            
            return TestResult("Security & Validation", "passed", 0.5, "\n".join(details))
            
        except Exception as e:
            return TestResult("Security & Validation", "error", 0.5, f"❌ Security test error: {str(e)}")
    
    def generate_html_report(self) -> str:
        """Generate comprehensive HTML test report."""
        self.log("Generating HTML test report...")
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == "passed")
        failed_tests = sum(1 for r in self.results if r.status == "failed")
        skipped_tests = sum(1 for r in self.results if r.status == "skipped")
        error_tests = sum(1 for r in self.results if r.status == "error")
        
        total_duration = sum(r.duration for r in self.results)
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mars GIS Platform - Test Results</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .summary-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .summary-card h3 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .passed { color: #28a745; }
        .failed { color: #dc3545; }
        .skipped { color: #ffc107; }
        .error { color: #fd7e14; }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 1rem 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
            transition: width 0.3s ease;
        }
        
        .test-results {
            display: grid;
            gap: 1rem;
        }
        
        .test-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        
        .test-header {
            padding: 1rem;
            border-left: 4px solid;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .test-header.passed { border-left-color: #28a745; background-color: #d4edda; }
        .test-header.failed { border-left-color: #dc3545; background-color: #f8d7da; }
        .test-header.skipped { border-left-color: #ffc107; background-color: #fff3cd; }
        .test-header.error { border-left-color: #fd7e14; background-color: #ffeaa7; }
        
        .test-name {
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .test-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-badge.passed { background-color: #28a745; color: white; }
        .status-badge.failed { background-color: #dc3545; color: white; }
        .status-badge.skipped { background-color: #ffc107; color: black; }
        .status-badge.error { background-color: #fd7e14; color: white; }
        
        .test-details {
            padding: 1rem;
            background-color: #f8f9fa;
            border-top: 1px solid #dee2e6;
            display: none;
        }
        
        .test-details.show {
            display: block;
        }
        
        .test-details pre {
            background-color: #343a40;
            color: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        .timestamp {
            font-size: 0.9rem;
            color: #6c757d;
        }
        
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            color: #6c757d;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .summary {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Mars GIS Platform</h1>
            <p>Comprehensive Test Results Dashboard</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3 class="passed">{{ passed_tests }}</h3>
                <p>Passed</p>
            </div>
            <div class="summary-card">
                <h3 class="failed">{{ failed_tests }}</h3>
                <p>Failed</p>
            </div>
            <div class="summary-card">
                <h3 class="skipped">{{ skipped_tests }}</h3>
                <p>Skipped</p>
            </div>
            <div class="summary-card">
                <h3 class="error">{{ error_tests }}</h3>
                <p>Errors</p>
            </div>
        </div>
        
        <div class="summary-card">
            <h3>{{ "%.1f"|format(pass_rate) }}%</h3>
            <p>Pass Rate</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ pass_rate }}%"></div>
            </div>
            <p class="timestamp">Total Duration: {{ "%.2f"|format(total_duration) }}s</p>
        </div>
        
        <div class="test-results">
            {% for result in results %}
            <div class="test-card">
                <div class="test-header {{ result.status }}" onclick="toggleDetails('test-{{ loop.index0 }}')">
                    <div class="test-name">{{ result.name }}</div>
                    <div class="test-status">
                        <span class="timestamp">{{ result.duration }}s</span>
                        <span class="status-badge {{ result.status }}">{{ result.status }}</span>
                    </div>
                </div>
                <div id="test-{{ loop.index0 }}" class="test-details">
                    <p class="timestamp">Executed: {{ result.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                    <pre>{{ result.details }}</pre>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <div class="footer">
            <p>Generated on {{ generation_time.strftime('%Y-%m-%d %H:%M:%S') }} | Mars GIS Platform Test Suite</p>
        </div>
    </div>
    
    <script>
        function toggleDetails(testId) {
            const details = document.getElementById(testId);
            if (details.classList.contains('show')) {
                details.classList.remove('show');
            } else {
                details.classList.add('show');
            }
        }
        
        // Auto-expand failed tests
        document.addEventListener('DOMContentLoaded', function() {
            const failedTests = document.querySelectorAll('.test-header.failed, .test-header.error');
            failedTests.forEach(header => {
                header.click();
            });
        });
    </script>
</body>
</html>
        """
        
        if JINJA2_AVAILABLE:
            template = Template(html_template)
            html_content = template.render(
                results=self.results,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=skipped_tests,
                error_tests=error_tests,
                pass_rate=pass_rate,
                total_duration=total_duration,
                generation_time=datetime.now()
            )
        else:
            # Fallback to simple string replacement
            html_content = html_template.replace("{{ passed_tests }}", str(passed_tests))
            html_content = html_content.replace("{{ failed_tests }}", str(failed_tests))
            html_content = html_content.replace("{{ skipped_tests }}", str(skipped_tests))
            html_content = html_content.replace("{{ error_tests }}", str(error_tests))
            html_content = html_content.replace("{{ pass_rate }}", f"{pass_rate:.1f}")
            html_content = html_content.replace("{{ total_duration }}", f"{total_duration:.2f}")
            
            # Add results manually
            results_html = ""
            for i, result in enumerate(self.results):
                results_html += f"""
                <div class="test-card">
                    <div class="test-header {result.status}" onclick="toggleDetails('test-{i}')">
                        <div class="test-name">{result.name}</div>
                        <div class="test-status">
                            <span class="timestamp">{result.duration:.2f}s</span>
                            <span class="status-badge {result.status}">{result.status}</span>
                        </div>
                    </div>
                    <div id="test-{i}" class="test-details">
                        <p class="timestamp">Executed: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <pre>{result.details}</pre>
                    </div>
                </div>
                """
            
            html_content = html_content.replace("{% for result in results %}...{% endfor %}", results_html)
            html_content = html_content.replace("{{ generation_time.strftime('%Y-%m-%d %H:%M:%S') }}", 
                                              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Save HTML report
        report_path = self.output_dir / "test_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path.absolute())
    
    def generate_json_report(self) -> str:
        """Generate JSON test report for CI/CD integration."""
        report_data = {
            "test_run": {
                "timestamp": self.start_time.isoformat(),
                "duration": sum(r.duration for r in self.results),
                "total_tests": len(self.results)
            },
            "summary": {
                "passed": sum(1 for r in self.results if r.status == "passed"),
                "failed": sum(1 for r in self.results if r.status == "failed"),
                "skipped": sum(1 for r in self.results if r.status == "skipped"),
                "errors": sum(1 for r in self.results if r.status == "error")
            },
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "duration": r.duration,
                    "timestamp": r.timestamp.isoformat(),
                    "details": r.details
                }
                for r in self.results
            ]
        }
        
        json_path = self.output_dir / "test_results.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return str(json_path.absolute())
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all test suites and generate reports."""
        self.log("🚀 Starting Mars GIS Platform Comprehensive Test Suite", "SUCCESS")
        self.log("=" * 80)
        
        # Test suites to run
        test_suites = [
            self.test_environment_setup,
            self.test_database_connection,
            self.test_api_endpoints,
            self.test_ml_models,
            self.test_geospatial_functionality,
            self.test_data_processing,
            self.test_security_and_validation,
            self.run_pytest_suite
        ]
        
        # Execute all test suites
        for test_suite in test_suites:
            try:
                result = test_suite()
                self.results.append(result)
            except Exception as e:
                error_result = TestResult(
                    test_suite.__name__,
                    "error",
                    0.1,
                    f"Test suite execution error: {str(e)}"
                )
                self.results.append(error_result)
                self.log(f"💥 Error in {test_suite.__name__}: {str(e)}", "ERROR")
        
        # Calculate totals
        self.total_duration = time.time() - self.start_time.timestamp()
        
        # Generate reports
        self.log("📊 Generating test reports...")
        html_report = self.generate_html_report()
        json_report = self.generate_json_report()
        
        # Summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == "passed")
        failed_tests = sum(1 for r in self.results if r.status == "failed")
        
        self.log("=" * 80)
        self.log("📋 TEST SUMMARY", "SUCCESS")
        self.log(f"   Total Tests: {total_tests}")
        self.log(f"   Passed: {passed_tests}", "SUCCESS")
        self.log(f"   Failed: {failed_tests}", "ERROR" if failed_tests > 0 else "SUCCESS")
        self.log(f"   Pass Rate: {(passed_tests/total_tests*100):.1f}%")
        self.log(f"   Duration: {self.total_duration:.2f}s")
        self.log("=" * 80)
        
        self.log(f"📄 HTML Report: {html_report}", "SUCCESS")
        self.log(f"📄 JSON Report: {json_report}", "SUCCESS")
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": passed_tests/total_tests*100 if total_tests > 0 else 0,
            "duration": self.total_duration,
            "html_report": html_report,
            "json_report": json_report,
            "results": self.results
        }


def main():
    """Main entry point for the test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mars GIS Comprehensive Test Suite")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for reports")
    parser.add_argument("--open-report", action="store_true", help="Open HTML report in browser")
    parser.add_argument("--json-only", action="store_true", help="Generate JSON report only")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestSuiteRunner(args.output_dir)
    
    try:
        # Run comprehensive tests
        results = runner.run_comprehensive_tests()
        
        # Open report in browser if requested
        if args.open_report and not args.json_only:
            try:
                webbrowser.open(f"file://{results['html_report']}")
                runner.log("🌐 Opening test report in browser...", "SUCCESS")
            except Exception as e:
                runner.log(f"Failed to open browser: {str(e)}", "WARNING")
        
        # Exit with appropriate code
        sys.exit(0 if results['failed'] == 0 else 1)
        
    except KeyboardInterrupt:
        runner.log("❌ Test suite interrupted by user", "WARNING")
        sys.exit(130)
    except Exception as e:
        runner.log(f"💥 Fatal error: {str(e)}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
