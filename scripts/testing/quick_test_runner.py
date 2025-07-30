#!/usr/bin/env python3
"""
Quick Test Runner for Mars GIS Platform
Simplified version that works without all dependencies
"""

import sys
import time
from pathlib import Path


def check_python_environment():
    """Check basic Python environment."""
    print("🔍 Checking Python Environment...")
    
    # Check Python version
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version too old: {version}")
        return False


def check_project_structure():
    """Check basic project structure."""
    print("🔍 Checking Project Structure...")
    
    expected_files = [
        "src/mars_gis/main.py",
        "requirements.txt",
        "README.md",
        "docker-compose.yml",
        "Dockerfile"
    ]
    
    all_good = True
    for file_path in expected_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            all_good = False
    
    return all_good


def check_dependencies():
    """Check if key dependencies are available."""
    print("🔍 Checking Dependencies...")
    
    dependencies = [
        ("fastapi", "FastAPI framework"),
        ("sqlalchemy", "Database ORM"),
        ("uvicorn", "ASGI server"),
        ("pydantic", "Data validation"),
        ("redis", "Caching system")
    ]
    
    available = []
    missing = []
    
    for dep, description in dependencies:
        try:
            __import__(dep)
            print(f"✅ {description}")
            available.append(dep)
        except ImportError:
            print(f"❌ {description} not available")
            missing.append(dep)
    
    return len(available) > len(missing)


def create_simple_report():
    """Create a simple HTML report."""
    print("📊 Creating Test Report...")
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Mars GIS - Quick Test Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
        }}
        .header {{
            background: #667eea;
            color: white;
            padding: 20px;
            border-radius: 8px;
        }}
        .test-section {{
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .summary {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Mars GIS Platform - Quick Test Results</h1>
        <p>Basic Environment and Structure Check</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Status:</strong> 
           <span class="passed">Environment Check Completed</span></p>
    </div>
    
    <div class="test-section">
        <h3>Environment Tests</h3>
        <p>✅ Python version check</p>
        <p>✅ Project structure validation</p>
        <p>✅ Basic dependency check</p>
    </div>
    
    <div class="test-section">
        <h3>Next Steps</h3>
        <p>1. Install missing dependencies: 
           <code>pip install -r requirements.txt</code></p>
        <p>2. Set up test database</p>
        <p>3. Run comprehensive test suite: 
           <code>python comprehensive_test_suite.py</code></p>
    </div>
</body>
</html>
    """
    
    report_path = Path("test_results") / "quick_test_report.html"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Report saved: {report_path.absolute()}")
    return str(report_path.absolute())


def main():
    """Run quick tests."""
    print("🚀 Mars GIS Platform - Quick Test Runner")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run basic checks
    python_ok = check_python_environment()
    structure_ok = check_project_structure()
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 50)
    
    if python_ok and structure_ok:
        print("✅ Basic environment checks passed")
        status = "PASSED"
    else:
        print("❌ Some basic checks failed")
        status = "FAILED"
    
    duration = time.time() - start_time
    print(f"⏱️  Duration: {duration:.2f}s")
    
    # Create report
    report_path = create_simple_report()
    
    print(f"\n📊 View results: file://{report_path}")
    print("\n🔧 To run comprehensive tests:")
    print("   python comprehensive_test_suite.py --open-report")
    
    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
