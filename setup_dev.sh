#!/bin/bash

# Mars GIS Platform Development Environment Setup
# This script sets up the complete development environment

set -e  # Exit on any error

echo "🚀 Setting up Mars GIS Platform Development Environment"
echo "======================================================"

# Check if Python 3.8+ is available
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ is required. Found: $python_version"
    exit 1
else
    echo "✅ Python version OK: $python_version"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "🔧 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "🔧 Creating project directories..."
mkdir -p logs
mkdir -p models
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/external
mkdir -p uploads
mkdir -p htmlcov
mkdir -p .pytest_cache

echo "✅ Directories created"

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔧 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created - please edit it with your configuration"
else
    echo "✅ .env file already exists"
fi

# Set up Git hooks if pre-commit is available
if command -v pre-commit &> /dev/null; then
    echo "🔧 Setting up pre-commit hooks..."
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  pre-commit not found, skipping Git hooks setup"
fi

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x run_tests.py
chmod +x setup_dev.sh
echo "✅ Scripts made executable"

# Run a quick test to verify setup
echo "🧪 Running basic tests to verify setup..."
python -c "
import sys
print(f'Python version: {sys.version}')

# Test core imports
try:
    import numpy
    print('✅ NumPy available')
except ImportError:
    print('❌ NumPy not available')

try:
    import pandas
    print('✅ Pandas available')
except ImportError:
    print('❌ Pandas not available')

try:
    import pytest
    print('✅ Pytest available')
except ImportError:
    print('❌ Pytest not available')

try:
    import fastapi
    print('✅ FastAPI available')
except ImportError:
    print('❌ FastAPI not available')

print('Basic setup verification complete')
"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Set up your database (PostgreSQL)"
echo "3. Set up Redis (optional, for caching)"
echo "4. Run tests: ./run_tests.py --fast"
echo "5. Start development server: uvicorn mars_gis.main:app --reload"
echo ""
echo "Useful commands:"
echo "- Run all tests: ./run_tests.py --all"
echo "- Run specific tests: ./run_tests.py --unit"
echo "- Start API server: python -m mars_gis.main"
echo "- Format code: black . && isort ."
echo "- Check types: mypy mars_gis/"
echo ""
echo "Happy coding! 🛰️🔴"
