#!/bin/bash
# MARS-GIS API Server Startup Script

echo "🚀 Starting MARS-GIS API Server..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment and start server
cd /home/kevin/Projects/mars-gis
source venv/bin/activate

echo "📡 Server starting on http://localhost:8000"
echo "📚 API documentation available at http://localhost:8000/docs"
echo "🔧 Press Ctrl+C to stop the server"
echo ""

PYTHONPATH=/home/kevin/Projects/mars-gis/src uvicorn mars_gis.main:app --host 0.0.0.0 --port 8000 --reload
