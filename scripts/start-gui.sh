#!/bin/bash

# 🚀 MARS-GIS GUI LAUNCHER SCRIPT
# Launches the complete Mars exploration platform interface
# Usage: ./scripts/start-gui.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

# ASCII Art Header
clear
echo -e "${CYAN}"
echo "███╗   ███╗ █████╗ ██████╗ ███████╗      ██████╗ ██╗███████╗"
echo "████╗ ████║██╔══██╗██╔══██╗██╔════╝     ██╔════╝ ██║██╔════╝"
echo "██╔████╔██║███████║██████╔╝███████╗     ██║  ███╗██║███████╗"
echo "██║╚██╔╝██║██╔══██║██╔══██╗╚════██║     ██║   ██║██║╚════██║"
echo "██║ ╚═╝ ██║██║  ██║██║  ██║███████║     ╚██████╔╝██║███████║"
echo "╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝      ╚═════╝ ╚═╝╚══════╝"
echo -e "${NC}"
echo -e "${PURPLE}🚀 Professional Mars Exploration Platform Launcher${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
print_status "Project root: $PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -f "$PROJECT_ROOT/README.md" ] || [ ! -d "$PROJECT_ROOT/frontend" ]; then
    print_error "Invalid project structure. Please run this script from the mars-gis project root or scripts directory."
    exit 1
fi

print_header "🔍 CHECKING PREREQUISITES..."

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js found: $NODE_VERSION"
else
    print_error "Node.js not found. Please install Node.js 16+ to run the frontend."
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    print_success "npm found: $NPM_VERSION"
else
    print_error "npm not found. Please install npm to manage frontend dependencies."
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.8+ to run the backend."
    exit 1
fi

print_header "📦 SETTING UP DEPENDENCIES..."

# Navigate to frontend directory and install dependencies
cd "$PROJECT_ROOT/frontend"
print_status "Installing frontend dependencies..."

if [ ! -d "node_modules" ] || [ ! -f "package-lock.json" ]; then
    print_status "Running npm install for frontend dependencies..."
    npm install
    if [ $? -eq 0 ]; then
        print_success "Frontend dependencies installed successfully"
    else
        print_error "Failed to install frontend dependencies"
        exit 1
    fi
else
    print_success "Frontend dependencies already installed"
fi

# Check Python virtual environment
cd "$PROJECT_ROOT"
if [ -d "venv" ]; then
    print_success "Python virtual environment found"
    source venv/bin/activate
    print_status "Activated Python virtual environment"
else
    print_warning "Python virtual environment not found. Backend may not work properly."
    print_status "To create virtual environment, run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
fi

print_header "🚀 LAUNCHING MARS-GIS PLATFORM..."

# Function to cleanup background processes
cleanup() {
    print_header "🛑 SHUTTING DOWN MARS-GIS PLATFORM..."
    if [ ! -z "$FRONTEND_PID" ] && kill -0 $FRONTEND_PID 2>/dev/null; then
        print_status "Stopping frontend server (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$BACKEND_PID" ] && kill -0 $BACKEND_PID 2>/dev/null; then
        print_status "Stopping backend server (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null || true
    fi
    print_success "Mars-GIS platform stopped successfully"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend server (if available)
if [ -f "$PROJECT_ROOT/requirements.txt" ] && [ -f "$PROJECT_ROOT/src/mars_gis/main.py" ]; then
    print_status "Starting backend API server..."
    cd "$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT/src"
    nohup uvicorn mars_gis.main:app --reload --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
    BACKEND_PID=$!
    print_success "Backend server started (PID: $BACKEND_PID) - http://localhost:8000"

    # Wait for backend to be ready
    print_status "Waiting for backend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_success "Backend server is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_warning "Backend server may not be ready, continuing anyway..."
        fi
        sleep 1
    done
else
    print_warning "Backend components not found. Only frontend will be available."
    BACKEND_PID=""
fi

# Start frontend server
print_status "Starting frontend GUI server..."
cd "$PROJECT_ROOT/frontend"
nohup npm start > /dev/null 2>&1 &
FRONTEND_PID=$!
print_success "Frontend server started (PID: $FRONTEND_PID)"

# Wait for frontend to be ready
print_status "Waiting for frontend to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        print_success "Frontend server is ready"
        break
    fi
    if [ $i -eq 60 ]; then
        print_error "Frontend server failed to start properly"
        cleanup
        exit 1
    fi
    sleep 1
done

print_header "🎉 MARS-GIS PLATFORM READY!"
echo
print_success "🌍 Frontend GUI: http://localhost:3000"
if [ ! -z "$BACKEND_PID" ]; then
    print_success "🛰️ Backend API: http://localhost:8000"
    print_success "📋 API Docs: http://localhost:8000/docs"
fi
echo
print_header "🎮 INTERACTIVE FEATURES AVAILABLE:"
echo -e "${CYAN}  • Mars Analysis Tab 🌍     - Click anywhere on Mars globe${NC}"
echo -e "${CYAN}  • Mission Planning Tab 🛰️  - Trajectory algorithms & optimization${NC}"
echo -e "${CYAN}  • AI/ML Analysis Tab 🤖     - Real-time Mars data & AI models${NC}"
echo -e "${CYAN}  • Data Management Tab 📊    - NASA/USGS datasets & export tools${NC}"
echo
print_header "🔬 PROFESSIONAL CAPABILITIES:"
echo -e "${CYAN}  • NASA Mars Trek API Integration - Real Mars surface imagery${NC}"
echo -e "${CYAN}  • MOLA Elevation Data - Mars Global Surveyor measurements${NC}"
echo -e "${CYAN}  • Interactive OpenLayers Mapping - Professional web GIS${NC}"
echo -e "${CYAN}  • AI-Powered Landing Site Optimization - Intelligent recommendations${NC}"
echo -e "${CYAN}  • Real-time Environmental Monitoring - Live Mars conditions${NC}"
echo
print_warning "Press Ctrl+C to stop all servers and exit"
echo

# Open browser automatically (if available)
if command -v xdg-open &> /dev/null; then
    print_status "Opening browser automatically..."
    xdg-open http://localhost:3000 &
elif command -v open &> /dev/null; then
    print_status "Opening browser automatically..."
    open http://localhost:3000 &
elif command -v start &> /dev/null; then
    print_status "Opening browser automatically..."
    start http://localhost:3000 &
else
    print_status "Please open your browser manually to: http://localhost:3000"
fi

# Keep script running and monitor processes
while true; do
    # Check if frontend is still running
    if [ ! -z "$FRONTEND_PID" ] && ! kill -0 $FRONTEND_PID 2>/dev/null; then
        print_error "Frontend server stopped unexpectedly"
        cleanup
        exit 1
    fi

    # Check if backend is still running (if it was started)
    if [ ! -z "$BACKEND_PID" ] && ! kill -0 $BACKEND_PID 2>/dev/null; then
        print_warning "Backend server stopped unexpectedly"
        BACKEND_PID=""
    fi

    sleep 5
done
