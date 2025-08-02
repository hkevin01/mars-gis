# 🚀 START-GUI.SH - Mars-GIS Platform Launcher

## 📋 **OVERVIEW**

The `start-gui.sh` script is a comprehensive launcher for the Mars-GIS platform that automatically sets up and starts both the frontend web interface and backend API server with a single command.

---

## 🎯 **USAGE**

### **Quick Start**
```bash
# From project root
./scripts/start-gui.sh

# Or from scripts directory
cd scripts && ./start-gui.sh
```

### **What It Does**
1. **✅ Checks Prerequisites** - Verifies Node.js, npm, Python installation
2. **📦 Installs Dependencies** - Automatically installs frontend npm packages
3. **🚀 Starts Backend** - Launches FastAPI server at http://localhost:8000
4. **🎮 Starts Frontend** - Launches React GUI at http://localhost:3000
5. **🌐 Opens Browser** - Automatically opens the Mars exploration interface
6. **👀 Monitors Services** - Keeps both servers running until manually stopped

---

## 🖥️ **SYSTEM REQUIREMENTS**

### **Essential**
- **Node.js 16+** - Frontend runtime
- **npm** - Package manager
- **Python 3.8+** - Backend runtime
- **Modern browser** - Chrome, Firefox, Safari, Edge

### **Optional**
- **Python virtual environment** - Recommended for backend isolation
- **curl** - For health checks and service monitoring

---

## 🎮 **INTERFACE ACCESS**

### **Primary Interface**
- **🌍 Mars Exploration GUI:** http://localhost:3000
  - Interactive Mars mapping with NASA data
  - 4-tab interface (Analysis, Planning, AI/ML, Data)
  - Click-to-explore Mars surface
  - Real-time environmental data

### **Developer Interface**
- **🛰️ Backend API:** http://localhost:8000
- **📋 API Documentation:** http://localhost:8000/docs
- **❤️ Health Check:** http://localhost:8000/health

---

## 🎨 **VISUAL FEATURES**

### **Colored Output**
- **🔵 [INFO]** - General status updates
- **🟢 [SUCCESS]** - Successful operations
- **🟡 [WARNING]** - Non-critical issues
- **🔴 [ERROR]** - Critical problems requiring attention

### **ASCII Art Header**
- Professional Mars-GIS branding
- Visual confirmation of correct script execution
- Clear platform identification

---

## 🔧 **AUTOMATED FEATURES**

### **Dependency Management**
- **Frontend Dependencies** - Automatically runs `npm install` if needed
- **Dependency Checking** - Verifies all required packages are available
- **Virtual Environment** - Detects and activates Python venv if present

### **Service Management**
- **Automatic Startup** - Starts both frontend and backend services
- **Health Monitoring** - Waits for services to be ready before proceeding
- **Process Tracking** - Monitors service PIDs for unexpected failures
- **Graceful Shutdown** - Properly stops all services on Ctrl+C

### **Browser Integration**
- **Auto-Open** - Automatically opens browser to Mars GUI
- **Cross-Platform** - Works on Linux (xdg-open), macOS (open), Windows (start)
- **Fallback Support** - Provides manual URL if auto-open fails

---

## 🛑 **STOPPING THE PLATFORM**

### **Graceful Shutdown**
```bash
# Press Ctrl+C in the terminal running the script
^C
```

### **What Happens**
1. **Signal Detection** - Script catches interrupt signal
2. **Service Shutdown** - Stops frontend server (npm start)
3. **API Shutdown** - Stops backend server (uvicorn)
4. **Cleanup** - Removes temporary processes and files
5. **Confirmation** - Displays shutdown success message

---

## 🔍 **TROUBLESHOOTING**

### **Common Issues**

#### **❌ "Node.js not found"**
```bash
# Install Node.js 16+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

#### **❌ "Python 3 not found"**
```bash
# Install Python 3.8+
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### **❌ "Frontend dependencies failed"**
```bash
# Manual dependency installation
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

#### **❌ "Backend server failed to start"**
```bash
# Check Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### **⚠️ "Port already in use"**
```bash
# Check what's using the ports
lsof -i :3000  # Frontend
lsof -i :8000  # Backend

# Kill existing processes
sudo kill -9 $(lsof -t -i:3000)
sudo kill -9 $(lsof -t -i:8000)
```

### **Debug Mode**
```bash
# Run with debug output
bash -x ./scripts/start-gui.sh
```

---

## 🎯 **SCRIPT FEATURES**

### **✅ Professional Quality**
- **Comprehensive error handling** - Graceful failure management
- **Colored terminal output** - Easy-to-read status messages
- **ASCII art branding** - Professional presentation
- **Cross-platform support** - Works on Linux, macOS, Windows

### **✅ User Experience**
- **One-command launch** - Single script starts entire platform
- **Automatic browser opening** - No manual navigation needed
- **Real-time monitoring** - Continuous service health checks
- **Clear status updates** - Always know what's happening

### **✅ Developer Friendly**
- **Detailed logging** - Comprehensive status information
- **Process management** - Proper PID tracking and cleanup
- **Dependency checking** - Pre-flight verification
- **Error reporting** - Clear problem identification

---

## 🎊 **LAUNCH SUCCESS**

When the script runs successfully, you'll see:

```
🎉 MARS-GIS PLATFORM READY!

🌍 Frontend GUI: http://localhost:3000
🛰️ Backend API: http://localhost:8000
📋 API Docs: http://localhost:8000/docs

🎮 INTERACTIVE FEATURES AVAILABLE:
  • Mars Analysis Tab 🌍     - Click anywhere on Mars globe
  • Mission Planning Tab 🛰️  - Trajectory algorithms & optimization
  • AI/ML Analysis Tab 🤖     - Real-time Mars data & AI models
  • Data Management Tab 📊    - NASA/USGS datasets & export tools

🔬 PROFESSIONAL CAPABILITIES:
  • NASA Mars Trek API Integration - Real Mars surface imagery
  • MOLA Elevation Data - Mars Global Surveyor measurements
  • Interactive OpenLayers Mapping - Professional web GIS
  • AI-Powered Landing Site Optimization - Intelligent recommendations
  • Real-time Environmental Monitoring - Live Mars conditions
```

**🚀 Your professional Mars exploration platform is now running and ready for scientific discovery!**

---

*Script Location: `/home/kevin/Projects/mars-gis/scripts/start-gui.sh`*
*Created: August 1, 2025*
*Purpose: One-command Mars-GIS platform launcher*
