# MARS-GIS Installation Guide

**Version:** 1.0.0
**Last Updated:** August 1, 2025
**Target Platforms:** Linux, macOS, Windows (WSL2)

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Installation Methods](#installation-methods)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Production Deployment](#production-deployment)

## System Requirements

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10/11 with WSL2
- **CPU**: 4-core processor (x86_64 architecture)
- **Memory**: 8 GB RAM
- **Storage**: 50 GB available disk space
- **Network**: Stable internet connection for data sources

### Recommended Requirements
- **CPU**: 8+ core processor with AVX2 support
- **Memory**: 16+ GB RAM
- **GPU**: CUDA-compatible GPU with 8+ GB VRAM (for ML operations)
- **Storage**: 100+ GB SSD storage
- **Network**: High-speed internet (100+ Mbps)

### Software Dependencies
- **Python**: 3.8.0 or higher (3.9+ recommended)
- **Node.js**: 16.0+ (for frontend development)
- **Docker**: 20.10+ (for containerized deployment)
- **PostgreSQL**: 13+ with PostGIS extension
- **Redis**: 6.0+ (for caching and real-time features)

## Prerequisites

### 1. Install Python 3.8+

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.9 python3.9-pip python3.9-venv python3.9-dev
```

**macOS (with Homebrew):**
```bash
brew install python@3.9
```

**Windows (WSL2):**
```bash
sudo apt update
sudo apt install python3.9 python3.9-pip python3.9-venv python3.9-dev
```

### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libffi-dev \
    libssl-dev \
    postgresql-client \
    redis-tools
```

**macOS:**
```bash
brew install gdal proj geos jpeg libpng libtiff postgresql redis
```

### 3. Install Docker (Optional)

**Ubuntu:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**macOS:**
```bash
brew install --cask docker
```

## Installation Methods

### Method 1: Local Development Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/hkevin01/mars-gis.git
cd mars-gis
```

#### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows
```

#### Step 3: Install Python Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

#### Step 4: Install Frontend Dependencies (Optional)
```bash
cd frontend
npm install
npm run build
cd ..
```

#### Step 5: Start the Application
```bash
./start_server.sh
```

### Method 2: Docker Installation

#### Step 1: Clone and Navigate
```bash
git clone https://github.com/hkevin01/mars-gis.git
cd mars-gis
```

#### Step 2: Build and Run with Docker Compose
```bash
docker-compose up --build
```

#### Step 3: Access the Application
- **API**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

### Method 3: Production Kubernetes Deployment

#### Step 1: Prepare Kubernetes Cluster
```bash
# Install kubectl and helm
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

#### Step 2: Deploy with Helm
```bash
helm repo add mars-gis https://charts.mars-gis.com
helm install mars-gis mars-gis/mars-gis \
  --set image.tag=1.0.0 \
  --set persistence.enabled=true \
  --set ingress.enabled=true
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env
```

**Required Environment Variables:**
```bash
# Application Settings
MARS_GIS_ENV=development
MARS_GIS_DEBUG=true
MARS_GIS_SECRET_KEY=your-secret-key-here
MARS_GIS_API_HOST=0.0.0.0
MARS_GIS_API_PORT=8000

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/mars_gis
REDIS_URL=redis://localhost:6379/0

# External API Keys
NASA_API_KEY=your-nasa-api-key
USGS_API_KEY=your-usgs-api-key

# ML Model Configuration
ML_MODEL_CACHE_DIR=/path/to/model/cache
TORCH_HOME=/path/to/torch/models

# Security Settings
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
CORS_ORIGINS=http://localhost:3000,https://your-frontend.com

# Feature Flags
ENABLE_ML_INFERENCE=true
ENABLE_REAL_TIME_STREAMING=true
ENABLE_BATCH_PROCESSING=true
```

### Database Setup

#### Option 1: Local PostgreSQL
```bash
# Install PostgreSQL and PostGIS
sudo apt install postgresql postgresql-contrib postgis

# Create database and user
sudo -u postgres psql
CREATE DATABASE mars_gis;
CREATE USER mars_gis_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE mars_gis TO mars_gis_user;
\q

# Enable PostGIS extension
sudo -u postgres psql -d mars_gis
CREATE EXTENSION postgis;
CREATE EXTENSION postgis_topology;
\q
```

#### Option 2: Docker PostgreSQL
```bash
docker run -d \
  --name mars-gis-postgres \
  -e POSTGRES_DB=mars_gis \
  -e POSTGRES_USER=mars_gis_user \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 \
  postgis/postgis:13-3.1
```

### Redis Setup

#### Local Redis
```bash
# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# macOS
brew services start redis
```

#### Docker Redis
```bash
docker run -d \
  --name mars-gis-redis \
  -p 6379:6379 \
  redis:6.2-alpine
```

## Verification

### 1. API Health Check
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-01T12:00:00Z",
  "version": "1.0.0"
}
```

### 2. Database Connection Test
```bash
# Activate virtual environment
source venv/bin/activate

# Test database connection
python -c "
from mars_gis.core.config import settings
import psycopg2
try:
    conn = psycopg2.connect(settings.DATABASE_URL)
    print('✅ Database connection successful')
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"
```

### 3. ML Models Test
```bash
# Test ML model loading
python -c "
try:
    from mars_gis.ml.foundation_models import EarthMarsTransferModel
    model = EarthMarsTransferModel()
    print('✅ ML models loaded successfully')
except Exception as e:
    print(f'❌ ML model loading failed: {e}')
"
```

### 4. Frontend Test (if installed)
```bash
# Check if frontend is accessible
curl http://localhost:3000
```

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError: No module named 'mars_gis'`
**Solution:**
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Install in development mode
pip install -e .

# Set PYTHONPATH if needed
export PYTHONPATH=/path/to/mars-gis/src:$PYTHONPATH
```

#### Issue: `GDAL installation failed`
**Solution:**
```bash
# Ubuntu/Debian
sudo apt install libgdal-dev gdal-bin
export GDAL_CONFIG=/usr/bin/gdal-config

# macOS
brew install gdal
export GDAL_CONFIG=/usr/local/bin/gdal-config

# Reinstall GDAL Python bindings
pip install --no-cache-dir GDAL==$(gdal-config --version)
```

#### Issue: `Database connection failed`
**Solution:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Verify database exists
sudo -u postgres psql -l | grep mars_gis

# Check connection parameters in .env file
cat .env | grep DATABASE_URL
```

#### Issue: `Permission denied for GPU access`
**Solution:**
```bash
# Add user to docker group (for GPU access)
sudo usermod -aG docker $USER

# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```

### Performance Optimization

#### Memory Usage Optimization
```bash
# Limit ML model memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Optimize Python garbage collection
export PYTHONHASHSEED=0
export MALLOC_ARENA_MAX=4
```

#### Database Performance
```sql
-- PostgreSQL optimization settings
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET random_page_cost = 1.1;
SELECT pg_reload_conf();
```

## Production Deployment

### Environment-Specific Configurations

#### Production `.env` Settings
```bash
MARS_GIS_ENV=production
MARS_GIS_DEBUG=false
MARS_GIS_SECRET_KEY=complex-secret-key-here
ALLOWED_HOSTS=your-domain.com,api.your-domain.com
CORS_ORIGINS=https://your-frontend.com

# Production database
DATABASE_URL=postgresql://user:password@db.your-domain.com:5432/mars_gis

# Redis cluster
REDIS_URL=redis://redis.your-domain.com:6379/0

# SSL/TLS
SSL_CERT_PATH=/etc/ssl/certs/mars-gis.pem
SSL_KEY_PATH=/etc/ssl/private/mars-gis.key
```

#### Systemd Service (Linux)
```bash
# Create service file
sudo nano /etc/systemd/system/mars-gis.service
```

```ini
[Unit]
Description=MARS-GIS API Server
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=mars-gis
Group=mars-gis
WorkingDirectory=/opt/mars-gis
Environment=PATH=/opt/mars-gis/venv/bin
ExecStart=/opt/mars-gis/venv/bin/uvicorn mars_gis.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable mars-gis
sudo systemctl start mars-gis
```

#### NGINX Configuration
```nginx
server {
    listen 80;
    server_name api.your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.your-domain.com;

    ssl_certificate /etc/ssl/certs/mars-gis.pem;
    ssl_certificate_key /etc/ssl/private/mars-gis.key;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Monitoring and Logging

#### Log Configuration
```bash
# Create log directory
sudo mkdir -p /var/log/mars-gis
sudo chown mars-gis:mars-gis /var/log/mars-gis

# Configure log rotation
sudo nano /etc/logrotate.d/mars-gis
```

```
/var/log/mars-gis/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 mars-gis mars-gis
    postrotate
        systemctl reload mars-gis
    endscript
}
```

#### Monitoring Setup
```bash
# Install monitoring tools
pip install prometheus-client

# Add health check endpoint monitoring
curl -s http://localhost:8000/health | jq '.status'
```

---

**Support**
- **Installation Issues**: [GitHub Issues](https://github.com/hkevin01/mars-gis/issues)
- **Documentation**: [Installation Wiki](https://github.com/hkevin01/mars-gis/wiki/Installation)
- **Community Support**: [Discussions](https://github.com/hkevin01/mars-gis/discussions)
