# MARS-GIS Troubleshooting Guide

**Version:** 1.0.0
**Last Updated:** August 1, 2025
**Support Level:** Production Ready

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [Installation Problems](#installation-problems)
4. [Runtime Errors](#runtime-errors)
5. [Performance Issues](#performance-issues)
6. [Database Problems](#database-problems)
7. [API Issues](#api-issues)
8. [Frontend Problems](#frontend-problems)
9. [ML Model Issues](#ml-model-issues)
10. [Deployment Troubleshooting](#deployment-troubleshooting)
11. [Monitoring and Logs](#monitoring-and-logs)
12. [Getting Help](#getting-help)

## Quick Diagnostics

### System Health Check

Run this comprehensive health check to identify common issues:

```bash
# Backend health check
curl -X GET "http://localhost:8000/health" -H "accept: application/json"

# Database connectivity
python -c "
from mars_gis.core.database import check_db_connection
import asyncio
print(asyncio.run(check_db_connection()))
"

# Redis connectivity
redis-cli ping

# Check running services
docker ps  # If using Docker
kubectl get pods  # If using Kubernetes

# Check logs
tail -f logs/mars_gis.log
```

### Quick Status Commands

```bash
# Check application status
systemctl status mars-gis  # Linux service

# Check port availability
netstat -tlnp | grep :8000  # Backend port
netstat -tlnp | grep :3000  # Frontend port

# Check disk space
df -h

# Check memory usage
free -m

# Check CPU usage
top -bn1 | grep "Cpu(s)"
```

## Common Issues

### Issue: Application Won't Start

**Symptoms:**
- Server fails to start
- Import errors
- Port binding failures

**Diagnosis:**
```bash
# Check for port conflicts
lsof -i :8000

# Check Python environment
python --version
pip list | grep mars-gis

# Check configuration
python -c "from mars_gis.core.config import settings; print(settings.dict())"
```

**Solutions:**

1. **Port Already in Use:**
```bash
# Kill process using port
sudo kill -9 $(lsof -t -i:8000)

# Or change port in configuration
export MARS_GIS_PORT=8001
```

2. **Missing Dependencies:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check for version conflicts
pip check
```

3. **Environment Configuration:**
```bash
# Set required environment variables
export DATABASE_URL="postgresql://user:pass@localhost:5432/mars_gis"
export REDIS_URL="redis://localhost:6379/0"
export MARS_GIS_SECRET_KEY="your-secret-key"
```

### Issue: Database Connection Failed

**Symptoms:**
- `ConnectionError: could not connect to server`
- Database timeout errors
- Authentication failures

**Diagnosis:**
```bash
# Test PostgreSQL connection
psql -h localhost -U mars_gis_user -d mars_gis_db -c "SELECT version();"

# Check PostgreSQL service
sudo systemctl status postgresql

# Test connection string
python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('postgresql://user:pass@localhost/mars_gis')
    print('Connected successfully')
    await conn.close()
asyncio.run(test())
"
```

**Solutions:**

1. **PostgreSQL Not Running:**
```bash
# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

2. **Wrong Credentials:**
```bash
# Reset PostgreSQL password
sudo -u postgres psql
\password mars_gis_user
```

3. **PostGIS Extension Missing:**
```bash
# Connect as superuser and enable PostGIS
sudo -u postgres psql -d mars_gis_db
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
```

### Issue: Memory Issues

**Symptoms:**
- `MemoryError` during ML inference
- Application crashes
- Slow performance

**Diagnosis:**
```bash
# Check memory usage
free -m
ps aux --sort=-%mem | head

# Check swap usage
swapon --show

# Monitor memory during operation
top -p $(pgrep -f mars_gis)
```

**Solutions:**

1. **Increase System Memory:**
```bash
# Add swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

2. **Optimize ML Models:**
```python
# Reduce batch size in configuration
ML_BATCH_SIZE = 8  # Reduce from default 32
ML_MAX_WORKERS = 2  # Reduce parallel workers
```

3. **Enable Memory Optimization:**
```bash
# Set Python memory optimization
export PYTHONOPTIMIZE=1
export MALLOC_TRIM_THRESHOLD_=100000
```

## Installation Problems

### Issue: pip install Failures

**Symptoms:**
- Package installation fails
- Compilation errors
- Dependency conflicts

**Diagnosis:**
```bash
# Check pip version
pip --version

# Check Python version compatibility
python --version

# Check available space
df -h /tmp

# Verbose installation for debugging
pip install -v mars-gis
```

**Solutions:**

1. **Upgrade pip and setuptools:**
```bash
python -m pip install --upgrade pip setuptools wheel
```

2. **Install system dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential python3-dev postgresql-dev gdal-bin libgdal-dev

# CentOS/RHEL
sudo yum install -y gcc gcc-c++ python3-devel postgresql-devel gdal-devel

# macOS
brew install postgresql gdal
```

3. **Use conda for complex dependencies:**
```bash
conda create -n mars-gis python=3.9
conda activate mars-gis
conda install -c conda-forge gdal postgresql
pip install mars-gis
```

### Issue: Docker Build Failures

**Symptoms:**
- Docker build fails
- Image size too large
- Layer caching issues

**Diagnosis:**
```bash
# Check Docker version
docker --version

# Check available disk space
docker system df

# Build with detailed output
docker build --no-cache --progress=plain -t mars-gis .
```

**Solutions:**

1. **Clean Docker Cache:**
```bash
# Remove unused containers and images
docker system prune -f

# Remove unused volumes
docker volume prune -f
```

2. **Optimize Dockerfile:**
```dockerfile
# Use multi-stage build
FROM python:3.9-slim as builder
# Install dependencies in builder stage

FROM python:3.9-slim as runtime
# Copy only necessary files from builder
```

3. **Increase Docker Resources:**
```bash
# Increase memory limit in Docker Desktop
# Or for Linux:
sudo systemctl edit docker
# Add:
# [Service]
# LimitMEMLOCK=infinity
```

## Runtime Errors

### Issue: FastAPI Server Errors

**Symptoms:**
- HTTP 500 Internal Server Error
- Unhandled exceptions
- Request timeouts

**Diagnosis:**
```bash
# Check server logs
tail -f logs/uvicorn.log

# Test API endpoints
curl -X GET "http://localhost:8000/health" -v

# Check FastAPI docs
curl "http://localhost:8000/docs"
```

**Solutions:**

1. **Check Error Logs:**
```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export MARS_GIS_DEBUG=true
```

2. **Validate Request Data:**
```python
# Test with curl
curl -X POST "http://localhost:8000/api/v1/missions" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Mission",
    "mission_type": "rover",
    "target_coordinates": [-14.5684, 175.4729]
  }'
```

3. **Check Dependencies:**
```bash
# Verify all required services are running
docker-compose ps
kubectl get services
```

### Issue: Authentication Failures

**Symptoms:**
- HTTP 401 Unauthorized
- Token validation errors
- Session timeouts

**Diagnosis:**
```bash
# Test authentication endpoint
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'

# Validate token
curl -X GET "http://localhost:8000/api/v1/missions" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Solutions:**

1. **Check JWT Configuration:**
```python
# Verify JWT settings
from mars_gis.core.config import settings
print(f"Secret Key: {settings.SECRET_KEY[:10]}...")
print(f"Token Expiry: {settings.ACCESS_TOKEN_EXPIRE_MINUTES}")
```

2. **Reset User Password:**
```bash
# Using management command
python -m mars_gis.cli reset-password --username admin
```

3. **Check Token Expiration:**
```python
# Decode JWT token
import jwt
token = "your_jwt_token"
decoded = jwt.decode(token, options={"verify_signature": False})
print(f"Expires: {decoded.get('exp')}")
```

## Performance Issues

### Issue: Slow API Response Times

**Symptoms:**
- Request timeouts
- High response latency
- Poor throughput

**Diagnosis:**
```bash
# Benchmark API endpoints
ab -n 100 -c 10 http://localhost:8000/api/v1/missions

# Profile with cProfile
python -m cProfile -o profile.stats -m mars_gis.main

# Monitor database queries
# Enable query logging in PostgreSQL
# log_statement = 'all' in postgresql.conf
```

**Solutions:**

1. **Database Optimization:**
```sql
-- Add missing indexes
CREATE INDEX idx_missions_status ON missions(status);
CREATE INDEX idx_missions_coordinates ON missions USING GIST(target_coordinates);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM missions WHERE status = 'active';
```

2. **Enable Caching:**
```python
# Redis caching configuration
CACHE_CONFIG = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://localhost:6379/1",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    }
}
```

3. **Optimize ML Inference:**
```python
# Batch predictions
async def batch_predict(requests):
    batch_size = 8
    results = []
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]
        batch_results = await model.predict_batch(batch)
        results.extend(batch_results)
    return results
```

### Issue: High Memory Usage

**Symptoms:**
- Memory leaks
- Out of memory errors
- Increasing memory usage over time

**Diagnosis:**
```bash
# Monitor memory usage
watch -n 1 'ps aux --sort=-%mem | head -10'

# Python memory profiling
pip install memory-profiler
python -m memory_profiler script.py

# Check for memory leaks
valgrind --tool=memcheck python -m mars_gis.main
```

**Solutions:**

1. **Implement Memory Management:**
```python
# Explicit garbage collection
import gc
gc.collect()

# Use context managers for resources
async with database.transaction():
    # Database operations
    pass
```

2. **Optimize Data Structures:**
```python
# Use generators instead of lists
def process_large_dataset():
    for item in database.stream_results():
        yield process_item(item)

# Clear caches periodically
@periodic_task(run_every=timedelta(hours=1))
def clear_caches():
    cache.clear()
```

3. **Configure Memory Limits:**
```yaml
# Docker container limits
version: '3.8'
services:
  mars-gis:
    image: mars-gis:latest
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

## Database Problems

### Issue: Migration Failures

**Symptoms:**
- Migration errors
- Schema inconsistencies
- Data corruption

**Diagnosis:**
```bash
# Check migration status
python -m alembic current
python -m alembic history

# Validate database schema
python -c "
from mars_gis.core.database import engine
from sqlalchemy import inspect
inspector = inspect(engine)
print(inspector.get_table_names())
"
```

**Solutions:**

1. **Fix Failed Migrations:**
```bash
# Rollback to last working migration
python -m alembic downgrade -1

# Mark migration as applied without running
python -m alembic stamp head

# Generate new migration
python -m alembic revision --autogenerate -m "fix_schema"
```

2. **Repair Database Schema:**
```sql
-- Check for missing tables
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public';

-- Repair corrupted indexes
REINDEX DATABASE mars_gis_db;

-- Check constraints
SELECT conname, contype FROM pg_constraint WHERE contype = 'c';
```

3. **Backup and Restore:**
```bash
# Create backup before repairs
pg_dump -h localhost -U mars_gis_user mars_gis_db > backup.sql

# Restore from backup if needed
dropdb mars_gis_db
createdb mars_gis_db
psql -h localhost -U mars_gis_user -d mars_gis_db < backup.sql
```

### Issue: Database Performance

**Symptoms:**
- Slow queries
- Connection timeouts
- High CPU usage

**Diagnosis:**
```sql
-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- Check database size
SELECT pg_size_pretty(pg_database_size('mars_gis_db'));

-- Check index usage
SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_tup_read DESC;
```

**Solutions:**

1. **Optimize Queries:**
```sql
-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_missions_created_at
ON missions(created_at);

-- Optimize joins
EXPLAIN (ANALYZE, BUFFERS)
SELECT m.*, u.username
FROM missions m
JOIN users u ON m.user_id = u.id;
```

2. **Configure PostgreSQL:**
```bash
# Edit postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Restart PostgreSQL
sudo systemctl restart postgresql
```

3. **Connection Pooling:**
```python
# Configure connection pool
DATABASE_CONFIG = {
    "min_size": 10,
    "max_size": 20,
    "max_queries": 50000,
    "max_inactive_connection_lifetime": 300
}
```

## API Issues

### Issue: CORS Errors

**Symptoms:**
- Cross-origin request blocked
- OPTIONS request failures
- Frontend cannot access API

**Diagnosis:**
```bash
# Test CORS headers
curl -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: X-Requested-With" \
  -X OPTIONS \
  http://localhost:8000/api/v1/missions
```

**Solutions:**

1. **Configure CORS Settings:**
```python
# In FastAPI application
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. **Nginx Configuration:**
```nginx
server {
    location /api/ {
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
        add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization";

        if ($request_method = 'OPTIONS') {
            return 204;
        }

        proxy_pass http://backend;
    }
}
```

### Issue: Rate Limiting Problems

**Symptoms:**
- HTTP 429 Too Many Requests
- API quota exceeded
- Legitimate requests blocked

**Diagnosis:**
```bash
# Check rate limit headers
curl -I "http://localhost:8000/api/v1/missions"

# Test rate limiting
for i in {1..100}; do
  curl -w "%{http_code}\n" -o /dev/null -s "http://localhost:8000/health"
done
```

**Solutions:**

1. **Adjust Rate Limits:**
```python
# Configure rate limiting
RATE_LIMIT_CONFIG = {
    "default": "100/minute",
    "burst": "200/minute",
    "authenticated": "1000/minute"
}
```

2. **Implement Rate Limit Exemptions:**
```python
# Whitelist internal services
RATE_LIMIT_WHITELIST = [
    "127.0.0.1",
    "10.0.0.0/8",
    "192.168.0.0/16"
]
```

## Frontend Problems

### Issue: Build Failures

**Symptoms:**
- Webpack compilation errors
- TypeScript errors
- Missing dependencies

**Diagnosis:**
```bash
# Check Node.js version
node --version
npm --version

# Clear cache
npm cache clean --force
rm -rf node_modules package-lock.json
npm install

# Build with verbose output
npm run build -- --verbose
```

**Solutions:**

1. **Fix Dependency Issues:**
```bash
# Update dependencies
npm update

# Audit and fix vulnerabilities
npm audit fix

# Install specific versions
npm install react@^18.0.0
```

2. **TypeScript Errors:**
```bash
# Check TypeScript configuration
npx tsc --noEmit

# Generate types for API
npm run generate-api-types
```

3. **Memory Issues:**
```bash
# Increase Node.js memory limit
export NODE_OPTIONS="--max-old-space-size=8192"
npm run build
```

### Issue: Runtime JavaScript Errors

**Symptoms:**
- Console errors
- White screen of death
- Component crashes

**Diagnosis:**
```bash
# Check browser console
# Open Developer Tools (F12)
# Look for JavaScript errors in Console tab

# Check network requests
# Look for failed API calls in Network tab

# Check source maps
# Verify source maps are working for debugging
```

**Solutions:**

1. **Error Boundaries:**
```typescript
class ErrorBoundary extends React.Component {
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('React Error Boundary:', error, errorInfo);
    // Send to error reporting service
  }

  render() {
    if (this.state.hasError) {
      return <div>Something went wrong.</div>;
    }
    return this.props.children;
  }
}
```

2. **API Error Handling:**
```typescript
const apiClient = {
  async request(url: string, options: RequestInit) {
    try {
      const response = await fetch(url, options);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
};
```

## ML Model Issues

### Issue: Model Loading Failures

**Symptoms:**
- Model not found errors
- Loading timeouts
- Memory errors during loading

**Diagnosis:**
```python
# Check model files
import os
model_path = "/path/to/models"
print(os.listdir(model_path))

# Test model loading
import torch
try:
    model = torch.load("model.pth")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
```

**Solutions:**

1. **Verify Model Paths:**
```python
# Check model configuration
from mars_gis.core.config import settings
print(f"Model path: {settings.ML_MODELS_PATH}")
print(f"Model exists: {os.path.exists(settings.ML_MODELS_PATH)}")
```

2. **Download Missing Models:**
```bash
# Download pre-trained models
python -m mars_gis.ml.download_models --all

# Or download specific model
python -m mars_gis.ml.download_models --model terrain_classifier
```

3. **Handle Model Loading Errors:**
```python
import torch
import logging

def load_model_safe(model_path: str):
    try:
        # Try loading on GPU first
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device)
        return model
    except RuntimeError as e:
        if "CUDA" in str(e):
            # Fallback to CPU
            logging.warning("CUDA error, falling back to CPU")
            model = torch.load(model_path, map_location="cpu")
            return model
        raise
```

### Issue: Inference Failures

**Symptoms:**
- Prediction errors
- Invalid input errors
- Model outputs NaN values

**Diagnosis:**
```python
# Test model inference
import numpy as np
import torch

# Create test input
test_input = torch.randn(1, 3, 512, 512)
print(f"Input shape: {test_input.shape}")
print(f"Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")

# Test model forward pass
with torch.no_grad():
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
```

**Solutions:**

1. **Input Validation:**
```python
def validate_input(input_tensor: torch.Tensor) -> torch.Tensor:
    # Check shape
    if input_tensor.shape != (1, 3, 512, 512):
        raise ValueError(f"Expected shape (1, 3, 512, 512), got {input_tensor.shape}")

    # Check for NaN values
    if torch.isnan(input_tensor).any():
        raise ValueError("Input contains NaN values")

    # Normalize input
    input_tensor = (input_tensor - input_tensor.mean()) / input_tensor.std()
    return input_tensor
```

2. **Model Debugging:**
```python
# Enable model debugging
torch.autograd.set_detect_anomaly(True)

# Check model gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

## Deployment Troubleshooting

### Issue: Docker Container Failures

**Symptoms:**
- Container won't start
- Container exits immediately
- Health check failures

**Diagnosis:**
```bash
# Check container logs
docker logs mars-gis-container

# Check container status
docker ps -a

# Inspect container configuration
docker inspect mars-gis-container

# Test container interactively
docker run -it --rm mars-gis:latest /bin/bash
```

**Solutions:**

1. **Fix Container Startup:**
```dockerfile
# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Proper signal handling
ENTRYPOINT ["python", "-m", "mars_gis.main"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
```

2. **Environment Configuration:**
```bash
# Check environment variables
docker exec mars-gis-container env | grep MARS_GIS

# Set required variables
docker run -e DATABASE_URL="postgresql://..." \
  -e REDIS_URL="redis://..." \
  mars-gis:latest
```

### Issue: Kubernetes Deployment Problems

**Symptoms:**
- Pods not starting
- Service unreachable
- ConfigMap/Secret issues

**Diagnosis:**
```bash
# Check pod status
kubectl get pods -l app=mars-gis

# Check pod logs
kubectl logs -l app=mars-gis --tail=100

# Describe pod for events
kubectl describe pod <pod-name>

# Check services
kubectl get services
```

**Solutions:**

1. **Fix Pod Issues:**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mars-gis
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: mars-gis
        image: mars-gis:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
```

2. **Network Issues:**
```bash
# Test service connectivity
kubectl port-forward service/mars-gis 8000:8000

# Check ingress
kubectl get ingress
kubectl describe ingress mars-gis-ingress
```

## Monitoring and Logs

### Log Locations

**Application Logs:**
```bash
# Standard log locations
/var/log/mars-gis/app.log          # Application logs
/var/log/mars-gis/access.log       # Access logs
/var/log/mars-gis/error.log        # Error logs

# Docker logs
docker logs mars-gis-container

# Kubernetes logs
kubectl logs -f deployment/mars-gis
```

**System Logs:**
```bash
# System journal
journalctl -u mars-gis.service -f

# Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log

# PostgreSQL logs
tail -f /var/log/postgresql/postgresql-*.log
```

### Log Analysis

**Search for Errors:**
```bash
# Find recent errors
grep -i error /var/log/mars-gis/app.log | tail -20

# Find database errors
grep -i "database\|sql" /var/log/mars-gis/error.log

# Find authentication errors
grep -i "auth\|token\|unauthorized" /var/log/mars-gis/access.log
```

**Performance Analysis:**
```bash
# Find slow requests
awk '$NF > 1000' /var/log/mars-gis/access.log  # Requests > 1s

# Top error codes
awk '{print $9}' /var/log/nginx/access.log | sort | uniq -c | sort -nr

# Memory usage patterns
grep -E "memory|oom" /var/log/syslog
```

### Monitoring Setup

**Basic Monitoring:**
```bash
# Check system resources
htop
iotop
nethogs

# Monitor application
watch -n 5 'curl -s http://localhost:8000/health | jq'

# Database monitoring
watch -n 10 'psql -d mars_gis_db -c "SELECT count(*) FROM pg_stat_activity;"'
```

**Advanced Monitoring:**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
- job_name: 'mars-gis'
  static_configs:
  - targets: ['localhost:8000']
  metrics_path: /metrics
```

## Getting Help

### Before Seeking Help

1. **Check this troubleshooting guide**
2. **Search existing issues** in the project repository
3. **Check system logs** for error messages
4. **Verify your configuration** matches requirements
5. **Test with minimal configuration**

### Information to Provide

When reporting issues, include:

- **MARS-GIS version**: `python -c "import mars_gis; print(mars_gis.__version__)"`
- **Python version**: `python --version`
- **Operating system**: `uname -a` (Linux/macOS) or system info (Windows)
- **Installation method**: pip, Docker, Kubernetes, source
- **Configuration**: Sanitized configuration (remove secrets)
- **Error messages**: Complete error messages and stack traces
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Expected vs actual behavior**

### Support Channels

1. **GitHub Issues**: [Report bugs and feature requests](https://github.com/your-org/mars-gis/issues)
2. **Documentation**: [Check the complete documentation](https://mars-gis.readthedocs.io)
3. **Community Forum**: [Ask questions and share solutions](https://community.mars-gis.org)
4. **Stack Overflow**: Tag questions with `mars-gis`

### Emergency Contacts

**For Production Issues:**
- **Security Issues**: security@mars-gis.org
- **Critical Bugs**: critical@mars-gis.org
- **Enterprise Support**: support@mars-gis.org

---

**Quick Reference Commands**

```bash
# Health Check
curl http://localhost:8000/health

# View Logs
tail -f logs/mars_gis.log

# Check Database
psql -d mars_gis_db -c "SELECT version();"

# Restart Services
systemctl restart mars-gis
docker-compose restart
kubectl rollout restart deployment/mars-gis

# Check Processes
ps aux | grep mars_gis
netstat -tlnp | grep :8000
```

**Remember**: Always backup your data before making configuration changes or attempting repairs!
