# Docker Configuration for MARS-GIS

This directory contains all Docker-related files for the MARS-GIS project, organized for easy maintenance and deployment.

## 📁 Directory Structure

```
docker/
├── backend/
│   └── Dockerfile              # Backend application container
├── frontend/
│   └── Dockerfile              # Frontend application container
├── compose/
│   ├── docker-compose.yml      # Main development environment
│   ├── docker-compose.prod.yml # Production environment overrides
│   ├── docker-compose.test.yml # Testing environment
│   └── docker-compose.override.yml # Local development overrides
└── README.md                   # This file
```

## 🚀 Quick Start

### Development Environment

```bash
# From project root
cd docker/compose
docker-compose up -d

# Or run specific profiles
docker-compose --profile development up -d
```

### Production Environment

```bash
cd docker/compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Testing Environment

```bash
cd docker/compose
docker-compose -f docker-compose.test.yml up -d
```

## 🏗️ Container Architecture

### Backend Container (`docker/backend/Dockerfile`)

**Multi-stage build with targets:**
- `base`: Common dependencies and system setup
- `development`: Development tools and hot-reload
- `testing`: Test runners and coverage tools
- `production`: Optimized for production deployment

**Features:**
- Python 3.11 with geospatial dependencies (GDAL, PostGIS)
- Non-root user for security
- Health checks
- Volume mounts for development
- Gunicorn for production

### Frontend Container (`docker/frontend/Dockerfile`)

**Multi-stage build with targets:**
- `base`: Node.js environment setup
- `dependencies`: Production dependencies
- `dev-dependencies`: Development dependencies
- `testing`: Test environment with coverage
- `development`: Hot-reload development server
- `builder`: Production build creation
- `production`: Nginx-served production build
- `e2e-testing`: Cypress E2E testing

**Features:**
- Node.js 18 Alpine
- Nginx for production serving
- Cypress for E2E testing
- Hot-reload for development
- Security hardening

## 🐳 Docker Compose Configurations

### Main Configuration (`docker-compose.yml`)

**Services:**
- `database`: PostgreSQL with PostGIS extension
- `redis`: Redis for caching and sessions
- `backend-dev`: Development backend API
- `backend`: Production backend API
- `frontend-dev`: Development React server
- `frontend`: Production Nginx-served frontend
- `worker`: Celery background worker
- `scheduler`: Celery beat scheduler
- `prometheus`: Monitoring and metrics
- `grafana`: Visualization dashboards

### Production Overrides (`docker-compose.prod.yml`)

**Additional Services:**
- `nginx`: Load balancer and reverse proxy
- `backup`: Automated database backups

**Features:**
- Resource limits and reservations
- Multi-replica deployment
- SSL/TLS termination
- Automated backups

### Testing Configuration (`docker-compose.test.yml`)

**Isolated Testing Environment:**
- `test-database`: Isolated test database
- `test-redis`: Isolated test cache
- `backend-test`: Backend unit/integration tests
- `frontend-unit-test`: Frontend unit tests
- `integration-tests`: End-to-end integration tests

### Development Overrides (`docker-compose.override.yml`)

**Development Tools:**
- Hot-reload for backend and frontend
- Volume mounts for fast development
- `mailhog`: Email testing service

## 📋 Environment Profiles

### Development Profile
```bash
docker-compose --profile development up -d
```
**Includes:** backend-dev, frontend-dev, database, redis, storybook

### Production Profile
```bash
docker-compose --profile production up -d
```
**Includes:** backend, frontend, database, redis, worker, scheduler

### Testing Profile
```bash
docker-compose --profile testing up -d
```
**Includes:** All testing services with isolated environments

### Documentation Profile
```bash
docker-compose --profile documentation up -d
```
**Includes:** Storybook documentation server

## 🔧 Environment Variables

### Backend Environment
```env
NODE_ENV=development|production|test
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port/db
SECRET_KEY=your-secret-key
DEBUG=true|false
ALLOWED_HOSTS=comma,separated,hosts
CORS_ORIGINS=comma,separated,origins
```

### Frontend Environment
```env
NODE_ENV=development|production|test
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_WS_URL=ws://localhost:8000
CHOKIDAR_USEPOLLING=true
```

## 🏥 Health Checks

All services include comprehensive health checks:

- **Backend**: HTTP endpoint check (`/health`)
- **Frontend**: HTTP endpoint check (`/`)
- **Database**: PostgreSQL ready check
- **Redis**: Ping command
- **Worker**: Celery inspect ping

## 📊 Monitoring and Observability

### Prometheus (Port 9090)
- Application metrics collection
- Service discovery
- Alert rules

### Grafana (Port 3001)
- Dashboard visualization
- Alert management
- Multi-datasource support

## 🛡️ Security Features

### Container Security
- Non-root users in all containers
- Read-only root filesystems where possible
- Resource limits and reservations
- Security scanning integration

### Network Security
- Isolated bridge networks
- Service-to-service communication
- No unnecessary port exposure

## 🔄 CI/CD Integration

### GitHub Actions Integration
```yaml
# .github/workflows/docker.yml
- name: Build and Test
  run: |
    cd docker/compose
    docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Build Optimization
- Multi-stage builds for minimal production images
- Layer caching for faster builds
- BuildKit support for advanced features

## 📝 Common Commands

### Build Services
```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build backend-dev

# Build with no cache
docker-compose build --no-cache
```

### Manage Services
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f backend-dev

# Execute commands in containers
docker-compose exec backend-dev bash
docker-compose exec database psql -U mars_user -d mars_gis

# Scale services
docker-compose up -d --scale backend=3
```

### Cleanup
```bash
# Stop and remove containers
docker-compose down

# Remove volumes
docker-compose down -v

# Remove everything including images
docker-compose down --rmi all -v
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000

   # Modify ports in docker-compose.yml
   ports:
     - "8001:8000"  # Host:Container
   ```

2. **Volume Permission Issues**
   ```bash
   # Fix ownership
   docker-compose exec backend-dev chown -R mars:mars /app
   ```

3. **Database Connection Issues**
   ```bash
   # Check database health
   docker-compose exec database pg_isready -U mars_user

   # Reset database
   docker-compose down -v
   docker-compose up -d database
   ```

4. **Frontend Build Issues**
   ```bash
   # Clear node_modules and rebuild
   docker-compose down
   docker volume rm $(docker volume ls -q | grep node_modules)
   docker-compose build --no-cache frontend-dev
   ```

### Performance Optimization

1. **Enable Docker BuildKit**
   ```bash
   export DOCKER_BUILDKIT=1
   export COMPOSE_DOCKER_CLI_BUILD=1
   ```

2. **Use Docker Desktop for Mac/Windows**
   - Enable VirtioFS for better volume performance
   - Allocate sufficient resources (CPU/Memory)

3. **Production Optimization**
   - Use multi-stage builds
   - Implement health checks
   - Configure resource limits

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [PostgreSQL Docker Hub](https://hub.docker.com/_/postgres)
- [Nginx Docker Hub](https://hub.docker.com/_/nginx)
- [Redis Docker Hub](https://hub.docker.com/_/redis)
