#!/bin/bash

# Mars GIS Platform Production Deployment Script
# This script deploys the Mars GIS Platform to production environment

set -e  # Exit on any error

# Configuration
DEPLOYMENT_ENV=${1:-production}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"mars-gis-registry"}
VERSION=${VERSION:-$(git rev-parse --short HEAD)}
BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        error "Docker is not running"
        exit 1
    fi
    
    # Check if required environment variables are set
    if [ -z "$DATABASE_URL" ]; then
        error "DATABASE_URL environment variable is not set"
        exit 1
    fi
    
    # Check if we're on the correct branch for production
    if [ "$DEPLOYMENT_ENV" = "production" ]; then
        current_branch=$(git branch --show-current)
        if [ "$current_branch" != "main" ]; then
            error "Production deployments must be from 'main' branch. Current branch: $current_branch"
            exit 1
        fi
    fi
    
    # Check if there are uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        warning "There are uncommitted changes. Consider committing them first."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Deployment cancelled"
            exit 1
        fi
    fi
    
    success "Pre-deployment checks passed"
}

# Create backup of current data
create_backup() {
    log "Creating backup of current data..."
    
    BACKUP_DIR="backups/$(date +'%Y%m%d_%H%M%S')"
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    if [ -n "$DATABASE_URL" ]; then
        log "Backing up database..."
        pg_dump "$DATABASE_URL" > "$BACKUP_DIR/database_backup.sql"
        success "Database backup created: $BACKUP_DIR/database_backup.sql"
    fi
    
    # Backup data volumes
    if docker volume ls | grep -q mars_gis_data; then
        log "Backing up data volumes..."
        docker run --rm -v mars_gis_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/data_volume.tar.gz -C /data .
        success "Data volume backup created: $BACKUP_DIR/data_volume.tar.gz"
    fi
    
    # Backup models
    if docker volume ls | grep -q mars_gis_models; then
        log "Backing up ML models..."
        docker run --rm -v mars_gis_models:/models -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/models_backup.tar.gz -C /models .
        success "Models backup created: $BACKUP_DIR/models_backup.tar.gz"
    fi
    
    echo "$BACKUP_DIR" > .last_backup
    success "Backup completed: $BACKUP_DIR"
}

# Build and push Docker images
build_and_push_images() {
    log "Building Docker images..."
    
    # Build production image
    docker build --target production -t "$DOCKER_REGISTRY/mars-gis:$VERSION" .
    docker build --target production -t "$DOCKER_REGISTRY/mars-gis:latest" .
    
    # Build development image (for testing)
    docker build --target development -t "$DOCKER_REGISTRY/mars-gis:dev-$VERSION" .
    
    success "Docker images built successfully"
    
    # Push to registry (if registry is configured)
    if [ "$DOCKER_REGISTRY" != "mars-gis-registry" ]; then
        log "Pushing images to registry..."
        docker push "$DOCKER_REGISTRY/mars-gis:$VERSION"
        docker push "$DOCKER_REGISTRY/mars-gis:latest"
        success "Images pushed to registry"
    fi
}

# Run tests before deployment
run_tests() {
    log "Running test suite..."
    
    # Start test environment
    docker-compose -f docker-compose.yml --profile testing up -d postgres redis
    
    # Wait for services to be ready
    sleep 10
    
    # Run tests
    if docker-compose --profile testing run --rm test-runner; then
        success "All tests passed"
    else
        error "Tests failed. Deployment aborted."
        docker-compose --profile testing down
        exit 1
    fi
    
    # Cleanup test environment
    docker-compose --profile testing down
}

# Deploy to environment
deploy() {
    log "Deploying to $DEPLOYMENT_ENV environment..."
    
    # Export version for docker-compose
    export MARS_GIS_VERSION="$VERSION"
    
    # Pull latest images
    if [ "$DOCKER_REGISTRY" != "mars-gis-registry" ]; then
        docker-compose pull
    fi
    
    # Deploy with zero-downtime strategy
    if [ "$DEPLOYMENT_ENV" = "production" ]; then
        # Production deployment with rolling update
        log "Performing rolling update..."
        
        # Update services one by one
        docker-compose up -d --scale mars-gis-api=2 mars-gis-api
        sleep 30  # Allow new instances to start
        
        # Stop old instances
        docker-compose up -d --scale mars-gis-api=1 mars-gis-api
        
        # Update other services
        docker-compose up -d nginx frontend jupyter
    else
        # Development/staging deployment
        docker-compose up -d
    fi
    
    success "Deployment completed"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Wait for database to be ready
    timeout 60 bash -c 'until docker-compose exec postgres pg_isready -U mars_user -d mars_gis; do sleep 2; done'
    
    # Run migrations
    docker-compose exec mars-gis-api python -m alembic upgrade head
    
    success "Database migrations completed"
}

# Health checks
health_checks() {
    log "Performing health checks..."
    
    # Check API health
    for i in {1..30}; do
        if curl -f http://localhost:8000/ > /dev/null 2>&1; then
            success "API health check passed"
            break
        fi
        if [ $i -eq 30 ]; then
            error "API health check failed after 30 attempts"
            return 1
        fi
        sleep 2
    done
    
    # Check database connectivity
    if docker-compose exec postgres pg_isready -U mars_user -d mars_gis > /dev/null 2>&1; then
        success "Database health check passed"
    else
        error "Database health check failed"
        return 1
    fi
    
    # Check Redis connectivity
    if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
        success "Redis health check passed"
    else
        error "Redis health check failed"
        return 1
    fi
    
    success "All health checks passed"
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    if [ -d "backups" ]; then
        find backups -type d -mtime +$BACKUP_RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
        success "Old backups cleaned up (retention: $BACKUP_RETENTION_DAYS days)"
    fi
}

# Rollback function
rollback() {
    log "Rolling back deployment..."
    
    if [ -f ".last_backup" ]; then
        BACKUP_DIR=$(cat .last_backup)
        
        if [ -d "$BACKUP_DIR" ]; then
            log "Restoring from backup: $BACKUP_DIR"
            
            # Stop services
            docker-compose down
            
            # Restore database
            if [ -f "$BACKUP_DIR/database_backup.sql" ]; then
                log "Restoring database..."
                psql "$DATABASE_URL" < "$BACKUP_DIR/database_backup.sql"
            fi
            
            # Restore data volumes
            if [ -f "$BACKUP_DIR/data_volume.tar.gz" ]; then
                log "Restoring data volume..."
                docker run --rm -v mars_gis_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar xzf /backup/data_volume.tar.gz -C /data
            fi
            
            # Restore models
            if [ -f "$BACKUP_DIR/models_backup.tar.gz" ]; then
                log "Restoring models..."
                docker run --rm -v mars_gis_models:/models -v "$(pwd)/$BACKUP_DIR":/backup alpine tar xzf /backup/models_backup.tar.gz -C /models
            fi
            
            # Restart services with previous version
            docker-compose up -d
            
            success "Rollback completed"
        else
            error "Backup directory not found: $BACKUP_DIR"
            exit 1
        fi
    else
        error "No backup information found. Cannot rollback."
        exit 1
    fi
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo
    echo "Environment: $DEPLOYMENT_ENV"
    echo "Version: $VERSION"
    echo "Services:"
    docker-compose ps
    echo
    echo "Health Status:"
    curl -s http://localhost:8000/ | jq . 2>/dev/null || echo "API not responding"
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        "deploy")
            pre_deployment_checks
            create_backup
            build_and_push_images
            run_tests
            deploy
            run_migrations
            health_checks
            cleanup_old_backups
            show_status
            success "🚀 Mars GIS Platform deployed successfully!"
            ;;
        "rollback")
            rollback
            ;;
        "status")
            show_status
            ;;
        "backup")
            create_backup
            ;;
        "test")
            run_tests
            ;;
        "build")
            build_and_push_images
            ;;
        *)
            echo "Usage: $0 {deploy|rollback|status|backup|test|build} [environment]"
            echo
            echo "Commands:"
            echo "  deploy   - Full deployment process (default)"
            echo "  rollback - Rollback to previous backup"
            echo "  status   - Show current deployment status"
            echo "  backup   - Create backup only"
            echo "  test     - Run tests only"
            echo "  build    - Build and push images only"
            echo
            echo "Environment: production|staging|development (default: production)"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
