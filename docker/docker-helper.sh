#!/bin/bash

# MARS-GIS Docker Management Script
# Usage: ./docker-helper.sh [command] [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_DIR="docker/compose"
PROJECT_NAME="mars-gis"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
}

check_docker_dir() {
    if [ ! -d "$DOCKER_DIR" ]; then
        log_error "Docker directory not found: $DOCKER_DIR"
        exit 1
    fi
}

# Command functions
cmd_help() {
    cat << EOF
MARS-GIS Docker Management Script

Usage: $0 [command] [options]

Commands:
    dev         Start development environment
    prod        Start production environment
    test        Run test environment
    stop        Stop all services
    down        Stop and remove all containers
    clean       Clean up containers, volumes, and images
    logs        Show logs for services
    build       Build all or specific services
    status      Show status of all services
    shell       Open shell in a service container
    db          Database management commands
    backup      Create database backup
    restore     Restore database from backup
    help        Show this help message

Examples:
    $0 dev                    # Start development environment
    $0 prod                   # Start production environment
    $0 test                   # Run tests
    $0 logs backend-dev       # Show backend development logs
    $0 shell backend-dev      # Open shell in backend container
    $0 build frontend         # Build frontend service
    $0 db migrate            # Run database migrations
    $0 backup                # Create database backup
    $0 clean                 # Clean up everything

EOF
}

cmd_dev() {
    log_info "Starting development environment..."
    cd "$DOCKER_DIR"
    docker-compose --profile development up -d
    log_success "Development environment started!"
    log_info "Services available at:"
    log_info "  Frontend: http://localhost:3000"
    log_info "  Backend API: http://localhost:8000"
    log_info "  Database: localhost:5432"
    log_info "  Redis: localhost:6379"
    log_info "  Grafana: http://localhost:3001"
    log_info "  Prometheus: http://localhost:9090"
}

cmd_prod() {
    log_info "Starting production environment..."
    cd "$DOCKER_DIR"
    docker-compose --profile production -f docker-compose.yml -f docker-compose.prod.yml up -d
    log_success "Production environment started!"
    log_info "Services available at:"
    log_info "  Frontend: http://localhost:80"
    log_info "  Backend API: http://localhost:8080"
}

cmd_test() {
    log_info "Running test environment..."
    cd "$DOCKER_DIR"
    docker-compose -f docker-compose.test.yml up --abort-on-container-exit
    log_success "Tests completed!"
}

cmd_stop() {
    log_info "Stopping all services..."
    cd "$DOCKER_DIR"
    docker-compose stop
    docker-compose -f docker-compose.test.yml stop 2>/dev/null || true
    log_success "All services stopped!"
}

cmd_down() {
    log_info "Stopping and removing all containers..."
    cd "$DOCKER_DIR"
    docker-compose down
    docker-compose -f docker-compose.test.yml down 2>/dev/null || true
    log_success "All containers removed!"
}

cmd_clean() {
    log_warning "This will remove all containers, volumes, and images for $PROJECT_NAME"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleaning up Docker resources..."
        cd "$DOCKER_DIR"

        # Stop and remove containers
        docker-compose down -v --remove-orphans 2>/dev/null || true
        docker-compose -f docker-compose.test.yml down -v --remove-orphans 2>/dev/null || true

        # Remove project images
        docker images | grep "$PROJECT_NAME" | awk '{print $3}' | xargs docker rmi -f 2>/dev/null || true

        # Remove unused volumes
        docker volume prune -f

        # Remove unused networks
        docker network prune -f

        log_success "Cleanup completed!"
    else
        log_info "Cleanup cancelled."
    fi
}

cmd_logs() {
    local service=${1:-}
    cd "$DOCKER_DIR"

    if [ -z "$service" ]; then
        log_info "Showing logs for all services..."
        docker-compose logs -f
    else
        log_info "Showing logs for service: $service"
        docker-compose logs -f "$service"
    fi
}

cmd_build() {
    local service=${1:-}
    cd "$DOCKER_DIR"

    if [ -z "$service" ]; then
        log_info "Building all services..."
        docker-compose build
    else
        log_info "Building service: $service"
        docker-compose build "$service"
    fi
    log_success "Build completed!"
}

cmd_status() {
    log_info "Checking service status..."
    cd "$DOCKER_DIR"
    docker-compose ps
}

cmd_shell() {
    local service=${1:-backend-dev}
    cd "$DOCKER_DIR"

    log_info "Opening shell in service: $service"
    docker-compose exec "$service" /bin/bash || docker-compose exec "$service" /bin/sh
}

cmd_db() {
    local action=${1:-connect}
    cd "$DOCKER_DIR"

    case $action in
        connect)
            log_info "Connecting to database..."
            docker-compose exec database psql -U mars_user -d mars_gis
            ;;
        migrate)
            log_info "Running database migrations..."
            docker-compose exec backend-dev python manage.py migrate
            ;;
        seed)
            log_info "Seeding database with test data..."
            docker-compose exec backend-dev python manage.py seed
            ;;
        reset)
            log_warning "This will reset the database and lose all data!"
            read -p "Are you sure? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                docker-compose down
                docker volume rm mars-gis_postgres_data 2>/dev/null || true
                docker-compose up -d database
                log_success "Database reset completed!"
            fi
            ;;
        *)
            log_error "Unknown database action: $action"
            log_info "Available actions: connect, migrate, seed, reset"
            ;;
    esac
}

cmd_backup() {
    local backup_name="backup_$(date +%Y%m%d_%H%M%S).sql"
    cd "$DOCKER_DIR"

    log_info "Creating database backup: $backup_name"
    mkdir -p ../../backups
    docker-compose exec -T database pg_dump -U mars_user -d mars_gis > "../../backups/$backup_name"
    log_success "Backup created: backups/$backup_name"
}

cmd_restore() {
    local backup_file=${1:-}

    if [ -z "$backup_file" ]; then
        log_error "Please specify a backup file"
        log_info "Usage: $0 restore <backup_file>"
        return 1
    fi

    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi

    cd "$DOCKER_DIR"
    log_warning "This will restore the database and lose current data!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Restoring database from: $backup_file"
        docker-compose exec -T database psql -U mars_user -d mars_gis < "$backup_file"
        log_success "Database restored!"
    fi
}

# Main script logic
main() {
    check_docker
    check_docker_dir

    local command=${1:-help}
    shift || true

    case $command in
        dev|development)
            cmd_dev "$@"
            ;;
        prod|production)
            cmd_prod "$@"
            ;;
        test|testing)
            cmd_test "$@"
            ;;
        stop)
            cmd_stop "$@"
            ;;
        down)
            cmd_down "$@"
            ;;
        clean)
            cmd_clean "$@"
            ;;
        logs)
            cmd_logs "$@"
            ;;
        build)
            cmd_build "$@"
            ;;
        status)
            cmd_status "$@"
            ;;
        shell)
            cmd_shell "$@"
            ;;
        db)
            cmd_db "$@"
            ;;
        backup)
            cmd_backup "$@"
            ;;
        restore)
            cmd_restore "$@"
            ;;
        help|--help|-h)
            cmd_help
            ;;
        *)
            log_error "Unknown command: $command"
            cmd_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
