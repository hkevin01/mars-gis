#!/bin/bash

# Mars-GIS Development Scripts
# Collection of utility scripts for development workflow

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function for colored output
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

# Function to show usage
show_help() {
    echo "Mars-GIS Development Scripts"
    echo ""
    echo "Usage: ./scripts/dev.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup           - Initial project setup"
    echo "  dev             - Start development environment"
    echo "  test            - Run all tests"
    echo "  test:unit       - Run unit tests only"
    echo "  test:e2e        - Run E2E tests only"
    echo "  test:coverage   - Run tests with coverage"
    echo "  lint            - Run linting"
    echo "  format          - Format code"
    echo "  build           - Build for production"
    echo "  clean           - Clean up containers and volumes"
    echo "  logs            - Show logs"
    echo "  shell           - Open shell in container"
    echo "  db:migrate      - Run database migrations"
    echo "  db:seed         - Seed database with sample data"
    echo "  db:reset        - Reset database"
    echo "  backup          - Create database backup"
    echo "  docs            - Start documentation server"
    echo "  help            - Show this help message"
}

# Setup function
setup() {
    print_status "Setting up Mars-GIS development environment..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Create necessary directories
    mkdir -p backend/data backend/logs frontend/coverage backend/coverage
    mkdir -p database/init monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources
    mkdir -p backups nginx/ssl

    # Copy environment files if they don't exist
    if [ ! -f .env ]; then
        print_status "Creating .env file..."
        cat > .env << EOF
# Mars-GIS Environment Variables
NODE_ENV=development
DATABASE_URL=postgresql://mars_user:mars_password@localhost:5432/mars_gis
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=development-secret-key-change-in-production
CORS_ORIGINS=http://localhost:3000,http://localhost:80
LOG_LEVEL=debug

# Frontend
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_WS_URL=ws://localhost:8000
REACT_APP_ENVIRONMENT=development

# Testing
CI=false
CYPRESS_BASE_URL=http://localhost:3000

# Optional: External APIs
# NASA_API_KEY=your_nasa_api_key
# MAPBOX_ACCESS_TOKEN=your_mapbox_token
EOF
        print_success "Created .env file. Please update it with your configuration."
    fi

    # Pull base images
    print_status "Pulling Docker images..."
    docker-compose pull

    print_success "Setup completed! Run './scripts/dev.sh dev' to start development."
}

# Development environment
dev() {
    print_status "Starting development environment..."
    docker-compose --profile development up -d

    print_success "Development environment started!"
    print_status "Frontend: http://localhost:3000"
    print_status "Backend: http://localhost:8000"
    print_status "Database: localhost:5432"
    print_status "Redis: localhost:6379"

    # Show logs
    docker-compose logs -f frontend-dev backend-dev
}

# Testing functions
test_all() {
    print_status "Running all tests..."

    # Run unit tests
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm frontend-unit-test
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm backend-test

    # Run integration tests
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm integration-tests

    # Run E2E tests
    docker-compose --profile testing run --rm e2e-tests

    print_success "All tests completed!"
}

test_unit() {
    print_status "Running unit tests..."
    docker-compose --profile testing run --rm frontend-test
    print_success "Unit tests completed!"
}

test_e2e() {
    print_status "Running E2E tests..."
    # Start development environment first
    docker-compose --profile development up -d
    sleep 10  # Wait for services to be ready

    # Run E2E tests
    docker-compose --profile testing run --rm e2e-tests

    print_success "E2E tests completed!"
}

test_coverage() {
    print_status "Running tests with coverage..."
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm frontend-unit-test npm run test:coverage
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm backend-test npm run test:coverage

    print_success "Coverage reports generated in ./frontend/coverage and ./backend/coverage"
}

# Linting and formatting
lint() {
    print_status "Running linters..."
    docker-compose run --rm frontend-dev npm run lint
    docker-compose run --rm backend-dev npm run lint
    print_success "Linting completed!"
}

format() {
    print_status "Formatting code..."
    docker-compose run --rm frontend-dev npm run format
    docker-compose run --rm backend-dev npm run format
    print_success "Code formatting completed!"
}

# Build for production
build() {
    print_status "Building for production..."
    docker-compose --profile production build
    print_success "Production build completed!"
}

# Cleanup
clean() {
    print_warning "This will remove all containers, images, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --rmi all
        docker system prune -f
        print_success "Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Show logs
logs() {
    if [ -n "$2" ]; then
        docker-compose logs -f "$2"
    else
        docker-compose logs -f
    fi
}

# Open shell
shell() {
    if [ -n "$2" ]; then
        docker-compose exec "$2" /bin/bash
    else
        docker-compose exec backend-dev /bin/bash
    fi
}

# Database operations
db_migrate() {
    print_status "Running database migrations..."
    docker-compose exec backend-dev npm run db:migrate
    print_success "Database migrations completed!"
}

db_seed() {
    print_status "Seeding database..."
    docker-compose exec backend-dev npm run db:seed
    print_success "Database seeded!"
}

db_reset() {
    print_warning "This will reset the database. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Resetting database..."
        docker-compose exec backend-dev npm run db:reset
        print_success "Database reset completed!"
    else
        print_status "Database reset cancelled."
    fi
}

# Backup database
backup() {
    print_status "Creating database backup..."
    timestamp=$(date +"%Y%m%d_%H%M%S")
    docker-compose exec database pg_dump -U mars_user mars_gis > "backups/backup_${timestamp}.sql"
    print_success "Backup created: backups/backup_${timestamp}.sql"
}

# Documentation
docs() {
    print_status "Starting documentation server..."
    docker-compose --profile documentation up -d storybook
    print_success "Documentation available at http://localhost:6006"
}

# Main command router
case "$1" in
    setup)
        setup
        ;;
    dev)
        dev
        ;;
    test)
        test_all
        ;;
    test:unit)
        test_unit
        ;;
    test:e2e)
        test_e2e
        ;;
    test:coverage)
        test_coverage
        ;;
    lint)
        lint
        ;;
    format)
        format
        ;;
    build)
        build
        ;;
    clean)
        clean
        ;;
    logs)
        logs "$@"
        ;;
    shell)
        shell "$@"
        ;;
    db:migrate)
        db_migrate
        ;;
    db:seed)
        db_seed
        ;;
    db:reset)
        db_reset
        ;;
    backup)
        backup
        ;;
    docs)
        docs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
