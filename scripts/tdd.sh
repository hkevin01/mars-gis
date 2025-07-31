#!/bin/bash

# Mars-GIS TDD Testing Suite Runner
# Comprehensive Test-Driven Development workflow executor

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# TDD Phase tracking
CURRENT_PHASE=""

print_tdd_phase() {
    local phase=$1
    local description=$2

    case $phase in
        "RED")
            echo -e "${RED}🔴 RED PHASE: ${description}${NC}"
            ;;
        "GREEN")
            echo -e "${GREEN}🟢 GREEN PHASE: ${description}${NC}"
            ;;
        "REFACTOR")
            echo -e "${BLUE}🔵 REFACTOR PHASE: ${description}${NC}"
            ;;
        *)
            echo -e "${CYAN}ℹ️  ${description}${NC}"
            ;;
    esac
    CURRENT_PHASE=$phase
}

print_status() {
    echo -e "${PURPLE}[TDD]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ SUCCESS:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  WARNING:${NC} $1"
}

print_error() {
    echo -e "${RED}❌ ERROR:${NC} $1"
}

# Show TDD workflow help
show_tdd_help() {
    echo "Mars-GIS TDD Testing Suite"
    echo ""
    echo "Usage: ./scripts/tdd.sh [COMMAND]"
    echo ""
    echo "TDD Workflow Commands:"
    echo "  red              - Run RED phase: Write failing tests"
    echo "  green            - Run GREEN phase: Make tests pass"
    echo "  refactor         - Run REFACTOR phase: Improve code"
    echo "  cycle            - Run complete TDD cycle (RED → GREEN → REFACTOR)"
    echo ""
    echo "Testing Commands:"
    echo "  test:all         - Run all test suites"
    echo "  test:unit        - Run unit tests only"
    echo "  test:integration - Run integration tests"
    echo "  test:e2e         - Run E2E tests with Cypress"
    echo "  test:coverage    - Generate coverage reports"
    echo "  test:performance - Run performance tests"
    echo "  test:a11y        - Run accessibility tests"
    echo ""
    echo "Development Commands:"
    echo "  setup            - Initial TDD environment setup"
    echo "  dev              - Start development environment"
    echo "  watch            - Start test watcher mode"
    echo "  lint             - Run linting and type checking"
    echo "  format           - Format code"
    echo ""
    echo "Validation Commands:"
    echo "  validate         - Validate TDD compliance"
    echo "  metrics          - Show TDD metrics"
    echo "  health           - Check testing environment health"
    echo ""
    echo "Cleanup Commands:"
    echo "  clean            - Clean test artifacts"
    echo "  reset            - Reset to clean state"
}

# TDD Phase Functions
run_red_phase() {
    print_tdd_phase "RED" "Writing failing tests first"

    print_status "Creating failing unit tests..."
    docker-compose run --rm frontend-test npm run test:unit -- --testNamePattern="RED" --passWithNoTests

    print_status "Creating failing integration tests..."
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm integration-tests || true

    print_status "Creating failing E2E tests..."
    docker-compose --profile testing run --rm e2e-tests || true

    print_success "RED phase completed - All tests should be failing!"
    print_warning "Next: Run './scripts/tdd.sh green' to make tests pass"
}

run_green_phase() {
    print_tdd_phase "GREEN" "Making tests pass with minimal code"

    print_status "Running frontend tests..."
    docker-compose run --rm frontend-test npm run test:coverage

    print_status "Running backend tests..."
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm backend-test

    print_status "Running integration tests..."
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm integration-tests

    print_status "Running E2E tests..."
    docker-compose --profile development up -d
    sleep 30
    docker-compose --profile testing run --rm e2e-tests

    print_success "GREEN phase completed - All tests should be passing!"
    print_warning "Next: Run './scripts/tdd.sh refactor' to improve code quality"
}

run_refactor_phase() {
    print_tdd_phase "REFACTOR" "Improving code without changing functionality"

    print_status "Running linting..."
    docker-compose run --rm frontend-dev npm run lint
    docker-compose run --rm backend-dev npm run lint

    print_status "Running type checking..."
    docker-compose run --rm frontend-dev npm run type-check
    docker-compose run --rm backend-dev npm run type-check

    print_status "Running code formatting..."
    docker-compose run --rm frontend-dev npm run format
    docker-compose run --rm backend-dev npm run format

    print_status "Re-running all tests to ensure functionality is preserved..."
    run_all_tests

    print_status "Running performance tests..."
    docker-compose run --rm frontend-test npm run test:performance

    print_success "REFACTOR phase completed - Code improved while maintaining functionality!"
    print_warning "Ready to start next TDD cycle with './scripts/tdd.sh red'"
}

run_tdd_cycle() {
    print_status "Starting complete TDD cycle..."

    run_red_phase
    echo ""
    read -p "Press Enter to continue to GREEN phase (make tests pass)..."

    run_green_phase
    echo ""
    read -p "Press Enter to continue to REFACTOR phase (improve code)..."

    run_refactor_phase

    print_success "Complete TDD cycle finished!"
}

# Testing Functions
run_all_tests() {
    print_status "Running complete test suite..."

    # Frontend tests
    docker-compose run --rm frontend-test npm run test:coverage

    # Backend tests
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm backend-test

    # Integration tests
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm integration-tests

    # E2E tests
    docker-compose --profile development up -d
    sleep 30
    docker-compose --profile testing run --rm e2e-tests

    print_success "All tests completed!"
}

run_unit_tests() {
    print_status "Running unit tests..."
    docker-compose run --rm frontend-test npm run test:unit:coverage
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm backend-test npm run test:unit
    print_success "Unit tests completed!"
}

run_integration_tests() {
    print_status "Running integration tests..."
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm integration-tests
    print_success "Integration tests completed!"
}

run_e2e_tests() {
    print_status "Running E2E tests..."
    docker-compose --profile development up -d
    sleep 30
    docker-compose --profile testing run --rm e2e-tests
    print_success "E2E tests completed!"
}

run_coverage_tests() {
    print_status "Generating coverage reports..."
    docker-compose run --rm frontend-test npm run test:coverage
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm backend-test npm run test:coverage
    print_success "Coverage reports generated in ./frontend/coverage and ./backend/coverage"
}

run_performance_tests() {
    print_status "Running performance tests..."
    docker-compose run --rm frontend-test npm run test:performance
    print_success "Performance tests completed!"
}

run_a11y_tests() {
    print_status "Running accessibility tests..."
    docker-compose run --rm frontend-test npm run test:a11y
    print_success "Accessibility tests completed!"
}

# Development Functions
setup_tdd_environment() {
    print_status "Setting up TDD environment..."

    # Run the main setup
    ./scripts/dev.sh setup

    # Create test data directories
    mkdir -p cypress/screenshots cypress/videos cypress/fixtures
    mkdir -p frontend/coverage backend/coverage

    # Install testing dependencies
    print_status "Installing testing dependencies..."
    docker-compose run --rm frontend-dev npm install

    print_success "TDD environment setup completed!"
}

start_dev_environment() {
    print_status "Starting development environment for TDD..."
    docker-compose --profile development up -d
    print_success "Development environment started!"
    print_status "Frontend: http://localhost:3000"
    print_status "Backend: http://localhost:8000"
}

start_test_watcher() {
    print_status "Starting test watcher mode..."
    docker-compose run --rm frontend-test npm run test:watch
}

# Validation Functions
validate_tdd_compliance() {
    print_status "Validating TDD compliance..."

    # Check test coverage
    local frontend_coverage=$(docker-compose run --rm frontend-test npm run test:coverage:check 2>/dev/null || echo "FAIL")
    local backend_coverage=$(docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm backend-test npm run test:coverage:check 2>/dev/null || echo "FAIL")

    # Check test organization
    local test_files=$(find frontend/src -name "*.test.*" -o -name "*.spec.*" | wc -l)
    local e2e_files=$(find cypress/e2e -name "*.cy.*" | wc -l)

    echo "TDD Compliance Report:"
    echo "  Frontend Coverage: ${frontend_coverage}"
    echo "  Backend Coverage: ${backend_coverage}"
    echo "  Unit Test Files: ${test_files}"
    echo "  E2E Test Files: ${e2e_files}"

    if [[ "$frontend_coverage" != "FAIL" ]] && [[ "$backend_coverage" != "FAIL" ]] && [[ $test_files -gt 0 ]] && [[ $e2e_files -gt 0 ]]; then
        print_success "TDD compliance validated!"
    else
        print_warning "TDD compliance issues detected. Check test coverage and organization."
    fi
}

show_tdd_metrics() {
    print_status "Gathering TDD metrics..."

    # Test counts
    local unit_tests=$(grep -r "describe\|it\|test" frontend/src --include="*.test.*" --include="*.spec.*" | wc -l)
    local e2e_tests=$(grep -r "describe\|it" cypress/e2e --include="*.cy.*" | wc -l)

    # Coverage info
    if [ -f "frontend/coverage/coverage-summary.json" ]; then
        local coverage=$(cat frontend/coverage/coverage-summary.json | grep -o '"pct":[0-9]*' | head -1 | cut -d':' -f2)
        echo "Test Coverage: ${coverage}%"
    fi

    echo "TDD Metrics:"
    echo "  Unit Tests: ${unit_tests}"
    echo "  E2E Tests: ${e2e_tests}"
    echo "  Test Files: $(find . -name "*.test.*" -o -name "*.spec.*" -o -name "*.cy.*" | wc -l)"
}

check_health() {
    print_status "Checking TDD environment health..."

    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running"
        return 1
    fi

    # Check if services are healthy
    local services_up=$(docker-compose ps --services --filter "status=running" | wc -l)
    echo "Services running: ${services_up}"

    # Check test dependencies
    if docker-compose run --rm frontend-dev npm list --depth=0 | grep -q "jest\|cypress\|@testing-library"; then
        print_success "Testing dependencies are installed"
    else
        print_warning "Some testing dependencies may be missing"
    fi
}

# Cleanup Functions
clean_test_artifacts() {
    print_status "Cleaning test artifacts..."

    rm -rf frontend/coverage backend/coverage
    rm -rf cypress/screenshots cypress/videos
    rm -rf node_modules/.cache

    docker-compose down -v
    docker system prune -f

    print_success "Test artifacts cleaned!"
}

reset_to_clean_state() {
    print_warning "This will reset the entire testing environment. Continue? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        clean_test_artifacts
        setup_tdd_environment
        print_success "Environment reset to clean state!"
    fi
}

# Main command router
case "$1" in
    # TDD Workflow
    red)
        run_red_phase
        ;;
    green)
        run_green_phase
        ;;
    refactor)
        run_refactor_phase
        ;;
    cycle)
        run_tdd_cycle
        ;;

    # Testing Commands
    test:all)
        run_all_tests
        ;;
    test:unit)
        run_unit_tests
        ;;
    test:integration)
        run_integration_tests
        ;;
    test:e2e)
        run_e2e_tests
        ;;
    test:coverage)
        run_coverage_tests
        ;;
    test:performance)
        run_performance_tests
        ;;
    test:a11y)
        run_a11y_tests
        ;;

    # Development Commands
    setup)
        setup_tdd_environment
        ;;
    dev)
        start_dev_environment
        ;;
    watch)
        start_test_watcher
        ;;
    lint)
        ./scripts/dev.sh lint
        ;;
    format)
        ./scripts/dev.sh format
        ;;

    # Validation Commands
    validate)
        validate_tdd_compliance
        ;;
    metrics)
        show_tdd_metrics
        ;;
    health)
        check_health
        ;;

    # Cleanup Commands
    clean)
        clean_test_artifacts
        ;;
    reset)
        reset_to_clean_state
        ;;

    help|--help|-h)
        show_tdd_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_tdd_help
        exit 1
        ;;
esac
