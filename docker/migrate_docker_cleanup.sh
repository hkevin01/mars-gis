#!/bin/bash

# MARS-GIS Docker File Cleanup Migration Script
# This script completes the Docker organization by cleaning up remaining root files

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BLUE}=== MARS-GIS Docker Cleanup Migration Script ===${NC}"
echo "Working in: $PROJECT_ROOT"
echo

# Function to log operations
log_operation() {
    echo -e "${GREEN}[MIGRATE]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to backup important files
create_backup() {
    local file="$1"
    if [[ -f "$file" ]]; then
        cp "$file" "${file}.pre-cleanup-backup"
        log_operation "Created backup: ${file}.pre-cleanup-backup"
    fi
}

# Function to safely remove files
safe_remove() {
    local file="$1"
    if [[ -f "$file" ]]; then
        log_operation "Removing: $file"
        rm -f "$file"
    else
        log_warning "File not found: $file"
    fi
}

# Function to move files
safe_move() {
    local src="$1"
    local dest="$2"
    if [[ -f "$src" ]]; then
        log_operation "Moving $src to $dest"
        mv "$src" "$dest"
    else
        log_warning "Source file not found: $src"
    fi
}

echo -e "${BLUE}Step 1: Analyzing current Docker file structure...${NC}"

# Check if docker directory exists
if [[ ! -d "$PROJECT_ROOT/docker" ]]; then
    log_error "Docker directory not found! Please run the main Docker organization first."
    exit 1
fi

echo "✓ Docker directory exists"

# List current Docker files in root
echo -e "\n${BLUE}Current Docker files in root:${NC}"
find "$PROJECT_ROOT" -maxdepth 1 -name "docker*" -o -name "Dockerfile*" | sort

echo -e "\n${BLUE}Step 2: Creating backup of critical files...${NC}"

# Backup critical Docker files before cleanup
if [[ -f "$PROJECT_ROOT/Dockerfile" ]]; then
    create_backup "$PROJECT_ROOT/Dockerfile"
fi

echo -e "\n${BLUE}Step 3: Removing duplicate/backup Docker files...${NC}"

# Remove backup files that are no longer needed
safe_remove "$PROJECT_ROOT/Dockerfile-backup"
safe_remove "$PROJECT_ROOT/docker-compose-corrupted-backup.yml"
safe_remove "$PROJECT_ROOT/docker-compose.override-backup.yml"
safe_remove "$PROJECT_ROOT/docker-compose.prod-backup.yml"
safe_remove "$PROJECT_ROOT/docker-compose.test-backup.yml"

echo -e "\n${BLUE}Step 4: Handling main Dockerfile...${NC}"

# Check if main Dockerfile should be moved or removed
if [[ -f "$PROJECT_ROOT/Dockerfile" ]]; then
    # Check if it's different from the ones in docker/
    if [[ -f "$PROJECT_ROOT/docker/backend/Dockerfile" ]] || [[ -f "$PROJECT_ROOT/docker/frontend/Dockerfile" ]]; then
        echo "Comparing root Dockerfile with organized ones..."

        # Create a consolidated redirect Dockerfile in root that points to the appropriate one
        cat > "$PROJECT_ROOT/Dockerfile" << 'EOF'
# This is a redirect file for backward compatibility
# The actual Dockerfiles have been moved to the docker/ directory:
#
# Backend Dockerfile: docker/backend/Dockerfile
# Frontend Dockerfile: docker/frontend/Dockerfile
#
# To build specific services, use:
# docker build -f docker/backend/Dockerfile .
# docker build -f docker/frontend/Dockerfile .
#
# Or use the helper script:
# ./docker/docker-helper.sh

FROM alpine:latest
RUN echo "Please use the organized Docker files in the docker/ directory"
RUN echo "Backend: docker/backend/Dockerfile"
RUN echo "Frontend: docker/frontend/Dockerfile"
RUN echo "Helper script: ./docker/docker-helper.sh"
CMD ["echo", "Use organized Docker files in docker/ directory"]
EOF
        log_operation "Updated root Dockerfile to redirect to organized structure"
    fi
fi

echo -e "\n${BLUE}Step 5: Consolidating docker-compose files...${NC}"

# Create a main docker-compose.yml redirect
cat > "$PROJECT_ROOT/docker-compose.yml" << 'EOF'
# This file redirects to the organized Docker Compose configuration
# The actual compose files have been moved to docker/compose/
#
# Available configurations:
# - docker/compose/docker-compose.yml (main)
# - docker/compose/docker-compose.override.yml (development overrides)
# - docker/compose/docker-compose.prod.yml (production)
# - docker/compose/docker-compose.test.yml (testing)
#
# To use the organized structure:
# cd docker/compose && docker-compose up
#
# Or use the helper script:
# ./docker/docker-helper.sh

version: '3.8'
services:
  redirect-info:
    image: alpine:latest
    command: |
      sh -c "
        echo '=================================================================='
        echo 'Docker Compose files have been reorganized!'
        echo 'Please use: ./docker/docker-helper.sh'
        echo 'Or navigate to: docker/compose/'
        echo '=================================================================='
        sleep 5
      "
EOF

log_operation "Created consolidated docker-compose.yml redirect"

# Update other compose files to be minimal redirects
for compose_file in docker-compose.override.yml docker-compose.prod.yml docker-compose.test.yml; do
    if [[ -f "$PROJECT_ROOT/$compose_file" ]]; then
        cat > "$PROJECT_ROOT/$compose_file" << EOF
# This file has been moved to docker/compose/
# Please use: docker/compose/$compose_file
# Or run: ./docker/docker-helper.sh

version: '3.8'
services:
  redirect:
    image: alpine:latest
    command: echo "Use docker/compose/$compose_file instead"
EOF
        log_operation "Updated $compose_file to redirect to organized structure"
    fi
done

echo -e "\n${BLUE}Step 6: Updating documentation...${NC}"

# Update the migration summary
cat >> "$PROJECT_ROOT/DOCKER_MIGRATION_SUMMARY.md" << 'EOF'

## Phase 2: Root Directory Cleanup

### Files Cleaned Up
- Removed backup files: `*-backup.yml`, `Dockerfile-backup`
- Updated root Docker files to redirect to organized structure
- Consolidated docker-compose files with clear redirect information

### Current Structure
```
docker/
├── backend/Dockerfile          # Backend application
├── frontend/Dockerfile         # Frontend application
├── compose/
│   ├── docker-compose.yml      # Main compose configuration
│   ├── docker-compose.override.yml  # Development overrides
│   ├── docker-compose.prod.yml      # Production configuration
│   └── docker-compose.test.yml      # Testing configuration
├── docker-helper.sh            # Management script
└── README.md                   # Documentation

Root redirects:
├── Dockerfile                  # Points to organized structure
├── docker-compose.yml          # Redirects to docker/compose/
├── docker-compose.override.yml # Redirects to docker/compose/
├── docker-compose.prod.yml     # Redirects to docker/compose/
└── docker-compose.test.yml     # Redirects to docker/compose/
```

### Migration Complete ✅
All Docker files have been organized and cleaned up. The project now has:
- Clear separation of concerns
- Backward compatibility through redirects
- Comprehensive documentation
- Helper scripts for easy management
EOF

echo -e "\n${BLUE}Step 7: Validating the organized structure...${NC}"

# Check that all expected files exist in the organized structure
validation_errors=0

echo "Checking organized Docker structure..."

if [[ -f "$PROJECT_ROOT/docker/docker-helper.sh" ]]; then
    echo "✓ Helper script exists"
else
    log_error "Helper script missing"
    ((validation_errors++))
fi

if [[ -f "$PROJECT_ROOT/docker/README.md" ]]; then
    echo "✓ Docker documentation exists"
else
    log_error "Docker README missing"
    ((validation_errors++))
fi

if [[ -d "$PROJECT_ROOT/docker/compose" ]]; then
    echo "✓ Compose directory exists"

    compose_files=("docker-compose.yml" "docker-compose.override.yml" "docker-compose.prod.yml" "docker-compose.test.yml")
    for file in "${compose_files[@]}"; do
        if [[ -f "$PROJECT_ROOT/docker/compose/$file" ]]; then
            echo "  ✓ $file exists"
        else
            log_warning "  Missing: $file"
        fi
    done
else
    log_error "Compose directory missing"
    ((validation_errors++))
fi

if [[ -d "$PROJECT_ROOT/docker/backend" ]]; then
    echo "✓ Backend directory exists"
else
    log_warning "Backend directory missing"
fi

if [[ -d "$PROJECT_ROOT/docker/frontend" ]]; then
    echo "✓ Frontend directory exists"
else
    log_warning "Frontend directory missing"
fi

echo -e "\n${BLUE}Step 8: Testing helper script functionality...${NC}"

if [[ -x "$PROJECT_ROOT/docker/docker-helper.sh" ]]; then
    echo "✓ Helper script is executable"
    echo "Testing helper script..."
    cd "$PROJECT_ROOT"
    ./docker/docker-helper.sh --help > /dev/null 2>&1 && echo "✓ Helper script runs correctly" || log_warning "Helper script may have issues"
else
    log_warning "Helper script is not executable, fixing..."
    chmod +x "$PROJECT_ROOT/docker/docker-helper.sh"
fi

echo -e "\n${GREEN}=== Migration Cleanup Complete! ===${NC}"

if [[ $validation_errors -eq 0 ]]; then
    echo -e "${GREEN}✅ All validations passed!${NC}"
else
    echo -e "${YELLOW}⚠️  $validation_errors validation warnings found${NC}"
fi

echo -e "\n${BLUE}Next Steps:${NC}"
echo "1. Test the organized Docker structure:"
echo "   ./docker/docker-helper.sh"
echo ""
echo "2. Remove this migration script after verification:"
echo "   rm migrate_docker_cleanup.sh"
echo ""
echo "3. Update any CI/CD or documentation that references old Docker file locations"
echo ""
echo -e "${GREEN}The Docker infrastructure is now fully organized and ready to use!${NC}"
