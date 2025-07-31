#!/bin/bash

# Docker Organization Validation Script
echo "🔍 Validating MARS-GIS Docker Organization..."
echo

# Check if organized structure exists
echo "📁 Checking organized structure..."

success=0
total=0

check_file() {
    local file="$1"
    local description="$2"
    ((total++))
    if [[ -f "$file" ]]; then
        echo "✅ $description: $file"
        ((success++))
    else
        echo "❌ Missing: $description ($file)"
    fi
}

check_dir() {
    local dir="$1"
    local description="$2"
    ((total++))
    if [[ -d "$dir" ]]; then
        echo "✅ $description: $dir"
        ((success++))
    else
        echo "❌ Missing: $description ($dir)"
    fi
}

# Check directories
check_dir "docker" "Docker root directory"
check_dir "docker/backend" "Backend directory"
check_dir "docker/frontend" "Frontend directory"
check_dir "docker/compose" "Compose directory"

# Check organized files
check_file "docker/backend/Dockerfile" "Backend Dockerfile"
check_file "docker/frontend/Dockerfile" "Frontend Dockerfile"
check_file "docker/compose/docker-compose.yml" "Main compose file"
check_file "docker/compose/docker-compose.override.yml" "Override compose file"
check_file "docker/compose/docker-compose.prod.yml" "Production compose file"
check_file "docker/compose/docker-compose.test.yml" "Test compose file"
check_file "docker/docker-helper.sh" "Helper script"
check_file "docker/README.md" "Docker documentation"

# Check root redirect files
echo
echo "🔗 Checking redirect files..."
check_file "Dockerfile" "Root Dockerfile redirect"
check_file "docker-compose.yml" "Root compose redirect"
check_file "docker-compose.override.yml" "Root override redirect"
check_file "docker-compose.prod.yml" "Root prod redirect"
check_file "docker-compose.test.yml" "Root test redirect"

# Check documentation
echo
echo "📚 Checking documentation..."
check_file "DOCKER_MIGRATION_SUMMARY.md" "Migration summary"
check_file "docs/project_plan_gui.md" "Project plan GUI"

echo
echo "🧪 Testing helper script..."
if [[ -x "docker/docker-helper.sh" ]]; then
    echo "✅ Helper script is executable"
    ((success++))
else
    echo "❌ Helper script is not executable"
fi
((total++))

echo
echo "📊 Validation Results:"
echo "✅ Successful checks: $success/$total"

if [[ $success -eq $total ]]; then
    echo "🎉 All validations passed! Docker organization is complete."
    echo
    echo "🚀 Ready to use:"
    echo "   ./docker/docker-helper.sh help"
    echo "   ./docker/docker-helper.sh dev"
    exit 0
else
    echo "⚠️  Some validations failed. Please check the missing items above."
    exit 1
fi
