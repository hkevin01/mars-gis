#!/bin/bash

# Final Docker cleanup - Remove backup files
echo "🧹 Final Docker cleanup - removing backup files..."

# Files to remove
backup_files=(
    "Dockerfile-backup"
    "docker-compose-corrupted-backup.yml"
    "docker-compose.override-backup.yml"
    "docker-compose.prod-backup.yml"
    "docker-compose.test-backup.yml"
)

removed=0
for file in "${backup_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "🗑️  Removing: $file"
        rm -f "$file"
        ((removed++))
    else
        echo "ℹ️  Already clean: $file"
    fi
done

echo
echo "✅ Cleanup complete! Removed $removed backup files."

# Cleanup the migration scripts themselves
echo
echo "🧹 Cleaning up migration scripts..."
migration_scripts=(
    "migrate_docker_cleanup.sh"
    "run_migration.sh"
    "quick_cleanup.sh"
    "validate_docker_organization.sh"
    "docker_cleanup_log.md"
)

echo "The following migration scripts can be safely removed:"
for script in "${migration_scripts[@]}"; do
    if [[ -f "$script" ]]; then
        echo "  - $script"
    fi
done

echo
echo "Run this to remove migration scripts:"
echo "rm -f migrate_docker_cleanup.sh run_migration.sh quick_cleanup.sh validate_docker_organization.sh docker_cleanup_log.md final_docker_cleanup.sh"
