#!/bin/bash
# Quick cleanup of backup files

echo "Removing Docker backup files..."

# Remove backup files
rm -f Dockerfile-backup
rm -f docker-compose-corrupted-backup.yml
rm -f docker-compose.override-backup.yml
rm -f docker-compose.prod-backup.yml
rm -f docker-compose.test-backup.yml

echo "Backup files removed!"

# List remaining Docker files
echo "Remaining Docker files in root:"
ls -la | grep -E "(Dockerfile|docker-compose)"
