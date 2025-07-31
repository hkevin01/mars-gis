#!/bin/bash

# MARS-GIS Docker File Cleanup Script
# This script cleans up the corrupted root Docker files and provides redirect files

echo "🧹 Starting Docker file cleanup..."

# Backup corrupted files
echo "📦 Backing up corrupted files..."
mv docker-compose.yml docker-compose-corrupted-backup.yml 2>/dev/null || true
mv docker-compose.prod.yml docker-compose.prod-backup.yml 2>/dev/null || true
mv docker-compose.test.yml docker-compose.test-backup.yml 2>/dev/null || true
mv docker-compose.override.yml docker-compose.override-backup.yml 2>/dev/null || true
mv Dockerfile Dockerfile-backup 2>/dev/null || true

# Install clean redirect files
echo "📁 Installing redirect files..."
mv docker-compose-final.yml docker-compose.yml
mv docker-compose.prod-final.yml docker-compose.prod.yml
mv docker-compose.test-final.yml docker-compose.test.yml
mv docker-compose.override-final.yml docker-compose.override.yml
mv Dockerfile-final Dockerfile

# Clean up deprecated files
echo "🗑️  Removing deprecated files..."
rm -f docker-compose-deprecated.yml 2>/dev/null || true
rm -f docker-compose-new.yml 2>/dev/null || true
rm -f docker-compose-prod-deprecated.yml 2>/dev/null || true
rm -f docker-compose-clean.yml 2>/dev/null || true
rm -f docker-compose.prod-redirect.yml 2>/dev/null || true
rm -f docker-compose.test-redirect.yml 2>/dev/null || true
rm -f docker-compose.override-redirect.yml 2>/dev/null || true
rm -f Dockerfile-redirect 2>/dev/null || true

# Make helper script executable
chmod +x docker/docker-helper.sh 2>/dev/null || true

echo "✅ Docker file cleanup completed!"
echo ""
echo "📁 Files moved to docker/ directory:"
echo "   ✅ docker/backend/Dockerfile"
echo "   ✅ docker/frontend/Dockerfile"
echo "   ✅ docker/compose/docker-compose.yml"
echo "   ✅ docker/compose/docker-compose.prod.yml"
echo "   ✅ docker/compose/docker-compose.test.yml"
echo "   ✅ docker/compose/docker-compose.override.yml"
echo "   ✅ docker/docker-helper.sh"
echo "   ✅ docker/README.md"
echo ""
echo "🚀 Use these commands:"
echo "   ./docker/docker-helper.sh dev    # Development environment"
echo "   ./docker/docker-helper.sh prod   # Production environment"
echo "   ./docker/docker-helper.sh test   # Run tests"
echo "   ./docker/docker-helper.sh help   # See all commands"
echo ""
echo "📚 See docker/README.md for complete documentation"
echo ""
echo "🎉 Docker organization complete! Root files now contain redirect notices."

# Clean up this script
rm -f cleanup-docker.sh
