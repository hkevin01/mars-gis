#!/bin/bash

# Simple runner to execute the Docker migration cleanup
echo "Making migration script executable..."
chmod +x migrate_docker_cleanup.sh

echo "Running Docker cleanup migration..."
./migrate_docker_cleanup.sh

echo "Migration complete!"
