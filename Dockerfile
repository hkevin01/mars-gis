# MARS-GIS Project Dockerfile
#
# This project uses organized Docker structure in docker/ directory:
#
# For backend:  docker/backend/Dockerfile
# For frontend: docker/frontend/Dockerfile
#
# Use the helper script for easy management:
# ./docker/docker-helper.sh dev|prod|test|build
#
# Or navigate to docker/compose/ for direct docker-compose usage

FROM alpine:latest

LABEL maintainer="Kevin Hildebrand <kevin.hildebrand@gmail.com>"
LABEL description="MARS-GIS: Mars Geospatial Intelligence System"
LABEL version="1.0.0"

# Create informational script
RUN echo '#!/bin/sh' > /usr/local/bin/mars-gis-info && \
    echo 'echo "🚀 MARS-GIS: Mars Geospatial Intelligence System"' >> /usr/local/bin/mars-gis-info && \
    echo 'echo ""' >> /usr/local/bin/mars-gis-info && \
    echo 'echo "Docker files are organized in docker/ directory:"' >> /usr/local/bin/mars-gis-info && \
    echo 'echo "  - Backend:  docker/backend/Dockerfile"' >> /usr/local/bin/mars-gis-info && \
    echo 'echo "  - Frontend: docker/frontend/Dockerfile"' >> /usr/local/bin/mars-gis-info && \
    echo 'echo ""' >> /usr/local/bin/mars-gis-info && \
    echo 'echo "Quick start:"' >> /usr/local/bin/mars-gis-info && \
    echo 'echo "  ./docker/docker-helper.sh dev"' >> /usr/local/bin/mars-gis-info && \
    echo 'echo ""' >> /usr/local/bin/mars-gis-info && \
    chmod +x /usr/local/bin/mars-gis-info

CMD ["mars-gis-info"]
