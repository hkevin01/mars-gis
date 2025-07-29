# Multi-stage Dockerfile for Mars GIS Platform
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libspatialite-dev \
    libsqlite3-mod-spatialite \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 mars && \
    useradd -r -u 1000 -g mars mars

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    isort \
    flake8 \
    mypy \
    jupyter

# Copy source code
COPY --chown=mars:mars . .

# Switch to non-root user
USER mars

# Expose ports
EXPOSE 8000 8888

# Default command for development
CMD ["uvicorn", "mars_gis.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Testing stage
FROM development as testing

# Switch back to root for test setup
USER root

# Create test directories
RUN mkdir -p /app/test-results /app/htmlcov
RUN chown -R mars:mars /app/test-results /app/htmlcov

# Switch to non-root user
USER mars

# Run tests
CMD ["python", "run_tests.py", "--all", "--coverage"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=mars:mars src/ ./src/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models && \
    chown -R mars:mars /app/logs /app/data /app/models

# Install gunicorn for production
RUN pip install --no-cache-dir gunicorn

# Switch to non-root user
USER mars

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["gunicorn", "mars_gis.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
