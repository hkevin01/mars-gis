"""Main application entry point for MARS-GIS."""

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    CORSMiddleware = None
    uvicorn = None

from mars_gis.core.config import settings

# Import API router
try:
    from mars_gis.api.routes import create_api_router
    API_ROUTES_AVAILABLE = True
except ImportError:
    API_ROUTES_AVAILABLE = False
    create_api_router = None


def create_app():
    """Create and configure the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI not available. Please install requirements.txt"
        )

    app = FastAPI(
        title="MARS-GIS API",
        description="Mars Geospatial Intelligence System API - "
                   "Implements ISO/IEC 29148:2011 Requirements",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {
                "name": "mars-gis-api",
                "description": "MARS-GIS Core API Endpoints",
            },
            {
                "name": "health",
                "description": "System health and monitoring endpoints",
            },
        ]
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes if available
    if API_ROUTES_AVAILABLE and create_api_router:
        api_router = create_api_router()
        if api_router:
            app.include_router(api_router)

    # Root health check endpoint
    @app.get("/", tags=["health"])
    async def root_health_check():
        """Root endpoint with system status."""
        return {
            "message": "MARS-GIS API is running",
            "version": "1.0.0",
            "status": "healthy",
            "api_documentation": "/docs",
            "requirements_compliance": "ISO/IEC 29148:2011",
            "endpoints": {
                "mars_data": "/api/v1/mars-data/*",
                "ml_inference": "/api/v1/inference/*",
                "missions": "/api/v1/missions/*",
                "streams": "/api/v1/streams/*",
                "system": "/api/v1/system/*"
            }
        }

    # Legacy health endpoint for backward compatibility
    @app.get("/health", tags=["health"])
    async def legacy_health_check():
        """Legacy health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": "2025-08-01T00:00:00Z",
            "version": "1.0.0"
        }

    return app


# Application instance for production deployment
app = create_app()


def run_development_server():
    """Run the development server."""
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available. Please install requirements:")
        print("pip install -r requirements.txt")
        return

    print("🚀 Starting MARS-GIS Development Server")
    print("📋 API Documentation: http://localhost:8000/docs")
    print("🔄 API Alternative Docs: http://localhost:8000/redoc")
    print("⚡ Server will auto-reload on code changes")

    uvicorn.run(
        "mars_gis.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


def main():
    """Main entry point."""
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Please install requirements.txt")
        return

    app = create_app()

    if uvicorn:
        uvicorn.run(
            app,
            host=settings.HOST,
            port=settings.PORT,
            reload=(
                settings.RELOAD
                if settings.ENVIRONMENT == "development"
                else False
            ),
        )


if __name__ == "__main__":
    main()
