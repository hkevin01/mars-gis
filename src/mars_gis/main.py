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


def create_app():
    """Create and configure the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI not available. Please install requirements.txt"
        )
    
    app = FastAPI(
        title="MARS-GIS API",
        description="Mars Geospatial Intelligence System API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Basic health check endpoint
    @app.get("/")
    async def health_check():
        return {
            "message": "MARS-GIS API is running",
            "version": "0.1.0",
            "status": "healthy"
        }

    return app


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
