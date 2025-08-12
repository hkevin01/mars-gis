"""Core configuration module for MARS-GIS."""

import os
from pathlib import Path
from typing import Optional


class Settings:
    """Application settings configuration."""

    def __init__(self) -> None:
        """Initialize settings with default values."""
        # Application
        self.APP_NAME = "MARS-GIS"
        self.VERSION = "0.1.0"
        self.DESCRIPTION = "Mars Geospatial Intelligence System"
        self.ENVIRONMENT = "development"
        self.DEBUG = True

        # Server
        self.HOST = "0.0.0.0"
        self.PORT = 8000
        self.WORKERS = 1
        self.RELOAD = True

        # CORS
        self.ALLOWED_HOSTS = [
            "http://localhost:3000",
            "http://localhost:8000"
        ]

        # Database
        self.DATABASE_URL = (
            "postgresql://postgres:password@localhost:5432/mars_gis"
        )
        self.DATABASE_ECHO = False

        # Redis
        self.REDIS_URL = "redis://localhost:6379/0"

        # NASA APIs
        self.NASA_API_KEY: Optional[str] = None
        self.NASA_PDS_BASE_URL = "https://pds-imaging.jpl.nasa.gov/data"

        # USGS APIs
        self.USGS_BASE_URL = "https://astrogeology.usgs.gov/search"

        # Mars tiles proxy (imagery)
        # Default to OpenPlanetaryMap MOLA colorized tiles (TMS scheme)
        default_mola = (
            "http://s3-eu-west-1.amazonaws.com/whereonmars.cartodb.net/"
            "mola_color-noshade_global/{z}/{x}/{y}.png"
        )
        self.MARS_TILES_TEMPLATE: str = os.environ.get(
            "MARS_TILES_TEMPLATE",
            default_mola,
        )
        self.MARS_TILES_IS_TMS: bool = (
            os.environ.get("MARS_TILES_IS_TMS", "true").lower()
            in {"1", "true", "yes"}
        )

        # ML/AI Settings
        self.TORCH_DEVICE = "cuda"
        self.MODEL_CACHE_DIR = "data/models"

        # File paths
        self.DATA_DIR = Path("data")
        self.LOGS_DIR = Path("logs")
        self.ASSETS_DIR = Path("assets")

        # Security
        self.SECRET_KEY = "mars-gis-secret-key-change-in-production"
        self.ALGORITHM = "HS256"
        self.ACCESS_TOKEN_EXPIRE_MINUTES = 30

        # Logging
        self.LOG_LEVEL = "INFO"
        self.LOG_FORMAT = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create directories
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.DATA_DIR / "raw",
            self.DATA_DIR / "processed",
            self.DATA_DIR / "models",
            self.LOGS_DIR,
            self.ASSETS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
