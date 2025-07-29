"""Basic test for MARS-GIS core functionality."""

import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from mars_gis.core.config import Settings


def test_settings_initialization():
    """Test that settings can be initialized."""
    settings = Settings()
    assert settings.APP_NAME == "MARS-GIS"
    assert settings.VERSION == "0.1.0"
    assert settings.PORT == 8000


def test_directory_creation():
    """Test that necessary directories are created."""
    settings = Settings()
    assert settings.DATA_DIR.exists()
    assert (settings.DATA_DIR / "raw").exists()
    assert (settings.DATA_DIR / "processed").exists()
    assert settings.LOGS_DIR.exists()


def test_main_module_import():
    """Test that main module can be imported."""
    try:
        from mars_gis import main
        assert main is not None
    except ImportError as e:
        # Expected if FastAPI not installed
        assert "FastAPI not available" in str(e) or "FastAPI" in str(e)
