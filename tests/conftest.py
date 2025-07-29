"""Test configuration and fixtures."""

from pathlib import Path

import pytest

from mars_gis.core.config import settings


@pytest.fixture
def test_data_dir():
    """Fixture for test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_mars_image():
    """Fixture for sample Mars image data."""
    # This would return path to sample Mars image for testing
    return test_data_dir() / "sample_mars.tif"


@pytest.fixture
def test_settings():
    """Fixture for test settings configuration."""
    test_settings = settings
    test_settings.ENVIRONMENT = "testing"
    test_settings.DEBUG = True
    return test_settings
