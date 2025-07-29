"""Test configuration and fixtures for MARS-GIS testing."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from mars_gis.core.config import settings

# Test database URL (in-memory SQLite for fast tests)
TEST_DATABASE_URL = "sqlite:///:memory:"

# Mock Mars coordinates for testing
MOCK_MARS_COORDINATES = [
    {"lat": -14.5684, "lon": 175.4729, "name": "Olympia Undae"},
    {"lat": -5.4453, "lon": 137.8414, "name": "Gale Crater"},  
    {"lat": 22.5, "lon": -49.97, "name": "Valles Marineris"},
    {"lat": 18.65, "lon": 226.2, "name": "Olympus Mons"},
    {"lat": -83.0, "lon": 160.0, "name": "South Polar Cap"},
]


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


@pytest.fixture(scope="session")
def test_database():
    """Create test database for the session."""
    engine = create_engine(
        TEST_DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False
    )
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(test_database):
    """Create database session for tests."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, 
        autoflush=False, 
        bind=test_database
    )
    
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def temp_directory():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_mars_image_data():
    """Generate mock Mars image data for testing."""
    try:
        import numpy as np

        # Create mock 224x224x3 Mars surface image
        image_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return image_data
    except ImportError:
        return None


@pytest.fixture
def mock_mission_data():
    """Generate mock mission data for testing."""
    return {
        "id": "test-mission-001",
        "name": "Test Mars Exploration Mission",
        "description": "A test mission for unit testing",
        "status": "planned",
        "asset_id": "test-rover-001",
        "start_date": "2024-02-01T08:00:00Z",
        "tasks": [
            {
                "id": "task-001",
                "name": "Navigate to target",
                "type": "navigation",
                "coordinates": [-14.5684, 175.4729],
                "status": "pending"
            }
        ]
    }


@pytest.fixture
def mock_terrain_model():
    """Create mock terrain classification model for testing."""
    try:
        import torch
        import torch.nn as nn
        
        class MockTerrainModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.fc = nn.Linear(64, 8)  # 8 terrain classes
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        return MockTerrainModel()
    except ImportError:
        return None


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "ml: mark test as a machine learning test")
    config.addinivalue_line("markers", "geospatial: mark test as a geospatial test")
    config.addinivalue_line("markers", "api: mark test as an API test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def requires_torch(func):
    """Decorator to skip tests that require PyTorch."""
    return pytest.mark.skipif(
        not torch_available(),
        reason="PyTorch not available"
    )(func)


def torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False
