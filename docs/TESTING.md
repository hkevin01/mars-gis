# MARS-GIS Testing Guide

**Version:** 1.0.0
**Date:** August 1, 2025
**Test Coverage:** 95%+ Required

## Table of Contents

1. [Testing Strategy](#testing-strategy)
2. [Test Environment Setup](#test-environment-setup)
3. [Unit Testing](#unit-testing)
4. [Integration Testing](#integration-testing)
5. [End-to-End Testing](#end-to-end-testing)
6. [Performance Testing](#performance-testing)
7. [Security Testing](#security-testing)
8. [Test Automation](#test-automation)
9. [Continuous Testing](#continuous-testing)
10. [Test Reporting](#test-reporting)

## Testing Strategy

### Test Pyramid Philosophy

The MARS-GIS testing strategy follows the test pyramid approach with emphasis on fast, reliable, and maintainable tests:

```
                    ╔══════════════════╗
                   ╔╝    E2E Tests     ╚╗  ← 10% (UI workflows, system integration)
                  ╔╝     (Slow)        ╚╗
                 ╔╝                     ╚╗
                ╔╝   Integration Tests   ╚╗ ← 20% (API, database, external services)
               ╔╝      (Medium)          ╚╗
              ╔╝                          ╚╗
             ╔╝        Unit Tests          ╚╗ ← 70% (Functions, classes, components)
            ╔╝         (Fast)              ╚╗
           ╚═══════════════════════════════╝
```

### Testing Principles

1. **Test-Driven Development (TDD)**: Write tests before implementation
2. **Fast Feedback**: Most tests run in under 1 second
3. **Isolation**: Tests don't depend on external systems
4. **Deterministic**: Tests produce consistent results
5. **Maintainable**: Tests are easy to understand and modify

### Test Coverage Goals

- **Unit Tests**: 95% code coverage
- **Integration Tests**: 90% critical path coverage
- **End-to-End Tests**: 100% user story coverage
- **Performance Tests**: All critical workflows under 2s load time
- **Security Tests**: 100% endpoint coverage

## Test Environment Setup

### Prerequisites

```bash
# Install testing dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock
pip install httpx  # For FastAPI testing
pip install factory-boy  # For test data generation
pip install testcontainers  # For integration testing
```

### Test Configuration

**pytest.ini**:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --strict-config
    --verbose
    --cov=mars_gis
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=95
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    security: Security tests
    performance: Performance tests
asyncio_mode = auto
```

### Test Database Setup

**conftest.py**:
```python
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from testcontainers.postgres import PostgresContainer
from mars_gis.core.database import Base, get_db_session

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def postgres_container():
    """Start PostgreSQL container for testing."""
    with PostgresContainer("postgis/postgis:13-3.1") as postgres:
        postgres.with_env("POSTGRES_DB", "test_mars_gis")
        postgres.with_env("POSTGRES_USER", "test_user")
        postgres.with_env("POSTGRES_PASSWORD", "test_pass")
        yield postgres

@pytest.fixture(scope="session")
async def test_engine(postgres_container):
    """Create test database engine."""
    database_url = postgres_container.get_connection_url().replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    engine = create_async_engine(database_url, echo=False)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def db_session(test_engine):
    """Create test database session."""
    async with AsyncSession(test_engine) as session:
        yield session
        await session.rollback()
```

## Unit Testing

### Backend Unit Tests

**Testing API Endpoints**:
```python
import pytest
from fastapi.testclient import TestClient
from mars_gis.main import app

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

@pytest.mark.unit
class TestMissionAPI:
    """Unit tests for mission API endpoints."""

    def test_create_mission_success(self, client, mock_db_session):
        """Test successful mission creation."""
        mission_data = {
            "name": "Test Mission",
            "description": "Test description",
            "mission_type": "rover",
            "target_coordinates": [-14.5684, 175.4729]
        }

        response = client.post("/api/v1/missions", json=mission_data)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == mission_data["name"]
        assert "id" in data["data"]

    def test_create_mission_invalid_coordinates(self, client):
        """Test mission creation with invalid coordinates."""
        mission_data = {
            "name": "Test Mission",
            "target_coordinates": [-95.0, 185.0]  # Invalid coordinates
        }

        response = client.post("/api/v1/missions", json=mission_data)

        assert response.status_code == 422
        error = response.json()
        assert "coordinate" in error["detail"][0]["msg"].lower()

    @pytest.mark.parametrize("missing_field", ["name", "mission_type", "target_coordinates"])
    def test_create_mission_missing_required_fields(self, client, missing_field):
        """Test mission creation with missing required fields."""
        mission_data = {
            "name": "Test Mission",
            "mission_type": "rover",
            "target_coordinates": [-14.5684, 175.4729]
        }
        del mission_data[missing_field]

        response = client.post("/api/v1/missions", json=mission_data)

        assert response.status_code == 422
```

**Testing Business Logic**:
```python
import pytest
from unittest.mock import Mock, patch
from mars_gis.services.mission_service import MissionService
from mars_gis.models.mission import Mission, MissionStatus

@pytest.mark.unit
class TestMissionService:
    """Unit tests for mission service."""

    @pytest.fixture
    def mission_service(self):
        """Create mission service instance."""
        return MissionService()

    async def test_calculate_risk_assessment(self, mission_service):
        """Test risk assessment calculation."""
        mission = Mission(
            name="Test Mission",
            target_coordinates=[-14.5684, 175.4729],
            constraints={"max_slope": 15.0}
        )

        with patch.object(mission_service, '_analyze_terrain_risks') as mock_terrain:
            mock_terrain.return_value = {"slope_risk": 0.3}

            risk_assessment = await mission_service.calculate_risk(mission)

            assert risk_assessment is not None
            assert "overall_risk" in risk_assessment
            assert risk_assessment["overall_risk"] in ["low", "medium", "high"]

    async def test_optimize_mission_path(self, mission_service):
        """Test mission path optimization."""
        mission = Mission(
            name="Test Mission",
            target_coordinates=[-14.5684, 175.4729]
        )

        with patch('mars_gis.services.path_planning.PathPlanningService') as mock_planner:
            mock_planner.optimize.return_value = Mock(
                distance=1500,
                estimated_time=3600,
                waypoints=[(0, 0), (100, 100), (-14.5684, 175.4729)]
            )

            path = await mission_service.optimize_path(mission)

            assert path is not None
            assert path.distance > 0
            assert len(path.waypoints) >= 2
```

**Testing ML Components**:
```python
import pytest
import torch
import numpy as np
from mars_gis.ml.foundation_models import TerrainClassifier

@pytest.mark.unit
class TestTerrainClassifier:
    """Unit tests for terrain classification model."""

    @pytest.fixture
    def model(self):
        """Create terrain classifier instance."""
        return TerrainClassifier()

    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model is not None
        assert hasattr(model, 'classify')
        assert hasattr(model, 'model')

    def test_classify_terrain_valid_input(self, model):
        """Test terrain classification with valid input."""
        # Create mock image tensor
        image_tensor = torch.randn(1, 3, 512, 512)

        with patch.object(model.model, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([[0.1, 0.8, 0.1]])  # Mock prediction

            result = model.classify(image_tensor)

            assert result is not None
            assert 'classification' in result
            assert 'confidence' in result
            assert 0 <= result['confidence'] <= 1

    def test_classify_terrain_invalid_input_shape(self, model):
        """Test terrain classification with invalid input shape."""
        # Wrong shape tensor
        image_tensor = torch.randn(1, 3, 256, 256)  # Model expects 512x512

        with pytest.raises(ValueError, match="Expected input shape"):
            model.classify(image_tensor)
```

### Frontend Unit Tests

**Testing React Components**:
```typescript
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MissionList } from '../components/MissionList';
import { Mission } from '../types/Mission';

const mockMissions: Mission[] = [
  {
    id: '1',
    name: 'Test Mission 1',
    status: 'active',
    type: 'rover',
    targetCoordinates: [-14.5684, 175.4729]
  },
  {
    id: '2',
    name: 'Test Mission 2',
    status: 'planned',
    type: 'orbital',
    targetCoordinates: [-15.2, 176.1]
  }
];

describe('MissionList Component', () => {
  test('renders mission list correctly', () => {
    render(<MissionList missions={mockMissions} />);

    expect(screen.getByText('Test Mission 1')).toBeInTheDocument();
    expect(screen.getByText('Test Mission 2')).toBeInTheDocument();
    expect(screen.getByText('active')).toBeInTheDocument();
    expect(screen.getByText('planned')).toBeInTheDocument();
  });

  test('filters missions by status', async () => {
    const mockOnFilter = jest.fn();
    render(
      <MissionList
        missions={mockMissions}
        onFilter={mockOnFilter}
      />
    );

    const filterSelect = screen.getByRole('combobox', { name: /status filter/i });
    fireEvent.change(filterSelect, { target: { value: 'active' } });

    await waitFor(() => {
      expect(mockOnFilter).toHaveBeenCalledWith({ status: 'active' });
    });
  });

  test('handles empty mission list', () => {
    render(<MissionList missions={[]} />);

    expect(screen.getByText(/no missions found/i)).toBeInTheDocument();
  });
});
```

**Testing Custom Hooks**:
```typescript
import { renderHook, act } from '@testing-library/react';
import { useMissionData } from '../hooks/useMissionData';
import { apiClient } from '../services/apiClient';

jest.mock('../services/apiClient');
const mockedApiClient = apiClient as jest.Mocked<typeof apiClient>;

describe('useMissionData Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('fetches missions on mount', async () => {
    const mockMissions = [
      { id: '1', name: 'Mission 1', status: 'active' }
    ];

    mockedApiClient.getMissions.mockResolvedValue({
      success: true,
      data: { missions: mockMissions }
    });

    const { result } = renderHook(() => useMissionData());

    expect(result.current.loading).toBe(true);

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.missions).toEqual(mockMissions);
    expect(mockedApiClient.getMissions).toHaveBeenCalledTimes(1);
  });

  test('handles API errors gracefully', async () => {
    mockedApiClient.getMissions.mockRejectedValue(new Error('API Error'));

    const { result } = renderHook(() => useMissionData());

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    expect(result.current.error).toBe('API Error');
    expect(result.current.missions).toEqual([]);
  });
});
```

## Integration Testing

### API Integration Tests

**Database Integration**:
```python
import pytest
from httpx import AsyncClient
from mars_gis.main import app

@pytest.mark.integration
class TestMissionAPIIntegration:
    """Integration tests for mission API with database."""

    async def test_mission_crud_workflow(self, test_engine):
        """Test complete mission CRUD workflow."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create mission
            mission_data = {
                "name": "Integration Test Mission",
                "description": "Full workflow test",
                "mission_type": "rover",
                "target_coordinates": [-14.5684, 175.4729]
            }

            create_response = await client.post("/api/v1/missions", json=mission_data)
            assert create_response.status_code == 201

            created_mission = create_response.json()["data"]
            mission_id = created_mission["id"]

            # Read mission
            get_response = await client.get(f"/api/v1/missions/{mission_id}")
            assert get_response.status_code == 200

            retrieved_mission = get_response.json()["data"]
            assert retrieved_mission["name"] == mission_data["name"]

            # Update mission status
            update_response = await client.put(
                f"/api/v1/missions/{mission_id}/status",
                params={"new_status": "active"}
            )
            assert update_response.status_code == 200

            # Verify update
            updated_response = await client.get(f"/api/v1/missions/{mission_id}")
            updated_mission = updated_response.json()["data"]
            assert updated_mission["status"] == "active"

            # List missions (should include our mission)
            list_response = await client.get("/api/v1/missions")
            missions_list = list_response.json()["data"]["missions"]
            assert any(m["id"] == mission_id for m in missions_list)
```

**External Service Integration**:
```python
import pytest
from unittest.mock import patch, AsyncMock
from mars_gis.services.data_service import MarsDataService

@pytest.mark.integration
class TestDataServiceIntegration:
    """Integration tests for external data services."""

    async def test_nasa_data_integration(self):
        """Test integration with NASA data APIs."""
        data_service = MarsDataService()

        # Mock NASA API response
        mock_nasa_response = {
            "imagery": {
                "files": ["MRO_HiRISE_20240101.tif"],
                "metadata": {"resolution": "0.25m", "date": "2024-01-01"}
            }
        }

        with patch.object(data_service.nasa_client, 'get_imagery_data') as mock_nasa:
            mock_nasa.return_value = mock_nasa_response

            query = {
                "lat_min": -15.0, "lat_max": -14.0,
                "lon_min": 175.0, "lon_max": 176.0,
                "data_types": ["imagery"]
            }

            result = await data_service.query_data(query)

            assert result is not None
            assert "nasa_data" in result
            assert "imagery" in result["nasa_data"]
            mock_nasa.assert_called_once()

    async def test_cache_integration(self):
        """Test cache integration with Redis."""
        from mars_gis.core.cache import CacheManager

        cache_manager = CacheManager()
        test_key = "test:integration:key"
        test_data = {"test": "data", "timestamp": "2025-08-01T12:00:00Z"}

        # Set data in cache
        await cache_manager.set(test_key, test_data, ttl=300)

        # Retrieve data from cache
        cached_data = await cache_manager.get(test_key)

        assert cached_data == test_data

        # Test cache expiration
        await cache_manager.delete(test_key)
        expired_data = await cache_manager.get(test_key)

        assert expired_data is None
```

### Service Integration Tests

**ML Model Integration**:
```python
import pytest
import numpy as np
from mars_gis.services.ml_service import MLInferenceService

@pytest.mark.integration
@pytest.mark.slow
class TestMLServiceIntegration:
    """Integration tests for ML inference service."""

    async def test_terrain_classification_pipeline(self):
        """Test complete terrain classification pipeline."""
        ml_service = MLInferenceService()

        # Create mock image data
        mock_image_data = np.random.rand(512, 512, 3).astype(np.uint8)

        inference_request = {
            "model_id": "terrain_classifier",
            "input_data": {
                "image": mock_image_data.tolist(),
                "coordinates": {"lat": -14.5684, "lon": 175.4729}
            }
        }

        result = await ml_service.predict(inference_request)

        assert result is not None
        assert "classification" in result
        assert "confidence" in result
        assert "processing_time" in result
        assert 0 <= result["confidence"] <= 1

    async def test_batch_processing_integration(self):
        """Test batch processing pipeline."""
        ml_service = MLInferenceService()

        batch_requests = [
            {
                "id": f"batch_item_{i}",
                "model_id": "landing_site_optimizer",
                "input_data": {
                    "coordinates": [-14.5 + i*0.1, 175.4 + i*0.1],
                    "constraints": {"max_slope": 15}
                }
            }
            for i in range(5)
        ]

        batch_result = await ml_service.process_batch(batch_requests)

        assert len(batch_result) == 5
        for i, result in enumerate(batch_result):
            assert result["id"] == f"batch_item_{i}"
            assert "optimization_score" in result
```

## End-to-End Testing

### Browser Testing with Playwright

**Setup**:
```python
import pytest
from playwright.async_api import async_playwright, Page, Browser

@pytest.fixture(scope="session")
async def browser():
    """Launch browser for E2E testing."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()

@pytest.fixture
async def page(browser: Browser):
    """Create new page for each test."""
    page = await browser.new_page()
    yield page
    await page.close()
```

**Mission Management Workflow**:
```python
@pytest.mark.e2e
class TestMissionWorkflow:
    """End-to-end tests for mission management workflow."""

    async def test_create_mission_workflow(self, page: Page):
        """Test complete mission creation workflow."""
        # Navigate to application
        await page.goto("http://localhost:3000")

        # Login
        await page.click('[data-testid="login-button"]')
        await page.fill('[data-testid="username"]', 'test_user')
        await page.fill('[data-testid="password"]', 'test_password')
        await page.click('[data-testid="submit-login"]')

        # Wait for dashboard to load
        await page.wait_for_selector('[data-testid="dashboard"]')

        # Navigate to mission creation
        await page.click('[data-testid="create-mission"]')

        # Fill mission form
        await page.fill('[data-testid="mission-name"]', 'E2E Test Mission')
        await page.fill('[data-testid="mission-description"]', 'End-to-end test mission')
        await page.select_option('[data-testid="mission-type"]', 'rover')

        # Set coordinates on map
        await page.click('[data-testid="map-canvas"]', position={'x': 400, 'y': 300})

        # Submit mission
        await page.click('[data-testid="submit-mission"]')

        # Verify mission created
        await page.wait_for_selector('[data-testid="mission-success"]')
        success_message = await page.text_content('[data-testid="mission-success"]')
        assert "successfully created" in success_message.lower()

        # Verify mission appears in list
        await page.click('[data-testid="missions-list"]')
        await page.wait_for_selector('[data-testid="mission-list-item"]')

        mission_items = await page.query_selector_all('[data-testid="mission-list-item"]')
        mission_names = [await item.text_content() for item in mission_items]
        assert any("E2E Test Mission" in name for name in mission_names)

    async def test_mission_visualization_workflow(self, page: Page):
        """Test mission visualization and analysis workflow."""
        await page.goto("http://localhost:3000/missions/test-mission-id")

        # Wait for 3D visualization to load
        await page.wait_for_selector('[data-testid="mars-globe"]')

        # Test zoom and pan interactions
        await page.mouse.wheel(0, -100)  # Zoom in
        await page.mouse.move(400, 300)
        await page.mouse.down()
        await page.mouse.move(450, 350)  # Pan
        await page.mouse.up()

        # Switch to 2D map view
        await page.click('[data-testid="map-2d-toggle"]')
        await page.wait_for_selector('[data-testid="map-2d"]')

        # Add data layers
        await page.click('[data-testid="add-layer"]')
        await page.check('[data-testid="layer-elevation"]')
        await page.check('[data-testid="layer-geology"]')

        # Verify layers are visible
        elevation_layer = await page.query_selector('[data-testid="elevation-layer"]')
        geology_layer = await page.query_selector('[data-testid="geology-layer"]')

        assert await elevation_layer.is_visible()
        assert await geology_layer.is_visible()
```

### API Workflow Testing

**Complete Data Pipeline Test**:
```python
@pytest.mark.e2e
class TestDataPipelineWorkflow:
    """End-to-end tests for data processing pipeline."""

    async def test_complete_data_analysis_workflow(self):
        """Test complete workflow from data query to analysis results."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Step 1: Query available datasets
            datasets_response = await client.get("/api/v1/mars-data/datasets")
            assert datasets_response.status_code == 200

            datasets = datasets_response.json()["data"]
            assert "nasa_datasets" in datasets

            # Step 2: Query Mars data
            query_data = {
                "lat_min": -15.0, "lat_max": -14.0,
                "lon_min": 175.0, "lon_max": 176.0,
                "data_types": ["imagery", "elevation"]
            }

            data_response = await client.post("/api/v1/mars-data/query", json=query_data)
            assert data_response.status_code == 200

            mars_data = data_response.json()["data"]

            # Step 3: Run ML analysis on the data
            ml_request = {
                "model_id": "terrain_classifier",
                "input_data": {
                    "region_data": mars_data,
                    "coordinates": {"lat": -14.5, "lon": 175.5}
                }
            }

            ml_response = await client.post("/api/v1/inference/predict", json=ml_request)
            assert ml_response.status_code == 200

            analysis_result = ml_response.json()["data"]
            assert "classification" in analysis_result

            # Step 4: Create mission based on analysis
            mission_data = {
                "name": "Automated Analysis Mission",
                "description": f"Mission based on terrain analysis: {analysis_result['classification']}",
                "mission_type": "rover",
                "target_coordinates": [-14.5, 175.5],
                "analysis_results": analysis_result
            }

            mission_response = await client.post("/api/v1/missions", json=mission_data)
            assert mission_response.status_code == 201

            mission = mission_response.json()["data"]
            assert mission["name"] == mission_data["name"]
            assert "risk_assessment" in mission
```

## Performance Testing

### Load Testing with Locust

**API Load Tests**:
```python
from locust import HttpUser, task, between

class MarsGISLoadTest(HttpUser):
    """Load testing for MARS-GIS API endpoints."""
    wait_time = between(1, 3)

    def on_start(self):
        """Login before starting tests."""
        response = self.client.post("/auth/token", json={
            "username": "test_user",
            "password": "test_password"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(3)
    def get_missions(self):
        """Test mission listing endpoint under load."""
        self.client.get("/api/v1/missions", headers=self.headers)

    @task(2)
    def query_mars_data(self):
        """Test Mars data query endpoint under load."""
        query_data = {
            "lat_min": -15.0, "lat_max": -14.0,
            "lon_min": 175.0, "lon_max": 176.0,
            "data_types": ["imagery"]
        }
        self.client.post("/api/v1/mars-data/query",
                        json=query_data, headers=self.headers)

    @task(1)
    def create_mission(self):
        """Test mission creation under load."""
        mission_data = {
            "name": f"Load Test Mission {self.user_id}",
            "mission_type": "rover",
            "target_coordinates": [-14.5684, 175.4729]
        }
        self.client.post("/api/v1/missions",
                        json=mission_data, headers=self.headers)

    @task(1)
    def ml_inference(self):
        """Test ML inference under load."""
        inference_data = {
            "model_id": "terrain_classifier",
            "input_data": {"coordinates": {"lat": -14.5, "lon": 175.5}}
        }
        self.client.post("/api/v1/inference/predict",
                        json=inference_data, headers=self.headers)
```

### Performance Benchmarks

**Database Performance Tests**:
```python
import pytest
import time
import asyncio
from mars_gis.services.mission_service import MissionService

@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    async def test_mission_query_performance(self, db_session):
        """Test mission query performance with large dataset."""
        mission_service = MissionService(db_session)

        # Create test data
        test_missions = []
        for i in range(1000):
            mission_data = {
                "name": f"Performance Test Mission {i}",
                "mission_type": "rover",
                "target_coordinates": [-14.5 + i*0.001, 175.4 + i*0.001]
            }
            test_missions.append(mission_data)

        # Bulk insert test missions
        await mission_service.bulk_create(test_missions)

        # Benchmark query performance
        start_time = time.time()

        for _ in range(100):  # 100 queries
            missions = await mission_service.list_missions(limit=10)
            assert len(missions) <= 10

        end_time = time.time()
        avg_query_time = (end_time - start_time) / 100

        # Assert query time is under 50ms
        assert avg_query_time < 0.05, f"Query too slow: {avg_query_time:.3f}s"

    async def test_ml_inference_performance(self):
        """Test ML inference performance."""
        from mars_gis.services.ml_service import MLInferenceService

        ml_service = MLInferenceService()

        # Prepare test data
        test_requests = [
            {
                "model_id": "terrain_classifier",
                "input_data": {"coordinates": {"lat": -14.5, "lon": 175.5}}
            }
            for _ in range(50)
        ]

        # Benchmark inference performance
        start_time = time.time()

        tasks = [ml_service.predict(req) for req in test_requests]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time
        avg_inference_time = total_time / len(test_requests)

        # Assert inference time is under 2 seconds
        assert avg_inference_time < 2.0, f"Inference too slow: {avg_inference_time:.3f}s"
        assert len(results) == len(test_requests)
```

## Security Testing

### Authentication and Authorization Tests

```python
import pytest
from httpx import AsyncClient
from mars_gis.main import app

@pytest.mark.security
class TestSecurityEndpoints:
    """Security tests for API endpoints."""

    async def test_unauthenticated_access_blocked(self):
        """Test that protected endpoints require authentication."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            protected_endpoints = [
                "/api/v1/missions",
                "/api/v1/mars-data/query",
                "/api/v1/inference/predict"
            ]

            for endpoint in protected_endpoints:
                response = await client.get(endpoint)
                assert response.status_code == 401

    async def test_role_based_access_control(self):
        """Test role-based access control."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test viewer role (read-only access)
            viewer_token = await self._get_token(client, "viewer_user", "viewer_pass")
            viewer_headers = {"Authorization": f"Bearer {viewer_token}"}

            # Viewer can read missions
            get_response = await client.get("/api/v1/missions", headers=viewer_headers)
            assert get_response.status_code == 200

            # Viewer cannot create missions
            mission_data = {"name": "Test", "mission_type": "rover"}
            create_response = await client.post("/api/v1/missions",
                                              json=mission_data, headers=viewer_headers)
            assert create_response.status_code == 403

    async def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            token = await self._get_token(client, "test_user", "test_pass")
            headers = {"Authorization": f"Bearer {token}"}

            # Attempt SQL injection in query parameters
            malicious_queries = [
                "'; DROP TABLE missions; --",
                "1' OR '1'='1",
                "' UNION SELECT * FROM users --"
            ]

            for malicious_query in malicious_queries:
                response = await client.get(
                    f"/api/v1/missions",
                    params={"status": malicious_query},
                    headers=headers
                )
                # Should not return 500 error or expose database structure
                assert response.status_code in [200, 400, 422]
                if response.status_code == 200:
                    # Response should not contain SQL error messages
                    response_text = response.text.lower()
                    assert "sql" not in response_text
                    assert "error" not in response_text or "validation error" in response_text

    async def test_input_validation_and_sanitization(self):
        """Test input validation prevents malicious payloads."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            token = await self._get_token(client, "test_user", "test_pass")
            headers = {"Authorization": f"Bearer {token}"}

            # Test XSS prevention
            xss_payloads = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "' onerror='alert(1)'"
            ]

            for payload in xss_payloads:
                mission_data = {
                    "name": payload,
                    "mission_type": "rover",
                    "target_coordinates": [-14.5684, 175.4729]
                }

                response = await client.post("/api/v1/missions",
                                           json=mission_data, headers=headers)

                if response.status_code == 201:
                    # Check that payload was sanitized
                    created_mission = response.json()["data"]
                    assert "<script>" not in created_mission["name"]
                    assert "javascript:" not in created_mission["name"]

    async def _get_token(self, client: AsyncClient, username: str, password: str) -> str:
        """Helper method to get authentication token."""
        response = await client.post("/auth/token", json={
            "username": username,
            "password": password
        })
        return response.json()["access_token"]
```

### Penetration Testing

```python
import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

@pytest.mark.security
@pytest.mark.slow
class TestPenetrationTesting:
    """Automated penetration testing."""

    def setup_method(self):
        """Setup HTTP session with retries."""
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def test_rate_limiting(self):
        """Test API rate limiting."""
        base_url = "http://localhost:8000"

        # Make rapid requests to trigger rate limiting
        responses = []
        for _ in range(150):  # Exceed rate limit
            response = self.session.get(f"{base_url}/health")
            responses.append(response.status_code)

        # Should receive 429 (Too Many Requests) responses
        assert 429 in responses, "Rate limiting not working"

    def test_directory_traversal_prevention(self):
        """Test directory traversal attack prevention."""
        base_url = "http://localhost:8000"

        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]

        for payload in traversal_payloads:
            response = self.session.get(f"{base_url}/static/{payload}")

            # Should not return sensitive system files
            assert response.status_code in [404, 403, 400]
            if response.status_code == 200:
                content = response.text.lower()
                assert "root:" not in content  # Unix passwd file
                assert "[boot loader]" not in content  # Windows files

    def test_cors_configuration(self):
        """Test CORS configuration security."""
        base_url = "http://localhost:8000"

        # Test CORS with malicious origin
        malicious_origins = [
            "http://evil.com",
            "javascript:alert(1)",
            "*"
        ]

        for origin in malicious_origins:
            headers = {"Origin": origin}
            response = self.session.options(f"{base_url}/api/v1/missions", headers=headers)

            # Should not allow arbitrary origins
            cors_origin = response.headers.get("Access-Control-Allow-Origin")
            if cors_origin:
                assert cors_origin != "*" or origin == "http://localhost:3000"
                assert "evil.com" not in cors_origin
```

## Test Automation

### GitHub Actions CI/CD Pipeline

**.github/workflows/test.yml**:
```yaml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgis/postgis:13-3.1
        env:
          POSTGRES_DB: test_mars_gis
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_pass
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:6.2-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=mars_gis --cov-report=xml
      env:
        DATABASE_URL: postgresql://test_user:test_pass@localhost:5432/test_mars_gis
        REDIS_URL: redis://localhost:6379/0

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
      env:
        DATABASE_URL: postgresql://test_user:test_pass@localhost:5432/test_mars_gis
        REDIS_URL: redis://localhost:6379/0

    - name: Run security tests
      run: |
        pytest tests/security/ -v

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  e2e-tests:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install playwright pytest-playwright
        playwright install

    - name: Start application
      run: |
        python -m mars_gis.main &
        sleep 10
      env:
        MARS_GIS_ENV: testing

    - name: Run E2E tests
      run: |
        pytest tests/e2e/ -v --headed

  performance-tests:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install locust

    - name: Run performance tests
      run: |
        pytest tests/performance/ -v -m "not slow"
```

### Pre-commit Hooks

**.pre-commit-config.yaml**:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings]

  - repo: local
    hooks:
      - id: pytest-fast
        name: pytest-fast
        entry: pytest tests/unit/ -x --ff
        language: system
        pass_filenames: false
        always_run: true
```

## Test Reporting

### Coverage Reports

**Generate comprehensive coverage reports**:
```bash
# Generate HTML coverage report
pytest --cov=mars_gis --cov-report=html

# Generate XML coverage report for CI
pytest --cov=mars_gis --cov-report=xml

# Generate terminal coverage report
pytest --cov=mars_gis --cov-report=term-missing
```

### Test Results Dashboard

**Allure Reports Integration**:
```python
# conftest.py additions
import allure
from allure_commons.types import AttachmentType

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Attach test artifacts to Allure report."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        # Attach screenshot for E2E tests
        if hasattr(item, 'funcargs') and 'page' in item.funcargs:
            page = item.funcargs['page']
            screenshot = page.screenshot()
            allure.attach(screenshot, name="screenshot", attachment_type=AttachmentType.PNG)

        # Attach logs
        if hasattr(item, 'caplog'):
            allure.attach(item.caplog.text, name="logs", attachment_type=AttachmentType.TEXT)
```

### Continuous Monitoring

**Test metrics collection**:
```python
import time
import json
from datetime import datetime

class TestMetricsCollector:
    """Collect and report test metrics."""

    def __init__(self):
        self.metrics = {
            "test_run_id": str(uuid.uuid4()),
            "start_time": datetime.utcnow().isoformat(),
            "tests": []
        }

    def record_test(self, test_name: str, duration: float, status: str):
        """Record individual test metrics."""
        self.metrics["tests"].append({
            "name": test_name,
            "duration": duration,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        })

    def generate_report(self) -> dict:
        """Generate comprehensive test report."""
        total_tests = len(self.metrics["tests"])
        passed_tests = len([t for t in self.metrics["tests"] if t["status"] == "passed"])
        failed_tests = total_tests - passed_tests

        avg_duration = sum(t["duration"] for t in self.metrics["tests"]) / total_tests if total_tests > 0 else 0

        return {
            **self.metrics,
            "end_time": datetime.utcnow().isoformat(),
            "summary": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "average_duration": avg_duration
            }
        }
```

---

**Test Execution Commands**

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m security
pytest -m performance

# Run tests with coverage
pytest --cov=mars_gis --cov-report=html

# Run tests in parallel
pytest -n auto

# Run tests with detailed output
pytest -v --tb=short

# Run specific test file
pytest tests/test_mission_api.py

# Run tests matching pattern
pytest -k "test_mission"
```
