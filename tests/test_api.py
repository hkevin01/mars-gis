"""API integration tests for MARS-GIS platform."""

import json
from unittest.mock import Mock, patch

import pytest


@pytest.mark.api
class TestMissionAPI:
    """Test cases for mission management API endpoints."""
    
    def test_get_missions_endpoint(self, api_client, auth_headers):
        """Test GET /api/v1/missions endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        response = api_client.get("/api/v1/missions", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "missions" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert isinstance(data["missions"], list)
    
    def test_create_mission_endpoint(self, api_client, auth_headers, mock_mission_data):
        """Test POST /api/v1/missions endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        response = api_client.post(
            "/api/v1/missions",
            headers=auth_headers,
            json=mock_mission_data
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["name"] == mock_mission_data["name"]
        assert data["status"] == "planned"
        assert "id" in data
        assert "created_at" in data
    
    def test_get_mission_by_id(self, api_client, auth_headers):
        """Test GET /api/v1/missions/{id} endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        mission_id = "test-mission-001"
        response = api_client.get(f"/api/v1/missions/{mission_id}", headers=auth_headers)
        
        if response.status_code == 404:
            pytest.skip("Test mission not found")
            
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == mission_id
        assert "name" in data
        assert "status" in data
        assert "tasks" in data
    
    def test_update_mission_endpoint(self, api_client, auth_headers):
        """Test PUT /api/v1/missions/{id} endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        mission_id = "test-mission-001"
        update_data = {
            "name": "Updated Test Mission",
            "description": "Updated description"
        }
        
        response = api_client.put(
            f"/api/v1/missions/{mission_id}",
            headers=auth_headers,
            json=update_data
        )
        
        if response.status_code == 404:
            pytest.skip("Test mission not found")
            
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]
    
    def test_start_mission_endpoint(self, api_client, auth_headers):
        """Test POST /api/v1/missions/{id}/start endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        mission_id = "test-mission-001"
        response = api_client.post(
            f"/api/v1/missions/{mission_id}/start",
            headers=auth_headers
        )
        
        if response.status_code == 404:
            pytest.skip("Test mission not found")
            
        assert response.status_code in [200, 202]  # OK or Accepted
        data = response.json()
        
        assert data["status"] in ["active", "starting"]
    
    def test_mission_validation_errors(self, api_client, auth_headers):
        """Test mission creation with invalid data."""
        if not api_client:
            pytest.skip("API client not available")
            
        invalid_mission = {
            "name": "",  # Empty name should be invalid
            "description": "Test mission",
            "asset_id": ""  # Empty asset ID should be invalid
        }
        
        response = api_client.post(
            "/api/v1/missions",
            headers=auth_headers,
            json=invalid_mission
        )
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        
        assert "detail" in data
        assert isinstance(data["detail"], list)


@pytest.mark.api
class TestAssetAPI:
    """Test cases for asset management API endpoints."""
    
    def test_get_assets_endpoint(self, api_client, auth_headers):
        """Test GET /api/v1/assets endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        response = api_client.get("/api/v1/assets", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        if len(data) > 0:
            asset = data[0]
            assert "id" in asset
            assert "name" in asset
            assert "type" in asset
            assert "status" in asset
            assert "location" in asset
    
    def test_get_asset_status(self, api_client, auth_headers):
        """Test GET /api/v1/assets/{id}/status endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        asset_id = "test-rover-001"
        response = api_client.get(f"/api/v1/assets/{asset_id}/status", headers=auth_headers)
        
        if response.status_code == 404:
            pytest.skip("Test asset not found")
            
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "last_update" in data
        assert data["status"] in ["online", "offline", "maintenance", "charging"]
    
    def test_send_asset_command(self, api_client, auth_headers):
        """Test POST /api/v1/assets/{id}/commands endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        asset_id = "test-rover-001"
        command_data = {
            "command": "move",
            "parameters": {
                "target_coordinates": [-14.5684, 175.4729],
                "speed": 0.5
            }
        }
        
        response = api_client.post(
            f"/api/v1/assets/{asset_id}/commands",
            headers=auth_headers,
            json=command_data
        )
        
        if response.status_code == 404:
            pytest.skip("Test asset not found")
            
        assert response.status_code in [200, 202]  # OK or Accepted


@pytest.mark.api
class TestAnalysisAPI:
    """Test cases for analysis API endpoints."""
    
    def test_get_analysis_results(self, api_client, auth_headers):
        """Test GET /api/v1/analysis/results endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        response = api_client.get("/api/v1/analysis/results", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert isinstance(data["items"], list)
    
    def test_create_terrain_analysis(self, api_client, auth_headers):
        """Test POST /api/v1/analysis/terrain endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        analysis_request = {
            "type": "terrain",
            "region": "Olympia Undae",
            "coordinates": {
                "latitude": -14.5684,
                "longitude": 175.4729,
                "radius": 1000
            },
            "parameters": {
                "analysis_depth": "detailed",
                "include_hazards": True
            }
        }
        
        response = api_client.post(
            "/api/v1/analysis/terrain",
            headers=auth_headers,
            json=analysis_request
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["type"] == "terrain"
        assert data["region"] == "Olympia Undae"
        assert data["status"] in ["queued", "processing"]
        assert "id" in data
    
    def test_get_analysis_by_id(self, api_client, auth_headers):
        """Test GET /api/v1/analysis/results/{id} endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        analysis_id = "analysis-001"
        response = api_client.get(
            f"/api/v1/analysis/results/{analysis_id}",
            headers=auth_headers
        )
        
        if response.status_code == 404:
            pytest.skip("Test analysis not found")
            
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == analysis_id
        assert "type" in data
        assert "status" in data
        assert "results" in data or data["status"] != "completed"


@pytest.mark.api
class TestDataAPI:
    """Test cases for data API endpoints."""
    
    def test_get_atmospheric_data(self, api_client, auth_headers):
        """Test GET /api/v1/data/atmospheric endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        params = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "limit": 100
        }
        
        response = api_client.get(
            "/api/v1/data/atmospheric",
            headers=auth_headers,
            params=params
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        if len(data) > 0:
            record = data[0]
            assert "timestamp" in record
            assert "temperature" in record
            assert "pressure" in record
            assert "coordinates" in record
    
    def test_get_geological_data(self, api_client, auth_headers):
        """Test GET /api/v1/data/geological endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        params = {
            "sample_type": "soil",
            "region": "Gale Crater",
            "limit": 50
        }
        
        response = api_client.get(
            "/api/v1/data/geological",
            headers=auth_headers,
            params=params
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        if len(data) > 0:
            sample = data[0]
            assert "id" in sample
            assert "sample_type" in sample
            assert "collection_date" in sample
            assert "coordinates" in sample
            assert "composition" in sample
    
    def test_get_terrain_data(self, api_client, auth_headers):
        """Test GET /api/v1/data/terrain endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        params = {
            "bounds": "-15,175,-14,176",  # minLat,minLon,maxLat,maxLon
            "classification": "rocky_terrain",
            "limit": 200
        }
        
        response = api_client.get(
            "/api/v1/data/terrain",
            headers=auth_headers,
            params=params
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        if len(data) > 0:
            terrain = data[0]
            assert "coordinates" in terrain
            assert "elevation" in terrain
            assert "classification" in terrain
            assert "hazard_level" in terrain
    
    def test_get_map_tiles(self, api_client):
        """Test GET /api/v1/data/tiles/{layer}/{z}/{x}/{y}.png endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        # Test basic map tile request
        response = api_client.get("/api/v1/data/tiles/topographic/5/16/8.png")
        
        if response.status_code == 404:
            pytest.skip("Tile service not available")
            
        assert response.status_code == 200
        assert response.headers.get("content-type") == "image/png"
        assert len(response.content) > 0


@pytest.mark.api
class TestSystemAPI:
    """Test cases for system API endpoints."""
    
    def test_system_status_endpoint(self, api_client):
        """Test GET /api/v1/system/status endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        response = api_client.get("/api/v1/system/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "uptime" in data
        assert "version" in data
        assert "services" in data
        assert data["status"] in ["healthy", "degraded", "down"]
    
    def test_system_metrics_endpoint(self, api_client, auth_headers):
        """Test GET /api/v1/system/metrics endpoint."""
        if not api_client:
            pytest.skip("API client not available")
            
        response = api_client.get("/api/v1/system/metrics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "cpu_usage" in data
        assert "memory_usage" in data
        assert "disk_usage" in data
        assert "active_connections" in data
        assert 0 <= data["cpu_usage"] <= 100
        assert 0 <= data["memory_usage"] <= 100


@pytest.mark.api
class TestAuthentication:
    """Test cases for authentication and authorization."""
    
    def test_unauthenticated_request(self, api_client):
        """Test that protected endpoints require authentication."""
        if not api_client:
            pytest.skip("API client not available")
            
        response = api_client.get("/api/v1/missions")
        
        assert response.status_code == 401  # Unauthorized
    
    def test_invalid_token(self, api_client):
        """Test request with invalid authentication token."""
        if not api_client:
            pytest.skip("API client not available")
            
        invalid_headers = {
            "Authorization": "Bearer invalid-token",
            "Content-Type": "application/json"
        }
        
        response = api_client.get("/api/v1/missions", headers=invalid_headers)
        
        assert response.status_code == 401  # Unauthorized
    
    def test_expired_token(self, api_client):
        """Test request with expired authentication token."""
        if not api_client:
            pytest.skip("API client not available")
            
        expired_headers = {
            "Authorization": "Bearer expired-token",
            "Content-Type": "application/json"
        }
        
        response = api_client.get("/api/v1/missions", headers=expired_headers)
        
        assert response.status_code == 401  # Unauthorized


@pytest.mark.api
class TestAPIValidation:
    """Test cases for API input validation."""
    
    def test_invalid_json_payload(self, api_client, auth_headers):
        """Test API response to invalid JSON payload."""
        if not api_client:
            pytest.skip("API client not available")
            
        response = api_client.post(
            "/api/v1/missions",
            headers=auth_headers,
            data="invalid-json-payload"  # Invalid JSON
        )
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self, api_client, auth_headers):
        """Test API response to missing required fields."""
        if not api_client:
            pytest.skip("API client not available")
            
        incomplete_mission = {
            "description": "Missing name and asset_id"
        }
        
        response = api_client.post(
            "/api/v1/missions",
            headers=auth_headers,
            json=incomplete_mission
        )
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        
        assert "detail" in data
        error_fields = [error["loc"][-1] for error in data["detail"]]
        assert "name" in error_fields
        assert "asset_id" in error_fields
    
    def test_invalid_coordinates(self, api_client, auth_headers):
        """Test API response to invalid coordinate values."""
        if not api_client:
            pytest.skip("API client not available")
            
        invalid_analysis = {
            "type": "terrain",
            "region": "Test Region",
            "coordinates": {
                "latitude": 200,  # Invalid latitude (>90)
                "longitude": 400,  # Invalid longitude (>180)
                "radius": -100    # Invalid negative radius
            }
        }
        
        response = api_client.post(
            "/api/v1/analysis/terrain",
            headers=auth_headers,
            json=invalid_analysis
        )
        
        assert response.status_code == 422  # Validation error


@pytest.mark.api
@pytest.mark.slow
class TestAPIPerformance:
    """Test cases for API performance and load handling."""
    
    def test_pagination_performance(self, api_client, auth_headers):
        """Test API pagination with large datasets."""
        if not api_client:
            pytest.skip("API client not available")
            
        # Test various page sizes
        for page_size in [10, 50, 100, 500]:
            params = {"page": 1, "page_size": page_size}
            
            response = api_client.get(
                "/api/v1/missions",
                headers=auth_headers,
                params=params
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify pagination parameters
            assert len(data["missions"]) <= page_size
            assert data["limit"] == page_size
    
    def test_concurrent_requests(self, api_client, auth_headers):
        """Test API handling of concurrent requests."""
        if not api_client:
            pytest.skip("API client not available")
            
        import concurrent.futures
        import time
        
        def make_request():
            start_time = time.time()
            response = api_client.get("/api/v1/system/status")
            end_time = time.time()
            return response.status_code, end_time - start_time
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        status_codes = [result[0] for result in results]
        assert all(code == 200 for code in status_codes)
        
        # Response times should be reasonable (< 5 seconds)
        response_times = [result[1] for result in results]
        assert all(time < 5.0 for time in response_times)
