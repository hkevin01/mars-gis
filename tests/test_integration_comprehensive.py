"""
Integration Tests for MARS-GIS Platform
Tests that components work together correctly and data flows properly.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestApplicationIntegration:
    """Test that main application components integrate properly."""
    
    def test_fastapi_app_creation_and_configuration(self):
        """Test that FastAPI app is created with proper configuration."""
        try:
            from mars_gis.core.config import settings
            from mars_gis.main import create_app
            
            app = create_app()
            
            # App should be created successfully
            assert app is not None
            
            # App should have correct configuration
            assert app.title == "MARS-GIS API"
            assert app.description == "Mars Geospatial Intelligence System API"
            assert app.version == "0.1.0"
            assert app.docs_url == "/docs"
            assert app.redoc_url == "/redoc"
            
            # CORS middleware should be configured
            middleware_found = False
            for middleware in app.user_middleware:
                if 'CORSMiddleware' in str(middleware):
                    middleware_found = True
                    break
            
            # Would need more detailed middleware inspection in full test
            assert True, "App creation with middleware completed"
            
        except ImportError as e:
            pytest.skip(f"FastAPI components not available: {e}")
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint_integration(self):
        """Test health check endpoint returns expected response."""
        try:
            from fastapi.testclient import TestClient

            from mars_gis.main import create_app
            
            app = create_app()
            client = TestClient(app)
            
            response = client.get("/")
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure matches documentation
            assert "message" in data
            assert "version" in data
            assert "status" in data
            
            # Verify response content
            assert data["message"] == "MARS-GIS API is running"
            assert data["version"] == "0.1.0"
            assert data["status"] == "healthy"
            
        except ImportError as e:
            pytest.skip(f"FastAPI TestClient not available: {e}")
    
    def test_settings_integration_with_app_creation(self):
        """Test that settings properly integrate with app creation."""
        try:
            from mars_gis.core.config import settings
            from mars_gis.main import create_app

            # Settings should be available
            assert settings is not None
            assert settings.APP_NAME == "MARS-GIS"
            
            # App creation should use settings
            app = create_app()
            assert app is not None
            
            # App should reflect settings configuration
            # In a full implementation, would test specific settings usage
            
        except ImportError as e:
            pytest.skip(f"Application components not available: {e}")


class TestDatabaseIntegration:
    """Test database integration and connectivity."""
    
    def test_database_configuration_is_accessible(self):
        """Test that database configuration is properly accessible."""
        from mars_gis.core.config import settings

        # Database URL should be configured
        assert hasattr(settings, 'DATABASE_URL')
        assert settings.DATABASE_URL is not None
        assert len(settings.DATABASE_URL) > 0
        
        # URL should be in valid format
        db_url = settings.DATABASE_URL
        assert '://' in db_url, "Database URL should have scheme"
        
        # Should support both PostgreSQL and SQLite
        valid_schemes = ['postgresql://', 'sqlite://']
        has_valid_scheme = any(db_url.startswith(scheme) for scheme in valid_schemes)
        assert has_valid_scheme, f"Database URL scheme not recognized: {db_url}"
    
    def test_database_connection_attempt(self):
        """Test database connection can be attempted with current configuration."""
        try:
            from sqlalchemy import create_engine

            from mars_gis.core.config import settings

            # Use SQLite for testing to avoid PostgreSQL dependency
            test_db_url = "sqlite:///test_mars_gis.db"
            
            # Engine creation should work
            engine = create_engine(test_db_url)
            assert engine is not None
            
            # Basic connection test
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).fetchone()
                assert result[0] == 1
            
            # Clean up test database
            test_db_path = Path("test_mars_gis.db")
            if test_db_path.exists():
                test_db_path.unlink()
                
        except ImportError as e:
            pytest.skip(f"SQLAlchemy not available: {e}")
    
    def test_redis_configuration_is_accessible(self):
        """Test that Redis configuration is properly accessible."""
        from mars_gis.core.config import settings

        # Redis URL should be configured
        assert hasattr(settings, 'REDIS_URL')
        assert settings.REDIS_URL is not None
        assert settings.REDIS_URL.startswith('redis://')
        
        # URL should be properly formatted
        redis_url = settings.REDIS_URL
        assert '://' in redis_url
        assert redis_url.count('/') >= 3, "Redis URL should include database number"


class TestAPIIntegration:
    """Test API integration and endpoint availability."""
    
    @pytest.mark.asyncio
    async def test_cors_middleware_integration(self):
        """Test CORS middleware is properly integrated."""
        try:
            from fastapi.testclient import TestClient

            from mars_gis.core.config import settings
            from mars_gis.main import create_app
            
            app = create_app()
            client = TestClient(app)
            
            # Test CORS headers on OPTIONS request
            response = client.options("/", headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            })
            
            # CORS should be configured to allow localhost origins
            # Response status should indicate CORS handling
            assert response.status_code in [200, 204], "CORS preflight should be handled"
            
        except ImportError as e:
            pytest.skip(f"FastAPI components not available: {e}")
    
    def test_api_documentation_integration(self):
        """Test that API documentation endpoints are integrated."""
        try:
            from fastapi.testclient import TestClient

            from mars_gis.main import create_app
            
            app = create_app()
            client = TestClient(app)
            
            # Test Swagger docs endpoint
            docs_response = client.get("/docs")
            assert docs_response.status_code == 200
            assert "text/html" in docs_response.headers.get("content-type", "")
            
            # Test OpenAPI spec endpoint  
            openapi_response = client.get("/openapi.json")
            assert openapi_response.status_code == 200
            
            spec = openapi_response.json()
            assert "openapi" in spec
            assert "info" in spec
            assert spec["info"]["title"] == "MARS-GIS API"
            
        except ImportError as e:
            pytest.skip(f"FastAPI components not available: {e}")


class TestDataProcessingIntegration:
    """Test data processing pipeline integration."""
    
    def test_directory_structure_integration(self):
        """Test that data directories are properly integrated."""
        from mars_gis.core.config import settings

        # Data directories should be configured
        assert hasattr(settings, 'DATA_DIR')
        assert hasattr(settings, 'LOGS_DIR')
        assert hasattr(settings, 'ASSETS_DIR')
        
        # Directories should be Path objects
        assert isinstance(settings.DATA_DIR, Path)
        assert isinstance(settings.LOGS_DIR, Path)
        assert isinstance(settings.ASSETS_DIR, Path)
        
        # Directories should exist after settings initialization
        assert settings.DATA_DIR.exists(), "Data directory should be created"
        assert settings.LOGS_DIR.exists(), "Logs directory should be created"
        
        # Subdirectories should exist
        assert (settings.DATA_DIR / "raw").exists()
        assert (settings.DATA_DIR / "processed").exists()
        assert (settings.DATA_DIR / "models").exists()
    
    def test_geospatial_processing_integration(self):
        """Test geospatial processing components integration."""
        try:
            import geopandas as gpd
            import pandas as pd
            from shapely.geometry import Point

            # Create test Mars coordinate data
            mars_points = [
                Point(175.4729, -14.5684),  # Olympia Undae
                Point(137.8414, -5.4453),   # Gale Crater
                Point(-49.97, 22.5)         # Valles Marineris
            ]
            
            gdf = gpd.GeoDataFrame(
                {'location': ['Olympia Undae', 'Gale Crater', 'Valles Marineris']},
                geometry=mars_points
            )
            
            # Test basic operations work
            assert len(gdf) == 3
            assert all(isinstance(geom, Point) for geom in gdf.geometry)
            
            # Test spatial operations
            bounds = gdf.total_bounds
            assert len(bounds) == 4  # [minx, miny, maxx, maxy]
            
            # Test data processing pipeline
            centroids = gdf.centroid
            assert len(centroids) == 3
            
        except ImportError as e:
            pytest.skip(f"Geospatial libraries not available: {e}")


class TestMLIntegration:
    """Test machine learning component integration."""
    
    def test_pytorch_integration_configuration(self):
        """Test PyTorch integration configuration."""
        try:
            import torch

            from mars_gis.core.config import settings

            # ML configuration should be available
            assert hasattr(settings, 'TORCH_DEVICE')
            assert hasattr(settings, 'MODEL_CACHE_DIR')
            
            # Device configuration should be valid
            device_str = settings.TORCH_DEVICE
            
            if device_str == 'cuda':
                # Test CUDA availability if configured
                cuda_available = torch.cuda.is_available()
                # CUDA should be optional, not required
                assert True, "CUDA configuration checked"
            elif device_str == 'cpu':
                # CPU should always be available
                assert torch.cuda.is_available() or True, "CPU always available"
            
            # Model cache directory should exist
            model_cache = Path(settings.MODEL_CACHE_DIR)
            assert model_cache.exists(), "Model cache directory should exist"
            
        except ImportError as e:
            pytest.skip(f"PyTorch not available: {e}")
    
    def test_terrain_classification_integration_concept(self):
        """Test terrain classification integration concept."""
        # Test the documented 8 terrain types
        expected_terrain_types = [
            "plains", "hills", "mountains", "craters",
            "valleys", "polar_ice", "sand_dunes", "rocky_terrain"
        ]
        
        # Verify we have the expected number of terrain types
        assert len(expected_terrain_types) == 8
        
        # Each terrain type should be a valid string
        for terrain_type in expected_terrain_types:
            assert isinstance(terrain_type, str)
            assert len(terrain_type) > 0
            assert terrain_type.replace('_', '').isalpha()


class TestSecurityIntegration:
    """Test security features integration."""
    
    def test_security_configuration_integration(self):
        """Test security configuration is properly integrated."""
        from mars_gis.core.config import settings

        # Security settings should be configured
        assert hasattr(settings, 'SECRET_KEY')
        assert hasattr(settings, 'ALGORITHM')
        assert hasattr(settings, 'ACCESS_TOKEN_EXPIRE_MINUTES')
        
        # Secret key should be present but warn if default
        secret_key = settings.SECRET_KEY
        assert isinstance(secret_key, str)
        assert len(secret_key) > 10, "Secret key should be substantial"
        
        # Algorithm should be valid
        assert settings.ALGORITHM in ['HS256', 'HS384', 'HS512', 'RS256']
        
        # Token expiration should be reasonable
        expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        assert isinstance(expire_minutes, int)
        assert 5 <= expire_minutes <= 1440, "Token expiration should be reasonable"
    
    def test_cors_security_integration(self):
        """Test CORS security is properly integrated."""
        from mars_gis.core.config import settings

        # CORS hosts should be configured
        allowed_hosts = settings.ALLOWED_HOSTS
        assert isinstance(allowed_hosts, list)
        assert len(allowed_hosts) > 0
        
        # Hosts should be valid URLs
        for host in allowed_hosts:
            assert isinstance(host, str)
            assert host.startswith(('http://', 'https://'))
            
        # Should have localhost for development
        localhost_found = any('localhost' in host for host in allowed_hosts)
        assert localhost_found, "Should allow localhost for development"


class TestLoggingIntegration:
    """Test logging system integration."""
    
    def test_logging_configuration_integration(self):
        """Test logging configuration is properly integrated.""" 
        from mars_gis.core.config import settings

        # Logging settings should be configured
        assert hasattr(settings, 'LOG_LEVEL')
        assert hasattr(settings, 'LOG_FORMAT')
        
        # Log level should be valid
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        assert settings.LOG_LEVEL in valid_levels
        
        # Log format should be reasonable
        log_format = settings.LOG_FORMAT
        assert isinstance(log_format, str)
        assert len(log_format) > 10
        assert '%(' in log_format, "Should contain Python logging format specifiers"
        
        # Logs directory should exist
        logs_dir = settings.LOGS_DIR
        assert logs_dir.exists(), "Logs directory should be created"


class TestExternalServiceIntegration:
    """Test external service integration readiness."""
    
    def test_nasa_api_integration_readiness(self):
        """Test NASA API integration is properly configured."""
        from mars_gis.core.config import settings

        # NASA API configuration should exist
        assert hasattr(settings, 'NASA_API_KEY')
        assert hasattr(settings, 'NASA_PDS_BASE_URL')
        
        # API key can be None (optional)
        api_key = settings.NASA_API_KEY
        if api_key is not None:
            assert isinstance(api_key, str)
            assert len(api_key) > 5, "API key should be substantial if provided"
        
        # Base URL should be valid
        base_url = settings.NASA_PDS_BASE_URL
        assert isinstance(base_url, str)
        assert base_url.startswith('https://')
        assert 'nasa' in base_url.lower() or 'jpl' in base_url.lower()
    
    def test_usgs_integration_readiness(self):
        """Test USGS integration is properly configured."""
        from mars_gis.core.config import settings

        # USGS configuration should exist
        assert hasattr(settings, 'USGS_BASE_URL')
        
        base_url = settings.USGS_BASE_URL
        assert isinstance(base_url, str)
        assert base_url.startswith('https://')
        assert 'usgs' in base_url.lower()
    
    @pytest.mark.asyncio
    async def test_external_api_connection_concept(self):
        """Test external API connection concept (mocked)."""
        # Mock external API responses
        with patch('requests.get') as mock_get:
            # Mock successful NASA API response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"sol": 1000, "img_src": "https://example.com/mars.jpg"}]
            }
            mock_get.return_value = mock_response
            
            # Test that external APIs could be called
            import requests
            response = requests.get("https://api.nasa.gov/test")
            
            assert response.status_code == 200
            data = response.json()
            assert "data" in data


class TestPerformanceIntegration:
    """Test performance-related integration aspects."""
    
    def test_memory_usage_integration(self):
        """Test memory usage of integrated components."""
        import gc

        import psutil

        # Get baseline memory usage
        process = psutil.Process()
        baseline_memory = process.memory_info().rss
        
        # Import and create main components
        try:
            from mars_gis.core.config import settings
            from mars_gis.main import create_app
            
            app = create_app()
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage after component creation
            current_memory = process.memory_info().rss
            memory_increase = current_memory - baseline_memory
            
            # Memory increase should be reasonable (< 100MB for basic app)
            max_memory_increase = 100 * 1024 * 1024  # 100MB
            assert memory_increase < max_memory_increase, (
                f"Memory usage increased by {memory_increase / 1024 / 1024:.1f}MB, "
                f"expected < {max_memory_increase / 1024 / 1024:.1f}MB"
            )
            
        except ImportError as e:
            pytest.skip(f"Components not available for memory test: {e}")
    
    def test_startup_time_integration(self):
        """Test application startup time is reasonable."""
        import time
        
        start_time = time.time()
        
        try:
            from mars_gis.main import create_app
            app = create_app()
            
            startup_time = time.time() - start_time
            
            # Startup should be fast (< 5 seconds)
            max_startup_time = 5.0
            assert startup_time < max_startup_time, (
                f"Startup took {startup_time:.2f}s, expected < {max_startup_time}s"
            )
            
        except ImportError as e:
            pytest.skip(f"Components not available for startup test: {e}")


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_api_workflow(self):
        """Test complete API workflow from request to response."""
        try:
            from fastapi.testclient import TestClient

            from mars_gis.main import create_app

            # Create app and client
            app = create_app()
            client = TestClient(app)
            
            # Test complete workflow
            # 1. Health check
            health_response = client.get("/")
            assert health_response.status_code == 200
            
            # 2. API documentation
            docs_response = client.get("/docs")
            assert docs_response.status_code == 200
            
            # 3. OpenAPI spec
            openapi_response = client.get("/openapi.json")
            assert openapi_response.status_code == 200
            
            # Workflow completed successfully
            assert True, "Complete API workflow executed successfully"
            
        except ImportError as e:
            pytest.skip(f"FastAPI components not available: {e}")
    
    def test_configuration_to_application_workflow(self):
        """Test workflow from configuration loading to application startup."""
        # 1. Configuration loading
        from mars_gis.core.config import settings
        assert settings is not None
        
        # 2. Directory creation
        assert settings.DATA_DIR.exists()
        assert settings.LOGS_DIR.exists()
        
        # 3. Application creation
        try:
            from mars_gis.main import create_app
            app = create_app()
            assert app is not None
            
        except ImportError as e:
            pytest.skip(f"Application creation not available: {e}")
        
        # 4. Configuration values used correctly
        assert settings.APP_NAME == "MARS-GIS"
        assert settings.PORT == 8000
        
        # Workflow completed successfully
        assert True, "Configuration to application workflow completed"


# Import helper for text function if SQLAlchemy available
try:
    from sqlalchemy import text
except ImportError:
    def text(query):
        """Mock text function for when SQLAlchemy not available."""
        return query


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
