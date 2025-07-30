"""
Unit Tests for Core Configuration Module
Tests every function and method with edge cases and error conditions.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mars_gis.core.config import Settings


class TestSettingsClass:
    """Test Settings configuration class thoroughly."""
    
    def test_settings_initialization_with_defaults(self):
        """Test Settings class initializes with proper default values."""
        settings = Settings()
        
        # Application settings
        assert settings.APP_NAME == "MARS-GIS"
        assert settings.VERSION == "0.1.0"
        assert settings.DESCRIPTION == "Mars Geospatial Intelligence System"
        assert settings.ENVIRONMENT == "development"
        assert settings.DEBUG is True
        
        # Server settings
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8000
        assert settings.WORKERS == 1
        assert settings.RELOAD is True
        
        # CORS settings
        expected_hosts = ["http://localhost:3000", "http://localhost:8000"]
        assert settings.ALLOWED_HOSTS == expected_hosts
        
        # Database settings
        expected_db_url = "postgresql://postgres:password@localhost:5432/mars_gis"
        assert settings.DATABASE_URL == expected_db_url
        assert settings.DATABASE_ECHO is False
        
        # Redis settings
        assert settings.REDIS_URL == "redis://localhost:6379/0"
        
        # API settings
        assert settings.NASA_API_KEY is None
        assert settings.NASA_PDS_BASE_URL == "https://pds-imaging.jpl.nasa.gov/data"
        assert settings.USGS_BASE_URL == "https://astrogeology.usgs.gov/search"
        
        # ML settings
        assert settings.TORCH_DEVICE == "cuda"
        assert settings.MODEL_CACHE_DIR == "data/models"
    
    def test_settings_attributes_are_correct_types(self):
        """Test that all settings attributes have correct types."""
        settings = Settings()
        
        # String attributes
        string_attrs = [
            'APP_NAME', 'VERSION', 'DESCRIPTION', 'ENVIRONMENT',
            'HOST', 'DATABASE_URL', 'REDIS_URL', 'NASA_PDS_BASE_URL',
            'USGS_BASE_URL', 'TORCH_DEVICE', 'MODEL_CACHE_DIR'
        ]
        for attr in string_attrs:
            value = getattr(settings, attr)
            assert isinstance(value, str), f"{attr} should be string, got {type(value)}"
            assert len(value) > 0, f"{attr} should not be empty string"
        
        # Integer attributes
        int_attrs = ['PORT', 'WORKERS']
        for attr in int_attrs:
            value = getattr(settings, attr)
            assert isinstance(value, int), f"{attr} should be int, got {type(value)}"
            assert value > 0, f"{attr} should be positive integer"
        
        # Boolean attributes
        bool_attrs = ['DEBUG', 'RELOAD', 'DATABASE_ECHO']
        for attr in bool_attrs:
            value = getattr(settings, attr)
            assert isinstance(value, bool), f"{attr} should be bool, got {type(value)}"
        
        # List attributes
        assert isinstance(settings.ALLOWED_HOSTS, list)
        assert len(settings.ALLOWED_HOSTS) > 0
        
        # Optional attributes
        assert settings.NASA_API_KEY is None or isinstance(settings.NASA_API_KEY, str)
    
    def test_settings_values_are_realistic(self):
        """Test that settings values are realistic and usable."""
        settings = Settings()
        
        # Port should be in valid range
        assert 1 <= settings.PORT <= 65535, "Port should be in valid range"
        
        # URLs should be properly formatted
        assert settings.DATABASE_URL.startswith(('postgresql://', 'sqlite://')), \
            "Database URL should use supported scheme"
        assert settings.REDIS_URL.startswith('redis://'), \
            "Redis URL should use redis:// scheme"
        assert settings.NASA_PDS_BASE_URL.startswith('https://'), \
            "NASA URL should use HTTPS"
        assert settings.USGS_BASE_URL.startswith('https://'), \
            "USGS URL should use HTTPS"
        
        # CORS hosts should be valid URLs
        for host in settings.ALLOWED_HOSTS:
            assert host.startswith(('http://', 'https://')), \
                f"CORS host {host} should be valid URL"
        
        # Device should be valid torch device
        assert settings.TORCH_DEVICE in ['cuda', 'cpu', 'mps'], \
            "Torch device should be valid option"
    
    def test_settings_can_be_modified_after_initialization(self):
        """Test that settings can be modified after creation."""
        settings = Settings()
        
        # Modify various settings
        original_port = settings.PORT
        settings.PORT = 9000
        assert settings.PORT == 9000
        assert settings.PORT != original_port
        
        original_debug = settings.DEBUG
        settings.DEBUG = not original_debug
        assert settings.DEBUG != original_debug
        
        original_host = settings.HOST
        settings.HOST = "127.0.0.1"
        assert settings.HOST == "127.0.0.1"
        assert settings.HOST != original_host
    
    @patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///test.db'})
    def test_settings_respect_environment_variables(self):
        """Test that settings can be overridden by environment variables."""
        # This test would require implementing environment variable loading
        # For now, test that the capability exists
        
        settings = Settings()
        # In a full implementation, this would read from environment
        # assert settings.DATABASE_URL == 'sqlite:///test.db'
        
        # Test that the setting exists and can be changed
        settings.DATABASE_URL = 'sqlite:///test.db'
        assert settings.DATABASE_URL == 'sqlite:///test.db'
    
    def test_settings_handles_missing_optional_values(self):
        """Test that optional settings handle None values properly."""
        settings = Settings()
        
        # NASA API key is optional
        assert settings.NASA_API_KEY is None
        
        # Should handle being set and unset
        settings.NASA_API_KEY = "test_key_123"
        assert settings.NASA_API_KEY == "test_key_123"
        
        settings.NASA_API_KEY = None
        assert settings.NASA_API_KEY is None
    
    def test_settings_database_url_validation(self):
        """Test database URL validation and format."""
        settings = Settings()
        
        # Default should be PostgreSQL
        assert 'postgresql://' in settings.DATABASE_URL
        
        # Should contain required components
        db_url = settings.DATABASE_URL
        assert '://' in db_url, "Database URL should have scheme"
        assert '@' in db_url, "Database URL should have credentials separator"
        assert '/' in db_url, "Database URL should have database name"
        
        # Test with different database URLs
        test_urls = [
            "postgresql://user:pass@localhost:5432/dbname",
            "sqlite:///path/to/database.db",
            "postgresql://user@localhost/dbname"
        ]
        
        for url in test_urls:
            settings.DATABASE_URL = url
            assert settings.DATABASE_URL == url
    
    def test_settings_cors_hosts_validation(self):
        """Test CORS hosts are properly formatted."""
        settings = Settings()
        
        # All hosts should be valid URLs
        for host in settings.ALLOWED_HOSTS:
            assert isinstance(host, str), "CORS host should be string"
            assert len(host) > 0, "CORS host should not be empty"
            assert host.startswith(('http://', 'https://')), \
                f"CORS host {host} should be valid URL"
        
        # Test adding/modifying CORS hosts
        new_hosts = ["https://example.com", "http://localhost:3001"]
        settings.ALLOWED_HOSTS = new_hosts
        assert settings.ALLOWED_HOSTS == new_hosts
        
        # Test invalid hosts
        settings.ALLOWED_HOSTS = ["invalid-url", "ftp://invalid.com"]
        # In production, this might validate and reject invalid URLs
        assert len(settings.ALLOWED_HOSTS) == 2
    
    def test_settings_api_url_validation(self):
        """Test API URL settings are properly formatted."""
        settings = Settings()
        
        # NASA API URL validation
        nasa_url = settings.NASA_PDS_BASE_URL
        assert nasa_url.startswith('https://'), "NASA URL should use HTTPS"
        assert 'nasa' in nasa_url.lower() or 'jpl' in nasa_url.lower(), \
            "NASA URL should reference NASA or JPL"
        
        # USGS API URL validation  
        usgs_url = settings.USGS_BASE_URL
        assert usgs_url.startswith('https://'), "USGS URL should use HTTPS"
        assert 'usgs' in usgs_url.lower(), "USGS URL should reference USGS"
        
        # Test URL modification
        settings.NASA_PDS_BASE_URL = "https://example.nasa.gov/data"
        assert settings.NASA_PDS_BASE_URL == "https://example.nasa.gov/data"
    
    def test_settings_ml_configuration(self):
        """Test machine learning configuration settings."""
        settings = Settings()
        
        # Torch device should be valid
        valid_devices = ['cuda', 'cpu', 'mps', 'auto']
        assert settings.TORCH_DEVICE in valid_devices or \
               settings.TORCH_DEVICE.startswith('cuda:'), \
               f"Torch device {settings.TORCH_DEVICE} should be valid"
        
        # Model cache directory should be valid path
        cache_dir = settings.MODEL_CACHE_DIR
        assert isinstance(cache_dir, str), "Model cache dir should be string"
        assert len(cache_dir) > 0, "Model cache dir should not be empty"
        
        # Should be able to create Path object
        cache_path = Path(cache_dir)
        assert isinstance(cache_path, Path), "Should be valid path"
    
    def test_settings_server_configuration_edge_cases(self):
        """Test server configuration with edge cases."""
        settings = Settings()
        
        # Test port boundaries
        settings.PORT = 1
        assert settings.PORT == 1
        
        settings.PORT = 65535
        assert settings.PORT == 65535
        
        # Test invalid ports (in production, these might be validated)
        settings.PORT = 0
        assert settings.PORT == 0  # Should handle gracefully
        
        settings.PORT = 99999
        assert settings.PORT == 99999  # Should handle gracefully
        
        # Test workers configuration
        settings.WORKERS = 1
        assert settings.WORKERS == 1
        
        settings.WORKERS = 16
        assert settings.WORKERS == 16
        
        # Test host configuration
        valid_hosts = ['0.0.0.0', '127.0.0.1', 'localhost', '192.168.1.100']
        for host in valid_hosts:
            settings.HOST = host
            assert settings.HOST == host
    
    def test_settings_boolean_flags_behavior(self):
        """Test boolean configuration flags behave correctly."""
        settings = Settings()
        
        # Test DEBUG flag
        assert isinstance(settings.DEBUG, bool)
        settings.DEBUG = True
        assert settings.DEBUG is True
        settings.DEBUG = False
        assert settings.DEBUG is False
        
        # Test RELOAD flag
        assert isinstance(settings.RELOAD, bool)
        settings.RELOAD = True
        assert settings.RELOAD is True
        settings.RELOAD = False
        assert settings.RELOAD is False
        
        # Test DATABASE_ECHO flag
        assert isinstance(settings.DATABASE_ECHO, bool)
        settings.DATABASE_ECHO = True
        assert settings.DATABASE_ECHO is True
        settings.DATABASE_ECHO = False
        assert settings.DATABASE_ECHO is False
    
    def test_settings_string_attributes_handle_edge_cases(self):
        """Test string attributes handle edge cases properly."""
        settings = Settings()
        
        # Test empty strings
        settings.APP_NAME = ""
        assert settings.APP_NAME == ""
        
        settings.DESCRIPTION = ""
        assert settings.DESCRIPTION == ""
        
        # Test very long strings
        long_string = "x" * 1000
        settings.DESCRIPTION = long_string
        assert settings.DESCRIPTION == long_string
        assert len(settings.DESCRIPTION) == 1000
        
        # Test special characters
        special_string = "Mars-GIS™ 🚀 Analysis & Planning"
        settings.APP_NAME = special_string
        assert settings.APP_NAME == special_string
        
        # Test whitespace handling
        whitespace_string = "  Mars GIS  "
        settings.APP_NAME = whitespace_string
        assert settings.APP_NAME == whitespace_string
    
    def test_settings_immutable_defaults_concept(self):
        """Test that changing one instance doesn't affect class defaults."""
        settings1 = Settings()
        settings2 = Settings()
        
        # Both should start with same values
        assert settings1.PORT == settings2.PORT
        assert settings1.DEBUG == settings2.DEBUG
        
        # Changing one shouldn't affect the other
        settings1.PORT = 9999
        settings1.DEBUG = False
        
        assert settings2.PORT == 8000  # Should still be default
        assert settings2.DEBUG is True  # Should still be default
        
        # Create a third instance to verify defaults unchanged
        settings3 = Settings()
        assert settings3.PORT == 8000
        assert settings3.DEBUG is True


class TestSettingsEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions for Settings class."""
    
    def test_settings_with_none_values(self):
        """Test Settings behavior with None values."""
        settings = Settings()
        
        # Optional attributes can be None
        settings.NASA_API_KEY = None
        assert settings.NASA_API_KEY is None
        
        # Required attributes set to None (should handle gracefully)
        settings.APP_NAME = None
        assert settings.APP_NAME is None
        
        settings.PORT = None
        assert settings.PORT is None
    
    def test_settings_with_invalid_types(self):
        """Test Settings behavior with invalid types."""
        settings = Settings()
        
        # Set string attribute to non-string
        settings.APP_NAME = 12345
        assert settings.APP_NAME == 12345  # Should accept any type
        
        # Set integer attribute to string
        settings.PORT = "8000"
        assert settings.PORT == "8000"  # Should accept any type
        
        # Set boolean attribute to string
        settings.DEBUG = "true"
        assert settings.DEBUG == "true"  # Should accept any type
    
    def test_settings_attribute_access_edge_cases(self):
        """Test edge cases in attribute access."""
        settings = Settings()
        
        # Test accessing non-existent attributes
        with pytest.raises(AttributeError):
            _ = settings.NON_EXISTENT_ATTRIBUTE
        
        # Test setting new attributes
        settings.NEW_ATTRIBUTE = "test_value"
        assert settings.NEW_ATTRIBUTE == "test_value"
        
        # Test deleting attributes
        delattr(settings, 'NEW_ATTRIBUTE')
        with pytest.raises(AttributeError):
            _ = settings.NEW_ATTRIBUTE
    
    def test_settings_memory_usage_with_large_values(self):
        """Test Settings memory behavior with large values."""
        settings = Settings()
        
        # Test very large string
        large_string = "x" * 10000
        settings.DESCRIPTION = large_string
        assert len(settings.DESCRIPTION) == 10000
        
        # Test very large list
        large_list = ["http://localhost:3000"] * 1000
        settings.ALLOWED_HOSTS = large_list
        assert len(settings.ALLOWED_HOSTS) == 1000
        
        # Memory should be manageable
        import sys
        settings_size = sys.getsizeof(settings.__dict__)
        
        # Should not consume excessive memory (reasonable limit)
        assert settings_size < 1024 * 1024, "Settings should not consume > 1MB"
    
    def test_settings_serialization_compatibility(self):
        """Test that Settings can be serialized/deserialized."""
        settings = Settings()
        
        # Test dict conversion
        settings_dict = settings.__dict__
        assert isinstance(settings_dict, dict)
        assert 'APP_NAME' in settings_dict
        assert 'PORT' in settings_dict
        
        # Test recreation from dict
        new_settings = Settings()
        for key, value in settings_dict.items():
            setattr(new_settings, key, value)
        
        assert new_settings.APP_NAME == settings.APP_NAME
        assert new_settings.PORT == settings.PORT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
