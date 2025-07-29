"""NASA Mars data integration module."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from mars_gis.core.config import settings


class NASADataClient:
    """Client for NASA Mars data APIs."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize NASA data client."""
        self.api_key = api_key or settings.NASA_API_KEY
        self.base_url = settings.NASA_PDS_BASE_URL
        self.session = requests.Session()
        
        if self.api_key:
            self.session.params = {"api_key": self.api_key}
    
    def get_mro_images(
        self,
        lat_min: float = -90,
        lat_max: float = 90,
        lon_min: float = -180,
        lon_max: float = 180,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get Mars Reconnaissance Orbiter (MRO) images.
        
        Args:
            lat_min, lat_max: Latitude bounds
            lon_min, lon_max: Longitude bounds
            start_date, end_date: Date range (YYYY-MM-DD format)
            limit: Maximum number of results
            
        Returns:
            List of image metadata dictionaries
        """
        try:
            params = {
                "mission": "MRO",
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "limit": limit
            }
            
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            
            # Note: This is a placeholder URL structure
            # Real NASA PDS API endpoints would need to be researched
            url = f"{self.base_url}/mro/ctx/search"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json().get("results", [])
            
        except requests.RequestException as e:
            print(f"Error fetching MRO images: {e}")
            return []
    
    def get_mola_elevation_data(
        self,
        lat_center: float,
        lon_center: float,
        radius_km: float = 50
    ) -> Dict[str, Any]:
        """
        Get MOLA (Mars Orbiter Laser Altimeter) elevation data.
        
        Args:
            lat_center: Center latitude
            lon_center: Center longitude
            radius_km: Search radius in kilometers
            
        Returns:
            Elevation data dictionary
        """
        try:
            params = {
                "lat": lat_center,
                "lon": lon_center,
                "radius": radius_km,
                "format": "json"
            }
            
            url = f"{self.base_url}/mgs/mola/elevation"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            print(f"Error fetching MOLA data: {e}")
            return {}
    
    def download_image(
        self,
        image_url: str,
        save_path: Path,
        chunk_size: int = 8192
    ) -> bool:
        """
        Download Mars image from NASA.
        
        Args:
            image_url: URL of the image to download
            save_path: Local path to save the image
            chunk_size: Download chunk size in bytes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.get(image_url, stream=True, timeout=60)
            response.raise_for_status()
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except requests.RequestException as e:
            print(f"Error downloading image: {e}")
            return False
    
    def get_mission_metadata(self, mission_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific Mars mission.
        
        Args:
            mission_name: Name of the mission (MRO, MGS, etc.)
            
        Returns:
            Mission metadata dictionary
        """
        try:
            url = f"{self.base_url}/missions/{mission_name.lower()}/metadata"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            print(f"Error fetching mission metadata: {e}")
            return {}


class MarsDataProcessor:
    """Processor for Mars geospatial data."""
    
    def __init__(self):
        """Initialize Mars data processor."""
        self.nasa_client = NASADataClient()
    
    def process_mro_ctx_image(self, image_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process MRO Context Camera (CTX) image metadata.
        
        Args:
            image_metadata: Raw image metadata from NASA
            
        Returns:
            Processed metadata for database storage
        """
        processed = {
            "filename": image_metadata.get("product_id", "unknown"),
            "mission_name": "MRO",
            "instrument": "CTX",
            "acquisition_date": self._parse_date(
                image_metadata.get("start_time")
            ),
            "center_lat": float(image_metadata.get("center_latitude", 0)),
            "center_lon": float(image_metadata.get("center_longitude", 0)),
            "resolution_meters": float(
                image_metadata.get("map_resolution", 0)
            ),
            "processing_level": image_metadata.get("processing_level", "raw"),
            "metadata_json": json.dumps(image_metadata)
        }
        
        return processed
    
    def extract_geological_features(
        self,
        image_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract geological features from image metadata.
        
        Args:
            image_data: Image metadata dictionary
            
        Returns:
            List of geological feature dictionaries
        """
        features = []
        
        # Extract crater information if available
        if "crater_data" in image_data:
            for crater in image_data["crater_data"]:
                feature = {
                    "name": crater.get("name", f"Crater_{crater.get('id')}"),
                    "feature_type": "crater",
                    "diameter_km": float(crater.get("diameter_km", 0)),
                    "center_lat": float(crater.get("latitude", 0)),
                    "center_lon": float(crater.get("longitude", 0)),
                    "age_estimate": crater.get("age_estimate", "unknown"),
                    "description": f"Crater with diameter {crater.get('diameter_km')}km"
                }
                features.append(feature)
        
        return features
    
    def _parse_date(self, date_string: Optional[str]) -> Optional[datetime]:
        """Parse NASA date string to datetime object."""
        if not date_string:
            return None
            
        try:
            # Common NASA date formats
            formats = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%j",  # Year and day of year
                "%Y-%m-%d",
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_string, fmt)
                except ValueError:
                    continue
                    
            return None
            
        except Exception:
            return None


# Convenience functions
def get_nasa_client(api_key: Optional[str] = None) -> NASADataClient:
    """Get configured NASA data client."""
    return NASADataClient(api_key)


def download_mars_sample_data(data_dir: Path) -> bool:
    """
    Download sample Mars data for testing and development.
    
    Args:
        data_dir: Directory to save sample data
        
    Returns:
        True if successful, False otherwise
    """
    client = NASADataClient()
    
    # Create sample data directory
    sample_dir = data_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get sample MRO images (limited area for testing)
        images = client.get_mro_images(
            lat_min=-10, lat_max=10,
            lon_min=-10, lon_max=10,
            limit=5
        )
        
        # Save metadata
        metadata_file = sample_dir / "sample_images.json"
        with open(metadata_file, "w") as f:
            json.dump(images, f, indent=2)
        
        print(f"Downloaded {len(images)} sample image records")
        return True
        
    except Exception as e:
        print(f"Error downloading sample data: {e}")
        return False
