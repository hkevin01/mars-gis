"""USGS Mars data integration module."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from mars_gis.core.config import settings


class USGSDataClient:
    """Client for USGS Astrogeology Mars data."""
    
    def __init__(self):
        """Initialize USGS data client."""
        self.base_url = settings.USGS_BASE_URL
        self.session = requests.Session()
        
        # Set up session headers
        self.session.headers.update({
            "User-Agent": "MARS-GIS/0.1.0",
            "Accept": "application/json"
        })
    
    def search_mars_products(
        self,
        target: str = "Mars",
        instrument: Optional[str] = None,
        bbox: Optional[List[float]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for Mars data products.
        
        Args:
            target: Target body (Mars)
            instrument: Specific instrument filter
            bbox: Bounding box [west, south, east, north] in degrees
            limit: Maximum number of results
            
        Returns:
            List of product metadata dictionaries
        """
        try:
            params = {
                "target": target,
                "limit": limit,
                "format": "json"
            }
            
            if instrument:
                params["instrument"] = instrument
            
            if bbox and len(bbox) == 4:
                params["bbox"] = ",".join(map(str, bbox))
            
            response = self.session.get(
                f"{self.base_url}/api/search",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", [])
            
        except requests.RequestException as e:
            print(f"Error searching USGS products: {e}")
            return []
    
    def get_geological_map(
        self,
        region: str,
        scale: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get geological map data for a Mars region.
        
        Args:
            region: Name of the Mars region
            scale: Map scale (e.g., "1:5000000")
            
        Returns:
            Geological map metadata
        """
        try:
            params = {
                "target": "Mars",
                "product_type": "geological_map",
                "region": region
            }
            
            if scale:
                params["scale"] = scale
            
            response = self.session.get(
                f"{self.base_url}/api/geological-maps",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            print(f"Error fetching geological map: {e}")
            return {}
    
    def get_mineral_composition(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10
    ) -> Dict[str, Any]:
        """
        Get mineral composition data for a Mars location.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            radius_km: Search radius in kilometers
            
        Returns:
            Mineral composition data
        """
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "radius": radius_km,
                "data_type": "mineralogy"
            }
            
            response = self.session.get(
                f"{self.base_url}/api/point-query",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            print(f"Error fetching mineral composition: {e}")
            return {}
    
    def download_product(
        self,
        product_id: str,
        save_path: Path,
        file_format: str = "tiff"
    ) -> bool:
        """
        Download a USGS Mars data product.
        
        Args:
            product_id: USGS product identifier
            save_path: Local path to save the file
            file_format: Desired file format
            
        Returns:
            True if successful, False otherwise
        """
        try:
            params = {
                "product_id": product_id,
                "format": file_format
            }
            
            response = self.session.get(
                f"{self.base_url}/api/download",
                params=params,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except requests.RequestException as e:
            print(f"Error downloading product: {e}")
            return False


class MarsGeologyProcessor:
    """Processor for Mars geological data."""
    
    def __init__(self):
        """Initialize geology processor."""
        self.usgs_client = USGSDataClient()
    
    def process_geological_units(
        self,
        map_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process geological unit data from USGS maps.
        
        Args:
            map_data: Raw geological map data
            
        Returns:
            List of processed geological units
        """
        units = []
        
        if "geological_units" in map_data:
            for unit in map_data["geological_units"]:
                processed_unit = {
                    "name": unit.get("unit_name", "Unknown"),
                    "feature_type": "geological_unit",
                    "age_estimate": unit.get("age", "unknown"),
                    "rock_type": unit.get("lithology", "unknown"),
                    "description": unit.get("description", ""),
                    "formation_process": unit.get("formation", ""),
                    "metadata_json": json.dumps(unit)
                }
                
                # Extract geometry if available
                if "geometry" in unit:
                    processed_unit["geometry_wkt"] = unit["geometry"]
                
                units.append(processed_unit)
        
        return units
    
    def extract_mineral_data(
        self,
        composition_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract and normalize mineral composition data.
        
        Args:
            composition_data: Raw mineral composition data
            
        Returns:
            Normalized mineral data dictionary
        """
        minerals = {}
        
        if "minerals" in composition_data:
            for mineral, abundance in composition_data["minerals"].items():
                minerals[mineral.lower()] = {
                    "abundance_percent": float(abundance),
                    "confidence": composition_data.get(
                        f"{mineral}_confidence", 0.8
                    ),
                    "detection_method": composition_data.get(
                        "method", "spectroscopy"
                    )
                }
        
        return {
            "location": {
                "lat": composition_data.get("latitude", 0),
                "lon": composition_data.get("longitude", 0)
            },
            "minerals": minerals,
            "total_minerals": len(minerals),
            "analysis_date": composition_data.get("date"),
            "data_quality": composition_data.get("quality_score", 0.8)
        }
    
    def identify_landing_hazards(
        self,
        geological_data: Dict[str, Any],
        safety_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Identify potential landing hazards from geological data.
        
        Args:
            geological_data: Geological survey data
            safety_threshold: Minimum safety score (0-1)
            
        Returns:
            List of identified hazards
        """
        hazards = []
        
        # Check for steep slopes
        if "slope_data" in geological_data:
            for slope_point in geological_data["slope_data"]:
                if slope_point.get("slope_degrees", 0) > 15:
                    hazards.append({
                        "hazard_type": "steep_slope",
                        "severity": "high" if slope_point["slope_degrees"] > 25 else "medium",
                        "location": {
                            "lat": slope_point.get("lat", 0),
                            "lon": slope_point.get("lon", 0)
                        },
                        "slope_degrees": slope_point["slope_degrees"],
                        "safety_impact": 1.0 - min(slope_point["slope_degrees"] / 45, 1.0)
                    })
        
        # Check for boulder fields
        if "boulder_density" in geological_data:
            density = geological_data["boulder_density"]
            if density > 0.1:  # boulders per square meter
                hazards.append({
                    "hazard_type": "boulder_field",
                    "severity": "high" if density > 0.5 else "medium",
                    "boulder_density": density,
                    "safety_impact": 1.0 - min(density / 1.0, 0.9)
                })
        
        # Filter hazards by safety threshold
        return [h for h in hazards if h["safety_impact"] >= safety_threshold]


# Convenience functions
def get_usgs_client() -> USGSDataClient:
    """Get configured USGS data client."""
    return USGSDataClient()


def download_geological_sample_data(data_dir: Path) -> bool:
    """
    Download sample geological data for development.
    
    Args:
        data_dir: Directory to save sample data
        
    Returns:
        True if successful, False otherwise
    """
    client = USGSDataClient()
    
    sample_dir = data_dir / "geological_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Search for sample geological products
        products = client.search_mars_products(
            target="Mars",
            instrument="geological_survey",
            limit=10
        )
        
        # Save sample data
        sample_file = sample_dir / "geological_products.json"
        with open(sample_file, "w") as f:
            json.dump(products, f, indent=2)
        
        print(f"Downloaded {len(products)} geological product records")
        return True
        
    except Exception as e:
        print(f"Error downloading geological sample data: {e}")
        return False
