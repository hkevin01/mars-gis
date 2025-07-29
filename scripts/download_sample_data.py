"""Download sample Mars data for development and testing."""

import json
from typing import Any, Dict

from mars_gis.core.config import settings
from mars_gis.data.nasa_client import get_nasa_client
from mars_gis.data.usgs_client import get_usgs_client


def create_sample_data() -> Dict[str, Any]:
    """Create sample Mars data for development when APIs are not available."""
    
    sample_mro_images = [
        {
            "product_id": "CTX_001234_1234_XI_56N123W",
            "mission_name": "MRO",
            "instrument": "CTX",
            "start_time": "2020-03-15T14:30:00",
            "center_latitude": -14.5,
            "center_longitude": 175.3,
            "map_resolution": 6.0,
            "processing_level": "calibrated",
            "image_width": 5056,
            "image_height": 52224,
            "file_size_mb": 145.2,
            "quality_flag": "good"
        },
        {
            "product_id": "CTX_005678_5678_XI_23S045E",
            "mission_name": "MRO",
            "instrument": "CTX",
            "start_time": "2021-07-22T09:15:30",
            "center_latitude": 22.8,
            "center_longitude": -45.7,
            "map_resolution": 6.0,
            "processing_level": "calibrated",
            "image_width": 5056,
            "image_height": 48392,
            "file_size_mb": 132.8,
            "quality_flag": "excellent"
        }
    ]
    
    sample_geological_features = [
        {
            "name": "Gale Crater",
            "feature_type": "crater",
            "center_latitude": -5.4,
            "center_longitude": 137.8,
            "diameter_km": 154.0,
            "age_estimate": "Noachian-Hesperian",
            "description": (
                "Large impact crater, landing site of Curiosity rover"
            ),
            "discovered_by": "Mariner 9",
            "discovery_year": 1972
        },
        {
            "name": "Olympus Mons",
            "feature_type": "volcano",
            "center_latitude": 18.65,
            "center_longitude": -133.8,
            "diameter_km": 624.0,
            "elevation_km": 21.9,
            "age_estimate": "Amazonian",
            "description": "Largest volcano in the solar system",
            "discovered_by": "Mariner 9",
            "discovery_year": 1971
        },
        {
            "name": "Valles Marineris",
            "feature_type": "canyon",
            "center_latitude": -14.0,
            "center_longitude": -59.0,
            "length_km": 4000.0,
            "width_km": 200.0,
            "depth_km": 7.0,
            "age_estimate": "Hesperian",
            "description": (
                "Massive canyon system spanning 20% of Mars circumference"
            ),
            "discovered_by": "Mariner 9",
            "discovery_year": 1971
        }
    ]
    
    sample_mineral_data = [
        {
            "location": {"latitude": -5.4, "longitude": 137.8},
            "minerals": {
                "olivine": {"abundance_percent": 15.2, "confidence": 0.85},
                "pyroxene": {"abundance_percent": 22.7, "confidence": 0.92},
                "plagioclase": {"abundance_percent": 31.1, "confidence": 0.78},
                "hematite": {"abundance_percent": 8.3, "confidence": 0.89},
                "clay_minerals": {
                    "abundance_percent": 12.4,
                    "confidence": 0.73
                }
            },
            "analysis_method": "CRISM_spectroscopy",
            "data_quality": 0.84,
            "acquisition_date": "2021-04-12"
        }
    ]
    
    return {
        "mro_images": sample_mro_images,
        "geological_features": sample_geological_features,
        "mineral_composition": sample_mineral_data,
        "metadata": {
            "created_date": "2024-07-29",
            "data_type": "sample_development_data",
            "version": "1.0",
            "description": "Sample Mars data for MARS-GIS development"
        }
    }


def main():
    """Download and create sample Mars data."""
    print("Downloading sample Mars data for development...")
    
    # Create sample data directory
    data_dir = settings.DATA_DIR
    sample_dir = data_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to get real data from APIs (will likely fail without proper setup)
    nasa_client = get_nasa_client()
    usgs_client = get_usgs_client()
    
    real_data_available = False
    
    try:
        # Try to fetch small amount of real data
        mro_images = nasa_client.get_mro_images(
            lat_min=-10, lat_max=10,
            lon_min=-10, lon_max=10,
            limit=2
        )
        
        geological_products = usgs_client.search_mars_products(
            target="Mars",
            limit=2
        )
        
        if mro_images or geological_products:
            real_data_available = True
            
            # Save real data
            real_data = {
                "mro_images": mro_images,
                "geological_products": geological_products,
                "metadata": {
                    "created_date": "2024-07-29",
                    "data_type": "real_api_data",
                    "source": "NASA_PDS_USGS"
                }
            }
            
            real_data_file = sample_dir / "real_data_sample.json"
            with open(real_data_file, "w") as f:
                json.dump(real_data, f, indent=2)
            
            print(f"✓ Downloaded real data: {len(mro_images)} MRO images, "
                  f"{len(geological_products)} geological products")
    
    except Exception as e:
        print(f"Could not fetch real data (expected): {e}")
    
    # Always create sample data for development
    sample_data = create_sample_data()
    
    sample_data_file = sample_dir / "development_sample.json"
    with open(sample_data_file, "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"✓ Created sample development data with:")
    print(f"  - {len(sample_data['mro_images'])} sample MRO images")
    print(f"  - {len(sample_data['geological_features'])} geological features")
    print(f"  - {len(sample_data['mineral_composition'])} mineral analysis points")
    
    # Create summary file
    summary = {
        "sample_data_available": True,
        "real_data_available": real_data_available,
        "sample_files": [
            "development_sample.json",
            "real_data_sample.json" if real_data_available else None
        ],
        "total_records": {
            "mro_images": len(sample_data['mro_images']),
            "geological_features": len(sample_data['geological_features']),
            "mineral_analyses": len(sample_data['mineral_composition'])
        }
    }
    
    summary_file = sample_dir / "data_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🚀 Sample data setup complete!")
    print(f"Data saved to: {sample_dir}")
    print(f"Files created: {len([f for f in summary['sample_files'] if f])}")


if __name__ == "__main__":
    main()
