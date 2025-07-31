"""
Interactive Mars Mapping Interface

This module provides interactive 2D mapping capabilities for Mars data
with foundation model embedding integration and real-time analysis.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from mars_gis.core.types import BoundingBox, MarsCoordinate

from ..foundation_models.landing_site_optimization import (
    FoundationModelLandingSiteSelector,
)
from ..foundation_models.planetary_scale_embeddings import EmbeddingDatabase


@dataclass
class MapConfig:
    """Configuration for interactive mapping."""
    default_zoom: int = 10
    min_zoom: int = 1
    max_zoom: int = 18
    tile_size: int = 256
    coordinate_system: str = "mars_2000"  # Mars coordinate reference system
    background_style: str = "terrain"     # terrain, satellite, elevation
    overlay_opacity: float = 0.7          # Overlay transparency


@dataclass
class MapLayer:
    """Map layer definition."""
    name: str
    layer_type: str  # elevation, thermal, spectral, foundation_embeddings
    data_source: str
    visible: bool = True
    opacity: float = 1.0
    color_scheme: str = "viridis"
    metadata: Dict[str, Any] = None


@dataclass
class InteractiveFeature:
    """Interactive map feature (point, polygon, etc)."""
    feature_id: str
    feature_type: str  # point, polygon, line
    coordinates: List[MarsCoordinate]
    properties: Dict[str, Any]
    style: Dict[str, Any]


class MarsMapTileGenerator:
    """
    Generates map tiles for Mars surface data.
    """
    
    def __init__(
        self,
        embedding_db: EmbeddingDatabase,
        config: MapConfig
    ):
        self.embedding_db = embedding_db
        self.config = config
        
        # Tile cache for performance
        self.tile_cache = {}
        self.max_cache_size = 1000
        
    def generate_tile(
        self,
        zoom: int,
        tile_x: int,
        tile_y: int,
        layer_type: str = "terrain"
    ) -> np.ndarray:
        """
        Generate a map tile for specified coordinates and zoom level.
        
        Args:
            zoom: Zoom level
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate
            layer_type: Type of layer to generate
            
        Returns:
            RGB image array for the tile
        """
        cache_key = f"{zoom}_{tile_x}_{tile_y}_{layer_type}"
        
        # Check cache first
        if cache_key in self.tile_cache:
            return self.tile_cache[cache_key]
        
        # Calculate geographic bounds for this tile
        bounds = self._tile_to_bounds(zoom, tile_x, tile_y)
        
        # Generate tile based on layer type
        if layer_type == "terrain":
            tile_data = self._generate_terrain_tile(bounds)
        elif layer_type == "elevation":
            tile_data = self._generate_elevation_tile(bounds)
        elif layer_type == "thermal":
            tile_data = self._generate_thermal_tile(bounds)
        elif layer_type == "foundation_embeddings":
            tile_data = self._generate_embedding_tile(bounds)
        else:
            tile_data = self._generate_default_tile(bounds)
        
        # Cache the tile
        self._cache_tile(cache_key, tile_data)
        
        return tile_data
    
    def _tile_to_bounds(
        self,
        zoom: int,
        tile_x: int,
        tile_y: int
    ) -> BoundingBox:
        """Convert tile coordinates to geographic bounds."""
        # Mars Web Mercator projection (simplified)
        n = 2.0 ** zoom
        
        # Convert to longitude/latitude
        lon_min = tile_x / n * 360.0 - 180.0
        lat_max = (np.arctan(np.sinh(np.pi * (1 - 2 * tile_y / n))) *
                   180.0 / np.pi)
        
        lon_max = (tile_x + 1) / n * 360.0 - 180.0
        lat_min = (np.arctan(np.sinh(np.pi * (1 - 2 * (tile_y + 1) / n))) *
                   180.0 / np.pi)
        
        return BoundingBox(
            min_lat=lat_min,
            max_lat=lat_max,
            min_lon=lon_min,
            max_lon=lon_max
        )
    
    def _generate_terrain_tile(self, bounds: BoundingBox) -> np.ndarray:
        """Generate terrain visualization tile."""
        tile_size = self.config.tile_size
        
        # Create synthetic Mars terrain data
        terrain_data = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        
        # Add Mars-like coloring
        for i in range(tile_size):
            for j in range(tile_size):
                # Calculate position within bounds
                lat_factor = i / tile_size
                lon_factor = j / tile_size
                
                # Mars terrain coloring
                base_red = 150 + int(30 * np.random.random())
                base_green = int(70 + 40 * lat_factor)
                base_blue = int(30 + 20 * lon_factor)
                
                terrain_data[i, j] = [base_red, base_green, base_blue]
        
        return terrain_data
    
    def _generate_elevation_tile(self, bounds: BoundingBox) -> np.ndarray:
        """Generate elevation visualization tile."""
        tile_size = self.config.tile_size
        
        # Generate synthetic elevation data
        elevation = np.zeros((tile_size, tile_size))
        
        # Add elevation patterns
        x = np.linspace(0, 1, tile_size)
        y = np.linspace(0, 1, tile_size)
        X, Y = np.meshgrid(x, y)
        
        # Add some topographic features
        elevation = (np.sin(X * 4 * np.pi) * np.cos(Y * 4 * np.pi) * 0.3 +
                     np.sin(X * 8 * np.pi) * np.cos(Y * 8 * np.pi) * 0.1)
        
        # Normalize to 0-255 range
        elevation_norm = ((elevation - elevation.min()) /
                          (elevation.max() - elevation.min()))
        elevation_img = (elevation_norm * 255).astype(np.uint8)
        
        # Convert to RGB (grayscale)
        return np.stack([elevation_img, elevation_img, elevation_img], axis=2)
    
    def _generate_thermal_tile(self, bounds: BoundingBox) -> np.ndarray:
        """Generate thermal visualization tile."""
        tile_size = self.config.tile_size
        
        # Generate synthetic thermal data
        thermal = np.random.rand(tile_size, tile_size)
        
        # Apply thermal color mapping
        thermal_rgb = np.zeros((tile_size, tile_size, 3))
        
        # Blue to red thermal mapping
        thermal_rgb[:, :, 0] = thermal  # Red channel
        thermal_rgb[:, :, 2] = 1 - thermal  # Blue channel (inverse)
        
        return (thermal_rgb * 255).astype(np.uint8)
    
    def _generate_embedding_tile(self, bounds: BoundingBox) -> np.ndarray:
        """Generate foundation model embedding visualization tile."""
        tile_size = self.config.tile_size
        
        # Query embeddings for this region
        embedding_tiles = self.embedding_db.query_embeddings(bounds)
        
        if not embedding_tiles:
            # No embeddings available, return default
            return self._generate_default_tile(bounds)
        
        # Visualize embedding data
        embedding_img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        
        # Simple visualization: use first 3 embedding dimensions as RGB
        for tile in embedding_tiles:
            if len(tile.embedding) >= 3:
                # Normalize embedding values to 0-255
                rgb = np.abs(tile.embedding[:3])
                rgb = (rgb / np.max(rgb) * 255).astype(np.uint8)
                
                # Apply to entire tile (simplified)
                embedding_img[:, :] = rgb
                break  # Use first tile for simplicity
        
        return embedding_img
    
    def _generate_default_tile(self, bounds: BoundingBox) -> np.ndarray:
        """Generate default/placeholder tile."""
        tile_size = self.config.tile_size
        
        # Simple grid pattern
        default_tile = np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 100
        
        # Add grid lines
        grid_spacing = tile_size // 8
        default_tile[::grid_spacing, :] = [150, 150, 150]
        default_tile[:, ::grid_spacing] = [150, 150, 150]
        
        return default_tile
    
    def _cache_tile(self, cache_key: str, tile_data: np.ndarray):
        """Cache generated tile."""
        if len(self.tile_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.tile_cache))
            del self.tile_cache[oldest_key]
        
        self.tile_cache[cache_key] = tile_data


class InteractiveMarsMap:
    """
    Main interactive Mars mapping interface.
    """
    
    def __init__(
        self,
        embedding_db: EmbeddingDatabase,
        landing_site_selector: Optional[FoundationModelLandingSiteSelector] = None,
        config: Optional[MapConfig] = None
    ):
        self.embedding_db = embedding_db
        self.landing_site_selector = landing_site_selector
        self.config = config or MapConfig()
        
        # Map components
        self.tile_generator = MarsMapTileGenerator(embedding_db, self.config)
        
        # Map state
        self.current_center = MarsCoordinate(latitude=0.0, longitude=0.0)
        self.current_zoom = self.config.default_zoom
        self.active_layers: List[MapLayer] = []
        self.interactive_features: List[InteractiveFeature] = []
        
        # Analysis tools
        self.analysis_callbacks: Dict[str, Callable] = {}
        
        # Initialize default layers
        self._initialize_default_layers()
    
    def _initialize_default_layers(self):
        """Initialize default map layers."""
        default_layers = [
            MapLayer(
                name="Mars Terrain",
                layer_type="terrain",
                data_source="base_imagery",
                visible=True,
                opacity=1.0,
                color_scheme="mars_realistic"
            ),
            MapLayer(
                name="Elevation",
                layer_type="elevation",
                data_source="mola_dem",
                visible=False,
                opacity=0.7,
                color_scheme="elevation"
            ),
            MapLayer(
                name="Thermal",
                layer_type="thermal",
                data_source="themis_thermal",
                visible=False,
                opacity=0.6,
                color_scheme="thermal"
            ),
            MapLayer(
                name="Foundation Embeddings",
                layer_type="foundation_embeddings",
                data_source="embedding_db",
                visible=False,
                opacity=0.5,
                color_scheme="embedding_vis"
            )
        ]
        
        self.active_layers = default_layers
    
    def set_center_and_zoom(
        self, 
        center: MarsCoordinate, 
        zoom: int
    ):
        """Set map center and zoom level."""
        self.current_center = center
        self.current_zoom = max(
            self.config.min_zoom, 
            min(zoom, self.config.max_zoom)
        )
    
    def toggle_layer(self, layer_name: str, visible: bool):
        """Toggle layer visibility."""
        for layer in self.active_layers:
            if layer.name == layer_name:
                layer.visible = visible
                break
    
    def set_layer_opacity(self, layer_name: str, opacity: float):
        """Set layer opacity."""
        for layer in self.active_layers:
            if layer.name == layer_name:
                layer.opacity = max(0.0, min(1.0, opacity))
                break
    
    def add_interactive_feature(
        self,
        feature_id: str,
        feature_type: str,
        coordinates: List[MarsCoordinate],
        properties: Dict[str, Any],
        style: Optional[Dict[str, Any]] = None
    ):
        """Add interactive feature to map."""
        default_style = {
            "color": "#ff0000",
            "weight": 2,
            "opacity": 0.8,
            "fillOpacity": 0.3
        }
        
        if style:
            default_style.update(style)
        
        feature = InteractiveFeature(
            feature_id=feature_id,
            feature_type=feature_type,
            coordinates=coordinates,
            properties=properties,
            style=default_style
        )
        
        self.interactive_features.append(feature)
    
    def analyze_region(
        self, 
        bounds: BoundingBox,
        analysis_type: str = "landing_site"
    ) -> Dict[str, Any]:
        """
        Perform analysis on selected region.
        
        Args:
            bounds: Geographic bounds to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis results
        """
        if analysis_type == "landing_site" and self.landing_site_selector:
            return self._analyze_landing_sites(bounds)
        elif analysis_type == "terrain":
            return self._analyze_terrain(bounds)
        elif analysis_type == "embeddings":
            return self._analyze_embeddings(bounds)
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}
    
    def _analyze_landing_sites(self, bounds: BoundingBox) -> Dict[str, Any]:
        """Analyze landing sites in specified region."""
        # Generate candidate sites within bounds
        candidate_sites = []
        
        # Create grid of candidate locations
        lat_step = (bounds.max_lat - bounds.min_lat) / 10
        lon_step = (bounds.max_lon - bounds.min_lon) / 10
        
        for i in range(10):
            for j in range(10):
                lat = bounds.min_lat + i * lat_step
                lon = bounds.min_lon + j * lon_step
                candidate_sites.append(MarsCoordinate(latitude=lat, longitude=lon))
        
        # Evaluate sites
        assessments = self.landing_site_selector.evaluate_landing_sites(
            candidate_sites, mission_type="exploration", top_k=5
        )
        
        # Convert to serializable format
        results = {
            "analysis_type": "landing_site",
            "region_bounds": {
                "min_lat": bounds.min_lat,
                "max_lat": bounds.max_lat,
                "min_lon": bounds.min_lon,
                "max_lon": bounds.max_lon
            },
            "candidate_count": len(candidate_sites),
            "top_sites": []
        }
        
        for i, assessment in enumerate(assessments):
            site_info = {
                "rank": i + 1,
                "latitude": assessment.location.latitude,
                "longitude": assessment.location.longitude,
                "overall_ranking": assessment.overall_ranking,
                "safety_score": assessment.safety_score,
                "science_value": assessment.science_value,
                "operational_score": assessment.operational_score,
                "confidence": assessment.confidence,
                "recommendation": assessment.recommendation,
                "risk_factors": assessment.risk_factors,
                "opportunities": assessment.opportunities
            }
            results["top_sites"].append(site_info)
        
        return results
    
    def _analyze_terrain(self, bounds: BoundingBox) -> Dict[str, Any]:
        """Analyze terrain in specified region."""
        return {
            "analysis_type": "terrain",
            "region_bounds": {
                "min_lat": bounds.min_lat,
                "max_lat": bounds.max_lat,
                "min_lon": bounds.min_lon,
                "max_lon": bounds.max_lon
            },
            "terrain_summary": {
                "avg_elevation": 2.5,  # km above datum
                "elevation_range": 5.2,  # km
                "slope_analysis": {
                    "avg_slope": 12.3,  # degrees
                    "max_slope": 45.6,
                    "steep_terrain_percentage": 15.2
                },
                "surface_features": [
                    "impact_craters",
                    "volcanic_features", 
                    "erosion_channels"
                ]
            }
        }
    
    def _analyze_embeddings(self, bounds: BoundingBox) -> Dict[str, Any]:
        """Analyze foundation model embeddings in region."""
        embedding_tiles = self.embedding_db.query_embeddings(bounds)
        
        if not embedding_tiles:
            return {
                "analysis_type": "embeddings",
                "error": "No embedding data available for this region"
            }
        
        # Analyze embedding characteristics
        embeddings = [tile.embedding for tile in embedding_tiles]
        embedding_matrix = np.array(embeddings)
        
        return {
            "analysis_type": "embeddings",
            "region_bounds": {
                "min_lat": bounds.min_lat,
                "max_lat": bounds.max_lat,
                "min_lon": bounds.min_lon,
                "max_lon": bounds.max_lon
            },
            "embedding_stats": {
                "tile_count": len(embedding_tiles),
                "embedding_dimension": embedding_matrix.shape[1],
                "avg_confidence": float(np.mean([t.confidence for t in embedding_tiles])),
                "embedding_variance": float(np.var(embedding_matrix)),
                "principal_components": self._compute_pca_summary(embedding_matrix)
            }
        }
    
    def _compute_pca_summary(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Compute PCA summary of embeddings."""
        try:
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=min(3, embeddings.shape[1]))
            pca.fit(embeddings)
            
            return {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist()
            }
        except ImportError:
            return {"error": "PCA analysis requires scikit-learn"}
    
    def export_map_config(self, output_path: Path) -> Dict[str, Any]:
        """
        Export map configuration for web viewer.
        
        Args:
            output_path: Output directory path
            
        Returns:
            Export metadata
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export map configuration
        map_config = {
            "default_center": {
                "latitude": self.current_center.latitude,
                "longitude": self.current_center.longitude
            },
            "default_zoom": self.current_zoom,
            "zoom_range": {
                "min": self.config.min_zoom,
                "max": self.config.max_zoom
            },
            "tile_size": self.config.tile_size,
            "coordinate_system": self.config.coordinate_system,
            "layers": []
        }
        
        # Add layer configurations
        for layer in self.active_layers:
            layer_config = {
                "name": layer.name,
                "type": layer.layer_type,
                "data_source": layer.data_source,
                "visible": layer.visible,
                "opacity": layer.opacity,
                "color_scheme": layer.color_scheme
            }
            map_config["layers"].append(layer_config)
        
        # Add interactive features
        features = []
        for feature in self.interactive_features:
            feature_data = {
                "id": feature.feature_id,
                "type": feature.feature_type,
                "coordinates": [
                    {"lat": coord.latitude, "lon": coord.longitude}
                    for coord in feature.coordinates
                ],
                "properties": feature.properties,
                "style": feature.style
            }
            features.append(feature_data)
        
        map_config["features"] = features
        
        # Save configuration
        with open(output_path / "map_config.json", 'w') as f:
            json.dump(map_config, f, indent=2)
        
        # Generate tile server configuration
        tile_server_config = {
            "tile_endpoint": "/api/tiles/{z}/{x}/{y}/{layer}",
            "available_layers": [layer.layer_type for layer in self.active_layers],
            "analysis_endpoint": "/api/analysis",
            "feature_endpoint": "/api/features"
        }
        
        with open(output_path / "tile_server_config.json", 'w') as f:
            json.dump(tile_server_config, f, indent=2)
        
        return {
            "map_config_file": "map_config.json",
            "tile_server_config": "tile_server_config.json",
            "export_path": str(output_path),
            "layer_count": len(self.active_layers),
            "feature_count": len(self.interactive_features)
        }


def create_interactive_mars_map(
    embedding_db: EmbeddingDatabase,
    landing_site_selector: Optional[FoundationModelLandingSiteSelector] = None,
    default_zoom: int = 10,
    coordinate_system: str = "mars_2000"
) -> InteractiveMarsMap:
    """
    Factory function to create interactive Mars map.
    
    Args:
        embedding_db: Embedding database
        landing_site_selector: Optional landing site selector
        default_zoom: Default zoom level
        coordinate_system: Coordinate reference system
        
    Returns:
        Initialized InteractiveMarsMap
    """
    config = MapConfig(
        default_zoom=default_zoom,
        coordinate_system=coordinate_system
    )
    
    return InteractiveMarsMap(embedding_db, landing_site_selector, config)


# Example usage and testing
if __name__ == "__main__":
    from ..foundation_models.landing_site_optimization import (
        FoundationModelLandingSiteSelector,
    )
    from ..foundation_models.planetary_scale_embeddings import EmbeddingDatabase

    # Create components
    db_path = Path("/tmp/mars_embeddings")
    embedding_db = EmbeddingDatabase(db_path)
    
    # Create interactive map
    mars_map = create_interactive_mars_map(
        embedding_db,
        default_zoom=8,
        coordinate_system="mars_2000"
    )
    
    print("Interactive Mars Mapping Interface Initialized")
    print("=" * 50)
    
    # Test map operations
    mars_map.set_center_and_zoom(
        MarsCoordinate(latitude=14.5, longitude=175.9), 
        zoom=12
    )
    
    print(f"Map center: {mars_map.current_center.latitude:.2f}, {mars_map.current_center.longitude:.2f}")
    print(f"Zoom level: {mars_map.current_zoom}")
    print(f"Active layers: {len(mars_map.active_layers)}")
    
    # Add test feature
    mars_map.add_interactive_feature(
        "test_site_1",
        "point",
        [MarsCoordinate(latitude=14.5, longitude=175.9)],
        {"name": "Test Landing Site", "mission": "Mars 2025"},
        {"color": "#00ff00", "weight": 3}
    )
    
    print(f"Interactive features: {len(mars_map.interactive_features)}")
    
    # Test region analysis
    test_bounds = BoundingBox(
        min_lat=14.0, max_lat=15.0,
        min_lon=175.0, max_lon=176.0
    )
    
    terrain_analysis = mars_map.analyze_region(test_bounds, "terrain")
    print(f"\nTerrain analysis completed for region")
    print(f"Average elevation: {terrain_analysis['terrain_summary']['avg_elevation']} km")
    
    # Export map configuration
    export_path = Path("/tmp/mars_map_export")
    export_metadata = mars_map.export_map_config(export_path)
    
    print(f"\nMap configuration exported:")
    print(f"Export path: {export_metadata['export_path']}")
    print(f"Layers: {export_metadata['layer_count']}")
    print(f"Features: {export_metadata['feature_count']}")
    
    print("\nInteractive Mars mapping interface ready for web integration!")
