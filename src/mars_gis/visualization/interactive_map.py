"""Interactive mapping components for Mars GIS web interface."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MarsInteractiveMap:
    """Interactive map component for Mars exploration and analysis."""
    
    def __init__(self, map_id: str = "mars_map"):
        """
        Initialize interactive map.
        
        Args:
            map_id: Unique identifier for the map instance
        """
        self.map_id = map_id
        self.layers = {}
        self.controls = {}
        self.event_handlers = {}
        self.current_view = {
            "center": [0.0, 0.0],  # longitude, latitude
            "zoom": 2,
            "projection": "equirectangular"
        }
        
        # Map configuration
        self.config = {
            "coordinate_system": "MARS_2000",
            "base_radius_km": 3389.5,
            "zoom_levels": 20,
            "tile_size": 256,
            "max_bounds": [[-180, -90], [180, 90]],
            "attribution": "Mars data courtesy of NASA/JPL-Caltech"
        }
        
        # Available map projections
        self.projections = {
            "equirectangular": {
                "name": "Equirectangular",
                "description": "Simple cylindrical projection",
                "suitable_for": "global_overview"
            },
            "orthographic": {
                "name": "Orthographic",
                "description": "Globe-like projection",
                "suitable_for": "hemisphere_view"
            },
            "mercator": {
                "name": "Web Mercator",
                "description": "Standard web mapping projection",
                "suitable_for": "detailed_navigation"
            },
            "polar_stereographic": {
                "name": "Polar Stereographic",
                "description": "Optimized for polar regions",
                "suitable_for": "polar_exploration"
            }
        }
    
    def create_base_map_config(
        self,
        initial_center: Tuple[float, float] = (0.0, 0.0),
        initial_zoom: int = 2,
        projection: str = "equirectangular"
    ) -> Dict[str, Any]:
        """
        Create base map configuration.
        
        Args:
            initial_center: Initial map center (lon, lat)
            initial_zoom: Initial zoom level
            projection: Map projection type
            
        Returns:
            Base map configuration
        """
        base_config = {
            "map_id": self.map_id,
            "container_id": f"{self.map_id}_container",
            "initial_view": {
                "center": list(initial_center),
                "zoom": initial_zoom,
                "projection": projection
            },
            "constraints": {
                "min_zoom": 0,
                "max_zoom": self.config["zoom_levels"],
                "max_bounds": self.config["max_bounds"],
                "zoom_snap": 0.1
            },
            "interaction": {
                "dragging": True,
                "touch_zoom": True,
                "scroll_wheel_zoom": True,
                "double_click_zoom": True,
                "box_zoom": True,
                "keyboard": True
            },
            "animation": {
                "zoom_animation": True,
                "fade_animation": True,
                "marker_zoom_animation": True,
                "transform3d": True
            },
            "coordinate_system": self.config["coordinate_system"],
            "attribution": self.config["attribution"]
        }
        
        return base_config
    
    def add_base_layer(
        self,
        layer_type: str = "satellite",
        source_url: Optional[str] = None,
        layer_name: str = "base_layer"
    ) -> Dict[str, Any]:
        """
        Add base map layer (satellite imagery, terrain, etc.).
        
        Args:
            layer_type: Type of base layer
            source_url: URL template for tile source
            layer_name: Name for the layer
            
        Returns:
            Layer configuration
        """
        if source_url is None:
            # Default Mars imagery sources
            source_urls = {
                "satellite": "https://api.mars.nasa.gov/tiles/satellite/{z}/{x}/{y}.jpg",
                "terrain": "https://api.mars.nasa.gov/tiles/terrain/{z}/{x}/{y}.jpg",
                "elevation": "https://api.mars.nasa.gov/tiles/elevation/{z}/{x}/{y}.png",
                "thermal": "https://api.mars.nasa.gov/tiles/thermal/{z}/{x}/{y}.jpg"
            }
            source_url = source_urls.get(layer_type, source_urls["satellite"])
        
        layer_config = {
            "id": layer_name,
            "type": "tile_layer",
            "subtype": layer_type,
            "source": {
                "type": "xyz",
                "url": source_url,
                "tile_size": self.config["tile_size"],
                "max_zoom": self.config["zoom_levels"],
                "attribution": self.config["attribution"],
                "crossOrigin": "anonymous"
            },
            "style": {
                "opacity": 1.0,
                "blend_mode": "normal",
                "brightness": 1.0,
                "contrast": 1.0,
                "saturation": 1.0,
                "hue_rotate": 0
            },
            "properties": {
                "visible": True,
                "interactive": False,
                "z_index": 0
            },
            "metadata": {
                "layer_type": layer_type,
                "data_source": "NASA/JPL-Caltech",
                "created_at": self._get_timestamp()
            }
        }
        
        self.layers[layer_name] = layer_config
        return layer_config
    
    def add_terrain_layer(
        self,
        terrain_data: Dict[str, Any],
        layer_name: str = "terrain_overlay",
        style: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add interactive terrain layer from 3D terrain data.
        
        Args:
            terrain_data: 3D terrain data
            layer_name: Name for the layer
            style: Optional styling configuration
            
        Returns:
            Terrain layer configuration
        """
        if style is None:
            style = {
                "fill_color": "elevation_colormap",
                "fill_opacity": 0.7,
                "stroke_color": "#333333",
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        
        # Extract bounds and features from terrain data
        bounds = terrain_data.get("bounds", [-180, -90, 180, 90])
        elevation_stats = terrain_data.get("statistics", {})
        
        layer_config = {
            "id": layer_name,
            "type": "geojson_layer",
            "subtype": "terrain",
            "source": {
                "type": "geojson",
                "data": self._terrain_to_geojson(terrain_data),
                "cluster": False,
                "buffer": 64,
                "max_zoom": 14
            },
            "style": style,
            "properties": {
                "visible": True,
                "interactive": True,
                "z_index": 10,
                "hover_enabled": True,
                "popup_enabled": True
            },
            "bounds": bounds,
            "elevation_stats": elevation_stats,
            "metadata": {
                "layer_type": "terrain",
                "resolution": terrain_data.get("resolution", 100),
                "created_at": self._get_timestamp()
            }
        }
        
        self.layers[layer_name] = layer_config
        return layer_config
    
    def add_mission_layer(
        self,
        mission_data: Dict[str, Any],
        layer_name: str = "mission_data",
        mission_type: str = "rover"
    ) -> Dict[str, Any]:
        """
        Add mission data layer (rover paths, landing sites, etc.).
        
        Args:
            mission_data: Mission data
            layer_name: Name for the layer
            mission_type: Type of mission data
            
        Returns:
            Mission layer configuration
        """
        # Convert mission data to GeoJSON features
        geojson_data = self._mission_to_geojson(mission_data, mission_type)
        
        # Style based on mission type
        style = self._get_mission_style(mission_type)
        
        layer_config = {
            "id": layer_name,
            "type": "geojson_layer",
            "subtype": "mission",
            "source": {
                "type": "geojson",
                "data": geojson_data,
                "cluster": mission_type == "science_data",
                "cluster_max_zoom": 10,
                "cluster_radius": 50
            },
            "style": style,
            "properties": {
                "visible": True,
                "interactive": True,
                "z_index": 20,
                "hover_enabled": True,
                "popup_enabled": True,
                "animation_enabled": mission_type in ["rover", "aircraft"]
            },
            "mission_info": {
                "type": mission_type,
                "name": mission_data.get("name", "Unknown Mission"),
                "status": mission_data.get("status", "active"),
                "start_date": mission_data.get("start_date"),
                "end_date": mission_data.get("end_date")
            },
            "metadata": {
                "layer_type": "mission",
                "mission_type": mission_type,
                "created_at": self._get_timestamp()
            }
        }
        
        self.layers[layer_name] = layer_config
        return layer_config
    
    def add_analysis_layer(
        self,
        analysis_results: Dict[str, Any],
        layer_name: str = "analysis_overlay",
        analysis_type: str = "terrain_analysis"
    ) -> Dict[str, Any]:
        """
        Add analysis results as interactive layer.
        
        Args:
            analysis_results: Analysis results data
            layer_name: Name for the layer
            analysis_type: Type of analysis
            
        Returns:
            Analysis layer configuration
        """
        # Convert analysis results to map features
        if analysis_type == "terrain_analysis":
            features = self._terrain_analysis_to_features(analysis_results)
        elif analysis_type == "hazard_detection":
            features = self._hazard_analysis_to_features(analysis_results)
        elif analysis_type == "path_planning":
            features = self._path_analysis_to_features(analysis_results)
        else:
            features = []
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        
        style = self._get_analysis_style(analysis_type)
        
        layer_config = {
            "id": layer_name,
            "type": "geojson_layer",
            "subtype": "analysis",
            "source": {
                "type": "geojson",
                "data": geojson_data,
                "cluster": False
            },
            "style": style,
            "properties": {
                "visible": True,
                "interactive": True,
                "z_index": 15,
                "hover_enabled": True,
                "popup_enabled": True,
                "temporal_animation": analysis_type == "temporal_analysis"
            },
            "analysis_info": {
                "type": analysis_type,
                "timestamp": analysis_results.get("timestamp"),
                "confidence": analysis_results.get("confidence", 1.0)
            },
            "metadata": {
                "layer_type": "analysis",
                "analysis_type": analysis_type,
                "created_at": self._get_timestamp()
            }
        }
        
        self.layers[layer_name] = layer_config
        return layer_config
    
    def create_map_controls(
        self,
        control_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create interactive map controls.
        
        Args:
            control_types: List of control types to include
            
        Returns:
            Map controls configuration
        """
        if control_types is None:
            control_types = [
                "zoom", "layer_switcher", "coordinate_display",
                "measurement", "search", "fullscreen"
            ]
        
        controls_config = {}
        
        for control_type in control_types:
            if control_type == "zoom":
                controls_config["zoom"] = {
                    "position": "top-left",
                    "zoom_in_text": "+",
                    "zoom_out_text": "-",
                    "zoom_in_title": "Zoom in",
                    "zoom_out_title": "Zoom out"
                }
            
            elif control_type == "layer_switcher":
                controls_config["layer_switcher"] = {
                    "position": "top-right",
                    "title": "Layers",
                    "collapsed": True,
                    "auto_z_index": True,
                    "sort_layers": True,
                    "group_check_boxes": True
                }
            
            elif control_type == "coordinate_display":
                controls_config["coordinate_display"] = {
                    "position": "bottom-left",
                    "format": "DD",  # Decimal degrees
                    "precision": 6,
                    "show_elevation": True,
                    "coordinate_system": self.config["coordinate_system"]
                }
            
            elif control_type == "measurement":
                controls_config["measurement"] = {
                    "position": "top-left",
                    "primary_length_unit": "kilometers",
                    "primary_area_unit": "sqkilometers",
                    "active_color": "#ff6b35",
                    "completed_color": "#369936"
                }
            
            elif control_type == "search":
                controls_config["search"] = {
                    "position": "top-left",
                    "placeholder": "Search Mars locations...",
                    "max_results": 10,
                    "search_sources": ["landmarks", "craters", "missions"]
                }
            
            elif control_type == "fullscreen":
                controls_config["fullscreen"] = {
                    "position": "top-left",
                    "title": "Toggle fullscreen"
                }
            
            elif control_type == "minimap":
                controls_config["minimap"] = {
                    "position": "bottom-right",
                    "width": 150,
                    "height": 150,
                    "toggle_display": True,
                    "zoom_level_offset": -5
                }
        
        self.controls = controls_config
        return controls_config
    
    def create_interaction_handlers(self) -> Dict[str, Any]:
        """Create event handlers for map interactions."""
        handlers = {
            "click": {
                "enabled": True,
                "actions": ["show_popup", "highlight_feature"],
                "propagate": True
            },
            "hover": {
                "enabled": True,
                "actions": ["show_tooltip", "highlight_feature"],
                "debounce_ms": 100
            },
            "zoom": {
                "enabled": True,
                "actions": ["update_layer_visibility", "refresh_data"],
                "zoom_levels": {
                    "global": [0, 3],
                    "regional": [4, 8],
                    "local": [9, 15],
                    "detailed": [16, 20]
                }
            },
            "drag": {
                "enabled": True,
                "actions": ["update_bounds", "lazy_load_data"]
            },
            "selection": {
                "enabled": True,
                "selection_modes": ["point", "rectangle", "polygon"],
                "multi_select": True
            }
        }
        
        self.event_handlers = handlers
        return handlers
    
    def create_popup_templates(self) -> Dict[str, str]:
        """Create popup templates for different feature types."""
        templates = {
            "terrain": """
                <div class="mars-popup terrain-popup">
                    <h3>Terrain Information</h3>
                    <p><strong>Elevation:</strong> {{elevation}} m</p>
                    <p><strong>Slope:</strong> {{slope}}°</p>
                    <p><strong>Terrain Type:</strong> {{terrain_type}}</p>
                    <p><strong>Coordinates:</strong> {{coordinates}}</p>
                </div>
            """,
            "mission": """
                <div class="mars-popup mission-popup">
                    <h3>{{mission_name}}</h3>
                    <p><strong>Type:</strong> {{mission_type}}</p>
                    <p><strong>Status:</strong> {{status}}</p>
                    <p><strong>Date:</strong> {{date}}</p>
                    <div class="mission-details">{{details}}</div>
                </div>
            """,
            "analysis": """
                <div class="mars-popup analysis-popup">
                    <h3>Analysis Results</h3>
                    <p><strong>Type:</strong> {{analysis_type}}</p>
                    <p><strong>Confidence:</strong> {{confidence}}%</p>
                    <p><strong>Timestamp:</strong> {{timestamp}}</p>
                    <div class="results">{{results}}</div>
                </div>
            """,
            "landing_site": """
                <div class="mars-popup landing-site-popup">
                    <h3>{{site_name}}</h3>
                    <p><strong>Mission:</strong> {{mission}}</p>
                    <p><strong>Landing Date:</strong> {{landing_date}}</p>
                    <p><strong>Coordinates:</strong> {{coordinates}}</p>
                    <p><strong>Elevation:</strong> {{elevation}} m</p>
                    <p><strong>Status:</strong> {{status}}</p>
                </div>
            """
        }
        
        return templates
    
    def generate_complete_map_config(
        self,
        include_base_layers: bool = True,
        include_controls: bool = True,
        include_interactions: bool = True
    ) -> Dict[str, Any]:
        """Generate complete map configuration for frontend."""
        config = self.create_base_map_config()
        
        # Add layers
        if include_base_layers and not self.layers:
            self.add_base_layer("satellite", layer_name="mars_satellite")
            self.add_base_layer("terrain", layer_name="mars_terrain")
        
        config["layers"] = list(self.layers.values())
        
        # Add controls
        if include_controls:
            config["controls"] = self.create_map_controls()
        
        # Add interactions
        if include_interactions:
            config["interactions"] = self.create_interaction_handlers()
        
        # Add popup templates
        config["popup_templates"] = self.create_popup_templates()
        
        # Add theme configuration
        config["theme"] = {
            "name": "mars_explorer",
            "colors": {
                "primary": "#ff6b35",
                "secondary": "#004e89",
                "background": "#1a1a1a",
                "text": "#ffffff",
                "accent": "#ffc857"
            },
            "fonts": {
                "primary": "Roboto, sans-serif",
                "monospace": "Roboto Mono, monospace"
            }
        }
        
        return config
    
    def _terrain_to_geojson(self, terrain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert terrain data to GeoJSON format."""
        # Simplified conversion - would need proper implementation
        bounds = terrain_data.get("bounds", [-180, -90, 180, 90])
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [bounds[0], bounds[1]],
                    [bounds[2], bounds[1]],
                    [bounds[2], bounds[3]],
                    [bounds[0], bounds[3]],
                    [bounds[0], bounds[1]]
                ]]
            },
            "properties": {
                "type": "terrain",
                "statistics": terrain_data.get("statistics", {}),
                "resolution": terrain_data.get("resolution", 100)
            }
        }
        
        return {
            "type": "FeatureCollection",
            "features": [feature]
        }
    
    def _mission_to_geojson(
        self,
        mission_data: Dict[str, Any],
        mission_type: str
    ) -> Dict[str, Any]:
        """Convert mission data to GeoJSON format."""
        features = []
        
        if mission_type == "rover" and "path" in mission_data:
            # Rover path as LineString
            path_coords = mission_data["path"]
            if len(path_coords) > 1:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": path_coords
                    },
                    "properties": {
                        "type": "rover_path",
                        "mission_name": mission_data.get("name", "Unknown"),
                        "distance": mission_data.get("path_length", 0)
                    }
                })
        
        elif mission_type == "landing_site" and "coordinates" in mission_data:
            # Landing site as Point
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": mission_data["coordinates"]
                },
                "properties": {
                    "type": "landing_site",
                    "name": mission_data.get("name", "Landing Site"),
                    "mission": mission_data.get("mission", "Unknown"),
                    "date": mission_data.get("date")
                }
            })
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def _get_mission_style(self, mission_type: str) -> Dict[str, Any]:
        """Get styling for mission data based on type."""
        styles = {
            "rover": {
                "stroke_color": "#00ff00",
                "stroke_width": 3,
                "stroke_opacity": 0.8,
                "marker_color": "#00ff00",
                "marker_size": 8
            },
            "landing_site": {
                "marker_color": "#ffff00",
                "marker_size": 12,
                "marker_symbol": "rocket",
                "marker_opacity": 1.0
            },
            "aircraft": {
                "stroke_color": "#0080ff",
                "stroke_width": 2,
                "stroke_opacity": 0.7,
                "marker_color": "#0080ff",
                "marker_size": 6
            }
        }
        
        return styles.get(mission_type, styles["rover"])
    
    def _terrain_analysis_to_features(
        self,
        analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert terrain analysis results to GeoJSON features."""
        features = []
        
        # Convert terrain features
        terrain_features = analysis_results.get("terrain_features", [])
        for feature in terrain_features:
            if "position" in feature:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": feature["position"][:2]  # lon, lat
                    },
                    "properties": {
                        "type": "terrain_feature",
                        "feature_type": feature.get("type", "unknown"),
                        "elevation": feature.get("elevation", 0),
                        "analysis_type": "terrain"
                    }
                })
        
        return features
    
    def _hazard_analysis_to_features(
        self,
        analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert hazard analysis results to GeoJSON features."""
        features = []
        
        hazard_zones = analysis_results.get("hazard_zones", [])
        for zone in hazard_zones:
            if "center" in zone:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": zone["center"][:2]
                    },
                    "properties": {
                        "type": "hazard_zone",
                        "hazard_type": zone.get("type", "unknown"),
                        "radius": zone.get("radius", 100),
                        "severity": zone.get("severity", "medium"),
                        "analysis_type": "hazard"
                    }
                })
        
        return features
    
    def _path_analysis_to_features(
        self,
        analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert path analysis results to GeoJSON features."""
        features = []
        
        if "path" in analysis_results:
            path_coords = analysis_results["path"]
            if len(path_coords) > 1:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": path_coords
                    },
                    "properties": {
                        "type": "planned_path",
                        "algorithm": analysis_results.get("algorithm", "unknown"),
                        "distance": analysis_results.get("path_length", 0),
                        "difficulty": analysis_results.get("difficulty", "unknown"),
                        "analysis_type": "path_planning"
                    }
                })
        
        return features
    
    def _get_analysis_style(self, analysis_type: str) -> Dict[str, Any]:
        """Get styling for analysis results based on type."""
        styles = {
            "terrain_analysis": {
                "marker_color": "#ff8c00",
                "marker_size": 6,
                "marker_opacity": 0.8
            },
            "hazard_detection": {
                "marker_color": "#ff0000",
                "marker_size": 8,
                "marker_opacity": 0.9,
                "fill_color": "#ff0000",
                "fill_opacity": 0.2
            },
            "path_planning": {
                "stroke_color": "#00ffff",
                "stroke_width": 4,
                "stroke_opacity": 0.8,
                "stroke_dasharray": "10,5"
            }
        }
        
        return styles.get(analysis_type, styles["terrain_analysis"])
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import time
        return time.strftime("%Y-%m-%d %H:%M:%S")
    
    def export_map_config(
        self,
        output_path: Path,
        format: str = "json"
    ) -> bool:
        """Export map configuration to file."""
        try:
            config = self.generate_complete_map_config()
            
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(config, f, indent=2)
                return True
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Map config export failed: {e}")
            return False
