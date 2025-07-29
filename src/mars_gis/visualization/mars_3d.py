"""3D visualization utilities for Mars terrain and data."""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class Mars3DVisualizer:
    """3D visualization engine for Mars terrain and mission data."""
    
    def __init__(self, globe_radius: float = 3389.5):
        """
        Initialize 3D visualizer.
        
        Args:
            globe_radius: Mars radius in kilometers
        """
        self.globe_radius = globe_radius  # Mars radius in km
        self.terrain_layers = {}
        self.mission_overlays = {}
        self.camera_positions = {}
        
        # Mars coordinate system
        self.coordinate_system = "MARS_2000"
        
        # Default material properties
        self.default_materials = {
            "terrain": {
                "color": [0.8, 0.4, 0.2],  # Reddish Mars color
                "roughness": 0.9,
                "metallic": 0.1
            },
            "ice": {
                "color": [0.9, 0.95, 1.0],  # Bluish white
                "roughness": 0.3,
                "metallic": 0.0
            },
            "rock": {
                "color": [0.6, 0.5, 0.4],  # Rocky brown
                "roughness": 0.95,
                "metallic": 0.0
            }
        }
    
    def create_mars_globe(
        self,
        resolution: int = 512,
        texture_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create 3D Mars globe geometry.
        
        Args:
            resolution: Sphere resolution (vertices per dimension)
            texture_path: Optional path to Mars surface texture
            
        Returns:
            Globe geometry and material data
        """
        try:
            if not NUMPY_AVAILABLE:
                logger.error("NumPy required for globe generation")
                return {}
            
            # Generate sphere geometry
            vertices, faces, uvs = self._generate_sphere_geometry(resolution)
            
            # Calculate normals
            normals = self._calculate_normals(vertices, faces)
            
            globe_data = {
                "type": "mars_globe",
                "radius_km": self.globe_radius,
                "resolution": resolution,
                "geometry": {
                    "vertices": vertices.tolist() if NUMPY_AVAILABLE else [],
                    "faces": faces.tolist() if NUMPY_AVAILABLE else [],
                    "normals": normals.tolist() if NUMPY_AVAILABLE else [],
                    "uvs": uvs.tolist() if NUMPY_AVAILABLE else [],
                    "vertex_count": len(vertices) if NUMPY_AVAILABLE else 0,
                    "face_count": len(faces) if NUMPY_AVAILABLE else 0
                },
                "material": {
                    "type": "mars_surface",
                    "base_color": self.default_materials["terrain"]["color"],
                    "roughness": self.default_materials["terrain"]["roughness"],
                    "metallic": self.default_materials["terrain"]["metallic"],
                    "texture_path": texture_path,
                    "normal_mapping": True,
                    "displacement_mapping": False
                },
                "coordinate_system": self.coordinate_system,
                "created_at": self._get_timestamp()
            }
            
            return globe_data
            
        except Exception as e:
            logger.error(f"Globe creation failed: {e}")
            return {}
    
    def create_terrain_layer(
        self,
        terrain_data: Dict[str, Any],
        layer_name: str = "terrain",
        height_scale: float = 0.001
    ) -> Dict[str, Any]:
        """
        Create 3D terrain layer from elevation data.
        
        Args:
            terrain_data: Terrain data from reconstruction
            layer_name: Name for the terrain layer
            height_scale: Scale factor for elevation (km)
            
        Returns:
            Terrain layer data for 3D rendering
        """
        try:
            if not terrain_data.get("elevation_grid"):
                return {"error": "No elevation data provided"}
            
            if not NUMPY_AVAILABLE:
                return {"error": "NumPy required for terrain layer"}
            
            elevation_grid = np.array(terrain_data["elevation_grid"])
            coordinates = terrain_data.get("coordinates", {})
            
            if not coordinates:
                return {"error": "No coordinate data provided"}
            
            # Convert to 3D coordinates on Mars sphere
            vertices_3d = self._elevation_to_3d(
                elevation_grid, coordinates, height_scale
            )
            
            # Generate mesh connectivity
            faces = self._generate_grid_faces(elevation_grid.shape)
            
            # Calculate vertex colors based on elevation
            vertex_colors = self._calculate_elevation_colors(elevation_grid)
            
            # Calculate normals
            normals = self._calculate_terrain_normals(vertices_3d, faces)
            
            # Detect and classify terrain features
            features = self._classify_terrain_features(elevation_grid, coordinates)
            
            layer_data = {
                "type": "terrain_layer",
                "name": layer_name,
                "bounds": terrain_data.get("bounds", []),
                "height_scale": height_scale,
                "geometry": {
                    "vertices": vertices_3d.tolist() if NUMPY_AVAILABLE else [],
                    "faces": faces.tolist() if NUMPY_AVAILABLE else [],
                    "normals": normals.tolist() if NUMPY_AVAILABLE else [],
                    "colors": vertex_colors.tolist() if NUMPY_AVAILABLE else [],
                    "vertex_count": len(vertices_3d) if NUMPY_AVAILABLE else 0
                },
                "elevation_stats": terrain_data.get("statistics", {}),
                "terrain_features": features,
                "rendering_hints": {
                    "wireframe": False,
                    "smooth_shading": True,
                    "transparency": 1.0,
                    "z_offset": 0.0
                },
                "created_at": self._get_timestamp()
            }
            
            self.terrain_layers[layer_name] = layer_data
            return layer_data
            
        except Exception as e:
            logger.error(f"Terrain layer creation failed: {e}")
            return {"error": str(e)}
    
    def create_mission_overlay(
        self,
        mission_data: Dict[str, Any],
        overlay_type: str = "landing_sites"
    ) -> Dict[str, Any]:
        """
        Create mission overlay (landing sites, rover paths, etc.).
        
        Args:
            mission_data: Mission-specific data
            overlay_type: Type of overlay
            
        Returns:
            Mission overlay data
        """
        try:
            overlay_data = {
                "type": overlay_type,
                "mission_name": mission_data.get("name", "Unknown"),
                "elements": [],
                "metadata": mission_data.get("metadata", {}),
                "created_at": self._get_timestamp()
            }
            
            if overlay_type == "landing_sites":
                overlay_data["elements"] = self._create_landing_site_markers(
                    mission_data
                )
            elif overlay_type == "rover_paths":
                overlay_data["elements"] = self._create_rover_path_lines(
                    mission_data
                )
            elif overlay_type == "scientific_data":
                overlay_data["elements"] = self._create_science_markers(
                    mission_data
                )
            elif overlay_type == "hazard_zones":
                overlay_data["elements"] = self._create_hazard_zones(
                    mission_data
                )
            
            self.mission_overlays[f"{overlay_type}_{mission_data.get('name', 'unknown')}"] = overlay_data
            return overlay_data
            
        except Exception as e:
            logger.error(f"Mission overlay creation failed: {e}")
            return {"error": str(e)}
    
    def create_interactive_camera(
        self,
        initial_position: Tuple[float, float, float] = (0, 0, 10000),
        target: Tuple[float, float, float] = (0, 0, 0),
        camera_type: str = "perspective"
    ) -> Dict[str, Any]:
        """
        Create interactive camera configuration.
        
        Args:
            initial_position: Camera position in Mars coordinates
            target: Look-at target
            camera_type: Camera type (perspective/orthographic)
            
        Returns:
            Camera configuration
        """
        camera_config = {
            "type": camera_type,
            "position": list(initial_position),
            "target": list(target),
            "up": [0, 0, 1],  # Mars Z-axis up
            "fov": 60.0 if camera_type == "perspective" else None,
            "near": 0.1,
            "far": 20000.0,
            "controls": {
                "enabled": True,
                "rotate_speed": 1.0,
                "zoom_speed": 1.0,
                "pan_speed": 0.8,
                "auto_rotate": False,
                "enable_zoom": True,
                "enable_rotate": True,
                "enable_pan": True,
                "min_distance": 100.0,
                "max_distance": 15000.0,
                "min_polar_angle": 0,
                "max_polar_angle": math.pi
            },
            "animation": {
                "enabled": False,
                "duration": 2000,
                "easing": "ease-in-out"
            }
        }
        
        return camera_config
    
    def generate_scene_config(
        self,
        include_globe: bool = True,
        terrain_layers: List[str] = None,
        mission_overlays: List[str] = None,
        lighting_preset: str = "mars_day"
    ) -> Dict[str, Any]:
        """
        Generate complete 3D scene configuration.
        
        Args:
            include_globe: Whether to include Mars globe
            terrain_layers: List of terrain layer names to include
            mission_overlays: List of mission overlay names to include
            lighting_preset: Lighting configuration preset
            
        Returns:
            Complete scene configuration
        """
        try:
            scene_config = {
                "scene_type": "mars_3d",
                "coordinate_system": self.coordinate_system,
                "objects": [],
                "lighting": self._get_lighting_config(lighting_preset),
                "camera": self.create_interactive_camera(),
                "background": {
                    "type": "starfield",
                    "color": [0.05, 0.05, 0.1],
                    "stars": True
                },
                "post_processing": {
                    "enabled": True,
                    "effects": ["antialiasing", "ambient_occlusion"]
                },
                "metadata": {
                    "created_at": self._get_timestamp(),
                    "mars_radius_km": self.globe_radius
                }
            }
            
            # Add Mars globe
            if include_globe:
                globe_data = self.create_mars_globe()
                if globe_data:
                    scene_config["objects"].append({
                        "id": "mars_globe",
                        "type": "globe",
                        "data": globe_data,
                        "visible": True,
                        "opacity": 1.0
                    })
            
            # Add terrain layers
            if terrain_layers:
                for layer_name in terrain_layers:
                    if layer_name in self.terrain_layers:
                        scene_config["objects"].append({
                            "id": f"terrain_{layer_name}",
                            "type": "terrain",
                            "data": self.terrain_layers[layer_name],
                            "visible": True,
                            "opacity": 1.0
                        })
            
            # Add mission overlays
            if mission_overlays:
                for overlay_name in mission_overlays:
                    if overlay_name in self.mission_overlays:
                        scene_config["objects"].append({
                            "id": f"overlay_{overlay_name}",
                            "type": "overlay",
                            "data": self.mission_overlays[overlay_name],
                            "visible": True,
                            "opacity": 0.8
                        })
            
            return scene_config
            
        except Exception as e:
            logger.error(f"Scene configuration failed: {e}")
            return {}
    
    def _generate_sphere_geometry(
        self,
        resolution: int
    ) -> Tuple[Any, Any, Any]:
        """Generate sphere vertices, faces, and UV coordinates."""
        if not NUMPY_AVAILABLE:
            return [], [], []
        
        vertices = []
        uvs = []
        
        # Generate vertices and UVs
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                # Spherical coordinates
                theta = i * math.pi / resolution  # 0 to pi
                phi = j * 2 * math.pi / resolution  # 0 to 2pi
                
                # Convert to Cartesian coordinates
                x = self.globe_radius * math.sin(theta) * math.cos(phi)
                y = self.globe_radius * math.sin(theta) * math.sin(phi)
                z = self.globe_radius * math.cos(theta)
                
                vertices.append([x, y, z])
                
                # UV coordinates
                u = j / resolution
                v = i / resolution
                uvs.append([u, v])
        
        vertices = np.array(vertices)
        uvs = np.array(uvs)
        
        # Generate faces
        faces = []
        for i in range(resolution):
            for j in range(resolution):
                # Quad indices
                a = i * (resolution + 1) + j
                b = a + 1
                c = (i + 1) * (resolution + 1) + j
                d = c + 1
                
                # Two triangles per quad
                faces.append([a, b, c])
                faces.append([b, d, c])
        
        faces = np.array(faces)
        
        return vertices, faces, uvs
    
    def _calculate_normals(
        self,
        vertices: Any,
        faces: Any
    ) -> Any:
        """Calculate vertex normals for smooth shading."""
        if not NUMPY_AVAILABLE:
            return []
        
        normals = np.zeros_like(vertices)
        
        # Calculate face normals and accumulate
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            face_normal = face_normal / np.linalg.norm(face_normal)
            
            # Accumulate to vertex normals
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal
        
        # Normalize vertex normals
        for i in range(len(normals)):
            norm = np.linalg.norm(normals[i])
            if norm > 0:
                normals[i] /= norm
        
        return normals
    
    def _elevation_to_3d(
        self,
        elevation_grid: Any,
        coordinates: Dict[str, List[float]],
        height_scale: float
    ) -> Any:
        """Convert elevation grid to 3D coordinates on Mars surface."""
        if not NUMPY_AVAILABLE:
            return []
        
        lon_coords = np.array(coordinates["longitude"])
        lat_coords = np.array(coordinates["latitude"])
        
        vertices_3d = []
        
        for i, lat in enumerate(lat_coords):
            for j, lon in enumerate(lon_coords):
                if i < elevation_grid.shape[0] and j < elevation_grid.shape[1]:
                    elevation = elevation_grid[i, j]
                    
                    # Convert to radians
                    lat_rad = math.radians(lat)
                    lon_rad = math.radians(lon)
                    
                    # Radius including elevation
                    radius = self.globe_radius + elevation * height_scale
                    
                    # Convert to Cartesian coordinates
                    x = radius * math.cos(lat_rad) * math.cos(lon_rad)
                    y = radius * math.cos(lat_rad) * math.sin(lon_rad)
                    z = radius * math.sin(lat_rad)
                    
                    vertices_3d.append([x, y, z])
        
        return np.array(vertices_3d)
    
    def _generate_grid_faces(self, grid_shape: Tuple[int, int]) -> Any:
        """Generate face connectivity for grid-based mesh."""
        if not NUMPY_AVAILABLE:
            return []
        
        rows, cols = grid_shape
        faces = []
        
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Grid vertex indices
                v0 = i * cols + j
                v1 = i * cols + (j + 1)
                v2 = (i + 1) * cols + j
                v3 = (i + 1) * cols + (j + 1)
                
                # Two triangles per grid cell
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        return np.array(faces)
    
    def _calculate_elevation_colors(self, elevation_grid: Any) -> Any:
        """Calculate vertex colors based on elevation."""
        if not NUMPY_AVAILABLE:
            return []
        
        # Normalize elevations to [0, 1]
        min_elev = np.min(elevation_grid)
        max_elev = np.max(elevation_grid)
        
        if max_elev > min_elev:
            normalized = (elevation_grid - min_elev) / (max_elev - min_elev)
        else:
            normalized = np.zeros_like(elevation_grid)
        
        colors = []
        
        for i in range(elevation_grid.shape[0]):
            for j in range(elevation_grid.shape[1]):
                elev_norm = normalized[i, j]
                
                # Color mapping: blue (low) -> red (high)
                if elev_norm < 0.33:
                    # Blue to green
                    r = 0.2
                    g = 0.3 + elev_norm * 0.4
                    b = 0.8 - elev_norm * 0.6
                elif elev_norm < 0.66:
                    # Green to yellow
                    r = 0.2 + (elev_norm - 0.33) * 1.5
                    g = 0.7
                    b = 0.2
                else:
                    # Yellow to red
                    r = 0.9
                    g = 0.7 - (elev_norm - 0.66) * 0.6
                    b = 0.1
                
                colors.append([r, g, b, 1.0])  # RGBA
        
        return np.array(colors)
    
    def _calculate_terrain_normals(self, vertices: Any, faces: Any) -> Any:
        """Calculate normals for terrain mesh."""
        return self._calculate_normals(vertices, faces)
    
    def _classify_terrain_features(
        self,
        elevation_grid: Any,
        coordinates: Dict[str, List[float]]
    ) -> List[Dict[str, Any]]:
        """Classify terrain features for visualization."""
        features = []
        
        if not NUMPY_AVAILABLE:
            return features
        
        try:
            # Simple feature detection
            grad_y, grad_x = np.gradient(elevation_grid)
            slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
            
            # Find steep areas (potential cliffs)
            steep_mask = slope > 30
            steep_points = np.where(steep_mask)
            
            for i, j in zip(steep_points[0], steep_points[1]):
                if i < len(coordinates["latitude"]) and j < len(coordinates["longitude"]):
                    features.append({
                        "type": "cliff",
                        "position": [
                            coordinates["longitude"][j],
                            coordinates["latitude"][i],
                            float(elevation_grid[i, j])
                        ],
                        "slope": float(slope[i, j])
                    })
            
            # Limit to reasonable number of features
            features = features[:100]
            
        except Exception as e:
            logger.error(f"Feature classification error: {e}")
        
        return features
    
    def _create_landing_site_markers(
        self,
        mission_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create landing site marker elements."""
        markers = []
        
        landing_sites = mission_data.get("landing_sites", [])
        
        for site in landing_sites:
            markers.append({
                "type": "marker",
                "subtype": "landing_site",
                "position": site.get("coordinates", [0, 0, 0]),
                "label": site.get("name", "Landing Site"),
                "style": {
                    "color": [1.0, 0.8, 0.0],  # Golden
                    "size": 5.0,
                    "shape": "diamond",
                    "glow": True
                },
                "metadata": site.get("metadata", {})
            })
        
        return markers
    
    def _create_rover_path_lines(
        self,
        mission_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create rover path line elements."""
        lines = []
        
        rover_paths = mission_data.get("rover_paths", [])
        
        for path in rover_paths:
            path_coords = path.get("coordinates", [])
            if len(path_coords) > 1:
                lines.append({
                    "type": "line",
                    "subtype": "rover_path",
                    "points": path_coords,
                    "label": path.get("name", "Rover Path"),
                    "style": {
                        "color": [0.0, 1.0, 0.0],  # Green
                        "width": 2.0,
                        "dash_pattern": None,
                        "animated": False
                    },
                    "metadata": path.get("metadata", {})
                })
        
        return lines
    
    def _create_science_markers(
        self,
        mission_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create scientific data markers."""
        markers = []
        
        science_data = mission_data.get("scientific_data", [])
        
        for data_point in science_data:
            markers.append({
                "type": "marker",
                "subtype": "science_data",
                "position": data_point.get("coordinates", [0, 0, 0]),
                "label": data_point.get("type", "Science Data"),
                "style": {
                    "color": [0.0, 0.8, 1.0],  # Cyan
                    "size": 3.0,
                    "shape": "circle",
                    "pulsate": True
                },
                "metadata": data_point.get("data", {})
            })
        
        return markers
    
    def _create_hazard_zones(
        self,
        mission_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create hazard zone elements."""
        zones = []
        
        hazard_zones = mission_data.get("hazard_zones", [])
        
        for zone in hazard_zones:
            zones.append({
                "type": "zone",
                "subtype": "hazard",
                "center": zone.get("center", [0, 0, 0]),
                "radius": zone.get("radius", 100),
                "label": zone.get("type", "Hazard Zone"),
                "style": {
                    "color": [1.0, 0.2, 0.2],  # Red
                    "opacity": 0.3,
                    "border_color": [1.0, 0.0, 0.0],
                    "border_width": 2.0
                },
                "metadata": zone.get("metadata", {})
            })
        
        return zones
    
    def _get_lighting_config(self, preset: str) -> Dict[str, Any]:
        """Get lighting configuration for preset."""
        if preset == "mars_day":
            return {
                "ambient": {
                    "color": [0.4, 0.3, 0.2],
                    "intensity": 0.3
                },
                "directional": {
                    "color": [1.0, 0.9, 0.7],
                    "intensity": 0.8,
                    "direction": [-0.3, -0.8, -0.5],
                    "cast_shadows": True
                }
            }
        elif preset == "mars_night":
            return {
                "ambient": {
                    "color": [0.1, 0.1, 0.15],
                    "intensity": 0.1
                },
                "directional": {
                    "color": [0.3, 0.3, 0.4],
                    "intensity": 0.2,
                    "direction": [-0.5, -0.5, -0.7],
                    "cast_shadows": False
                }
            }
        else:
            return self._get_lighting_config("mars_day")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import time
        return time.strftime("%Y-%m-%d %H:%M:%S")
    
    def export_scene(
        self,
        scene_config: Dict[str, Any],
        output_path: Path,
        format: str = "json"
    ) -> bool:
        """Export 3D scene configuration."""
        try:
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(scene_config, f, indent=2)
                return True
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Scene export failed: {e}")
            return False
