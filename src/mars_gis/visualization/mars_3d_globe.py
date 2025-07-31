"""
3D Mars Globe Visualization

This module provides interactive 3D visualization capabilities for Mars data
using foundation model embeddings and multi-modal data sources.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..foundation_models.planetary_scale_embeddings import EmbeddingDatabase


@dataclass
class VisualizationConfig:
    """Configuration for Mars globe visualization."""
    resolution: int = 1024  # Texture resolution
    tile_resolution: int = 256  # Individual tile resolution
    color_scheme: str = "mars_realistic"  # Color scheme for visualization
    elevation_scale: float = 1.0  # Elevation exaggeration factor
    atmosphere_enabled: bool = True  # Show atmospheric effects
    lighting_mode: str = "realistic"  # Lighting mode


@dataclass
class Globe3DData:
    """3D globe data structure."""
    vertices: np.ndarray  # Vertex positions (N, 3)
    faces: np.ndarray     # Face indices (M, 3)
    texture_coords: np.ndarray  # Texture coordinates (N, 2)
    elevation_data: np.ndarray  # Elevation data
    color_data: np.ndarray     # Color texture data
    normal_vectors: np.ndarray  # Surface normals
    metadata: Dict[str, Any]    # Additional metadata


class MarsColorMapper:
    """
    Color mapping for Mars surface features.
    """
    
    def __init__(self, color_scheme: str = "mars_realistic"):
        self.color_scheme = color_scheme
        
        # Define color schemes
        self.color_schemes = {
            "mars_realistic": {
                "low_elevation": [0.6, 0.3, 0.1],     # Brown/red lowlands
                "mid_elevation": [0.8, 0.5, 0.2],     # Orange mid-elevations
                "high_elevation": [0.9, 0.7, 0.4],    # Light brown highlands
                "polar_ice": [0.9, 0.9, 1.0],         # Ice caps
                "dust_storm": [0.7, 0.6, 0.4],        # Dust coverage
                "volcanic": [0.4, 0.2, 0.1],          # Volcanic regions
                "crater": [0.5, 0.3, 0.2],            # Crater materials
                "sedimentary": [0.7, 0.4, 0.2]        # Sedimentary layers
            },
            "scientific": {
                "low_elevation": [0.0, 0.0, 1.0],     # Blue for low
                "mid_elevation": [0.0, 1.0, 0.0],     # Green for mid
                "high_elevation": [1.0, 0.0, 0.0],    # Red for high
                "polar_ice": [1.0, 1.0, 1.0],         # White for ice
                "dust_storm": [1.0, 1.0, 0.0],        # Yellow for dust
                "volcanic": [0.5, 0.0, 0.5],          # Purple for volcanic
                "crater": [0.8, 0.8, 0.8],            # Gray for craters
                "sedimentary": [0.0, 0.8, 0.8]        # Cyan for sedimentary
            },
            "thermal": {
                "cold": [0.0, 0.0, 1.0],              # Blue for cold
                "cool": [0.0, 0.5, 1.0],              # Light blue
                "moderate": [0.0, 1.0, 0.0],          # Green
                "warm": [1.0, 1.0, 0.0],              # Yellow
                "hot": [1.0, 0.5, 0.0],               # Orange
                "very_hot": [1.0, 0.0, 0.0]           # Red for hot
            }
        }
    
    def map_elevation_to_color(
        self,
        elevation: np.ndarray,
        terrain_type: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Map elevation data to RGB colors.
        
        Args:
            elevation: Elevation data array
            terrain_type: Optional terrain classification
            
        Returns:
            RGB color array
        """
        colors = self.color_schemes[self.color_scheme]
        
        # Normalize elevation to 0-1 range
        elevation_norm = ((elevation - elevation.min()) /
                          (elevation.max() - elevation.min()))
        
        # Create base color from elevation
        color_array = np.zeros((*elevation.shape, 3))
        
        # Apply elevation-based coloring
        low_mask = elevation_norm < 0.3
        mid_mask = (elevation_norm >= 0.3) & (elevation_norm < 0.7)
        high_mask = elevation_norm >= 0.7
        
        color_array[low_mask] = colors["low_elevation"]
        color_array[mid_mask] = colors["mid_elevation"]
        color_array[high_mask] = colors["high_elevation"]
        
        # Apply terrain-specific modifications if available
        if terrain_type is not None:
            # This would use actual terrain classification
            # For now, add some variation based on elevation patterns
            crater_mask = self._detect_craters(elevation)
            color_array[crater_mask] = colors["crater"]
        
        return color_array
    
    def map_thermal_to_color(self, thermal_data: np.ndarray) -> np.ndarray:
        """Map thermal data to colors."""
        if self.color_scheme != "thermal":
            return self.map_elevation_to_color(thermal_data)
        
        colors = self.color_schemes["thermal"]
        thermal_norm = ((thermal_data - thermal_data.min()) /
                        (thermal_data.max() - thermal_data.min()))
        
        color_array = np.zeros((*thermal_data.shape, 3))
        
        # Temperature-based color mapping
        masks = [
            (thermal_norm < 0.17, "cold"),
            ((thermal_norm >= 0.17) & (thermal_norm < 0.33), "cool"),
            ((thermal_norm >= 0.33) & (thermal_norm < 0.5), "moderate"),
            ((thermal_norm >= 0.5) & (thermal_norm < 0.67), "warm"),
            ((thermal_norm >= 0.67) & (thermal_norm < 0.83), "hot"),
            (thermal_norm >= 0.83, "very_hot")
        ]
        
        for mask, color_key in masks:
            color_array[mask] = colors[color_key]
        
        return color_array
    
    def _detect_craters(self, elevation: np.ndarray) -> np.ndarray:
        """Simple crater detection based on elevation patterns."""
        # This is a simplified crater detection
        # Real implementation would use more sophisticated algorithms
        
        # Look for circular depressions
        from scipy import ndimage

        # Apply Gaussian filter to smooth
        smoothed = ndimage.gaussian_filter(elevation, sigma=2)
        
        # Find local minima that are significantly lower than surroundings
        local_min = ndimage.minimum_filter(smoothed, size=10)
        threshold = float(np.percentile(smoothed.flatten(), 20))
        crater_candidates = ((smoothed == local_min) &
                             (smoothed < threshold))
        
        return crater_candidates


class Mars3DGlobeGenerator:
    """
    Generator for 3D Mars globe visualization.
    """
    
    def __init__(
        self,
        embedding_db: EmbeddingDatabase,
        config: VisualizationConfig
    ):
        self.embedding_db = embedding_db
        self.config = config
        self.color_mapper = MarsColorMapper(config.color_scheme)
        
    def generate_globe_mesh(
        self,
        radius: float = 3390.0,  # Mars radius in km
        subdivision_level: int = 5
    ) -> Globe3DData:
        """
        Generate 3D mesh for Mars globe.
        
        Args:
            radius: Mars radius in kilometers
            subdivision_level: Mesh subdivision level for detail
            
        Returns:
            Globe3DData with mesh and textures
        """
        # Generate icosphere mesh
        vertices, faces = self._generate_icosphere(subdivision_level)
        
        # Scale to Mars radius
        vertices *= radius
        
        # Generate texture coordinates (spherical projection)
        texture_coords = self._spherical_projection(vertices)
        
        # Load elevation and color data
        elevation_data = self._load_global_elevation_data()
        color_data = self._generate_color_texture(elevation_data)
        
        # Calculate surface normals
        normal_vectors = self._calculate_normals(vertices, faces)
        
        # Apply elevation displacement
        if self.config.elevation_scale > 0:
            vertices = self._apply_elevation_displacement(
                vertices, elevation_data, self.config.elevation_scale
            )
        
        metadata = {
            "radius_km": radius,
            "subdivision_level": subdivision_level,
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "color_scheme": self.config.color_scheme,
            "elevation_scale": self.config.elevation_scale
        }
        
        return Globe3DData(
            vertices=vertices,
            faces=faces,
            texture_coords=texture_coords,
            elevation_data=elevation_data,
            color_data=color_data,
            normal_vectors=normal_vectors,
            metadata=metadata
        )
    
    def _generate_icosphere(
        self, subdivision_level: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate icosphere mesh."""
        # Start with icosahedron
        t = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
        
        # Initial vertices of icosahedron
        vertices = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=np.float32)
        
        # Initial faces
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])
        
        # Subdivide mesh
        for _ in range(subdivision_level):
            vertices, faces = self._subdivide_mesh(vertices, faces)
        
        # Normalize vertices to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        return vertices, faces
    
    def _subdivide_mesh(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subdivide mesh by adding midpoint vertices."""
        edge_vertex_map = {}
        new_vertices = vertices.tolist()
        new_faces = []
        
        def get_midpoint(v1_idx: int, v2_idx: int) -> int:
            edge = tuple(sorted([v1_idx, v2_idx]))
            if edge in edge_vertex_map:
                return edge_vertex_map[edge]
            
            # Create new vertex at midpoint
            v1, v2 = vertices[v1_idx], vertices[v2_idx]
            midpoint = (v1 + v2) / 2
            
            new_idx = len(new_vertices)
            new_vertices.append(midpoint)
            edge_vertex_map[edge] = new_idx
            
            return new_idx
        
        # Subdivide each face into 4 triangles
        for face in faces:
            v1, v2, v3 = face
            
            # Get midpoint vertices
            mid12 = get_midpoint(v1, v2)
            mid23 = get_midpoint(v2, v3)
            mid31 = get_midpoint(v3, v1)
            
            # Create 4 new faces
            new_faces.extend([
                [v1, mid12, mid31],
                [v2, mid23, mid12],
                [v3, mid31, mid23],
                [mid12, mid23, mid31]
            ])
        
        return np.array(new_vertices), np.array(new_faces)
    
    def _spherical_projection(self, vertices: np.ndarray) -> np.ndarray:
        """Convert 3D vertices to spherical texture coordinates."""
        # Normalize vertices
        normalized = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Convert to spherical coordinates
        x, y, z = normalized[:, 0], normalized[:, 1], normalized[:, 2]
        
        # Calculate longitude and latitude
        longitude = np.arctan2(z, x)
        latitude = np.arcsin(y)
        
        # Convert to texture coordinates (0-1 range)
        u = (longitude + np.pi) / (2 * np.pi)
        v = (latitude + np.pi/2) / np.pi
        
        return np.column_stack([u, v])
    
    def _load_global_elevation_data(self) -> np.ndarray:
        """Load global Mars elevation data."""
        # This would load actual MOLA elevation data
        # For now, generate synthetic elevation data
        
        resolution = self.config.resolution
        elevation = np.zeros((resolution, resolution))
        
        # Generate realistic Mars-like elevation patterns
        for i in range(resolution):
            for j in range(resolution):
                # Convert to lat/lon
                lat = (i / resolution - 0.5) * np.pi
                lon = (j / resolution) * 2 * np.pi
                
                # Add various elevation features
                # Polar caps (high elevation)
                polar_factor = 1 - abs(lat) / (np.pi/2)
                
                # Dichotomy (northern lowlands, southern highlands)
                dichotomy = 1 if lat > 0 else -1
                
                # Add some noise for surface roughness
                noise = np.random.random() * 0.1
                
                elevation[i, j] = dichotomy * 2 + polar_factor * 3 + noise
        
        return elevation
    
    def _generate_color_texture(self, elevation_data: np.ndarray) -> np.ndarray:
        """Generate color texture from elevation data."""
        return self.color_mapper.map_elevation_to_color(elevation_data)
    
    def _calculate_normals(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray
    ) -> np.ndarray:
        """Calculate surface normal vectors."""
        normals = np.zeros_like(vertices)
        
        # Calculate face normals and add to vertex normals
        for face in faces:
            v1, v2, v3 = vertices[face]
            
            # Calculate face normal
            edge1 = v2 - v1
            edge2 = v3 - v1
            face_normal = np.cross(edge1, edge2)
            face_normal = face_normal / np.linalg.norm(face_normal)
            
            # Add to vertex normals
            for vertex_idx in face:
                normals[vertex_idx] += face_normal
        
        # Normalize vertex normals
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        return normals
    
    def _apply_elevation_displacement(
        self,
        vertices: np.ndarray,
        elevation_data: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """Apply elevation displacement to vertices."""
        # Sample elevation at each vertex
        texture_coords = self._spherical_projection(vertices)
        
        # Sample elevation data
        h, w = elevation_data.shape
        u_coords = (texture_coords[:, 0] * (w - 1)).astype(int)
        v_coords = (texture_coords[:, 1] * (h - 1)).astype(int)
        
        # Clamp coordinates
        u_coords = np.clip(u_coords, 0, w - 1)
        v_coords = np.clip(v_coords, 0, h - 1)
        
        # Get elevation values
        elevations = elevation_data[v_coords, u_coords]
        
        # Apply displacement along normal direction
        normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        displaced_vertices = vertices + normals * elevations.reshape(-1, 1) * scale
        
        return displaced_vertices
    
    def export_for_web_viewer(
        self, 
        globe_data: Globe3DData, 
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Export globe data for web-based viewer.
        
        Args:
            globe_data: Generated globe data
            output_path: Output directory path
            
        Returns:
            Export metadata
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export mesh data
        mesh_data = {
            "vertices": globe_data.vertices.tolist(),
            "faces": globe_data.faces.tolist(),
            "texture_coords": globe_data.texture_coords.tolist(),
            "normals": globe_data.normal_vectors.tolist(),
            "metadata": globe_data.metadata
        }
        
        with open(output_path / "mars_globe_mesh.json", 'w') as f:
            json.dump(mesh_data, f)
        
        # Export texture data as images
        from PIL import Image

        # Color texture
        color_img = (globe_data.color_data * 255).astype(np.uint8)
        Image.fromarray(color_img).save(output_path / "mars_color_texture.png")
        
        # Elevation texture (grayscale)
        elevation_norm = (globe_data.elevation_data - globe_data.elevation_data.min())
        elevation_norm = elevation_norm / elevation_norm.max()
        elevation_img = (elevation_norm * 255).astype(np.uint8)
        Image.fromarray(elevation_img, mode='L').save(output_path / "mars_elevation_texture.png")
        
        # Export configuration
        config_data = {
            "resolution": self.config.resolution,
            "color_scheme": self.config.color_scheme,
            "elevation_scale": self.config.elevation_scale,
            "atmosphere_enabled": self.config.atmosphere_enabled,
            "lighting_mode": self.config.lighting_mode
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_data, f)
        
        return {
            "mesh_file": "mars_globe_mesh.json",
            "color_texture": "mars_color_texture.png", 
            "elevation_texture": "mars_elevation_texture.png",
            "config_file": "config.json",
            "vertex_count": len(globe_data.vertices),
            "face_count": len(globe_data.faces),
            "export_path": str(output_path)
        }


def create_mars_3d_globe(
    embedding_db: EmbeddingDatabase,
    color_scheme: str = "mars_realistic",
    resolution: int = 1024,
    elevation_scale: float = 1.0
) -> Mars3DGlobeGenerator:
    """
    Factory function to create Mars 3D globe generator.
    
    Args:
        embedding_db: Embedding database for Mars data
        color_scheme: Color scheme for visualization
        resolution: Texture resolution
        elevation_scale: Elevation exaggeration factor
        
    Returns:
        Initialized Mars3DGlobeGenerator
    """
    config = VisualizationConfig(
        resolution=resolution,
        color_scheme=color_scheme,
        elevation_scale=elevation_scale
    )
    
    return Mars3DGlobeGenerator(embedding_db, config)


# Example usage and testing
if __name__ == "__main__":
    from ..foundation_models.planetary_scale_embeddings import EmbeddingDatabase

    # Create embedding database (mock)
    db_path = Path("/tmp/mars_embeddings")
    embedding_db = EmbeddingDatabase(db_path)
    
    # Create 3D globe generator
    globe_generator = create_mars_3d_globe(
        embedding_db,
        color_scheme="mars_realistic",
        resolution=512,
        elevation_scale=2.0
    )
    
    print("Mars 3D Globe Visualization System Initialized")
    print("=" * 50)
    
    # Generate globe mesh
    print("Generating 3D globe mesh...")
    globe_data = globe_generator.generate_globe_mesh(
        radius=3390.0,  # Mars radius in km
        subdivision_level=4
    )
    
    print(f"Globe mesh generated:")
    print(f"Vertices: {len(globe_data.vertices)}")
    print(f"Faces: {len(globe_data.faces)}")
    print(f"Texture resolution: {globe_data.color_data.shape}")
    print(f"Elevation range: {globe_data.elevation_data.min():.1f} to {globe_data.elevation_data.max():.1f}")
    
    # Export for web viewer
    output_path = Path("/tmp/mars_globe_export")
    export_metadata = globe_generator.export_for_web_viewer(globe_data, output_path)
    
    print(f"\nExport completed:")
    print(f"Export path: {export_metadata['export_path']}")
    print(f"Mesh file: {export_metadata['mesh_file']}")
    print(f"Color texture: {export_metadata['color_texture']}")
    print(f"Elevation texture: {export_metadata['elevation_texture']}")
    
    print("\n3D Mars globe visualization ready for web integration!")
