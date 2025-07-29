"""3D terrain reconstruction and analysis for Mars."""

import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from scipy.spatial import ConvexHull, Delaunay
    from scipy.interpolate import griddata, RBFInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ConvexHull = Delaunay = griddata = RBFInterpolator = None


class Mars3DTerrainReconstructor:
    """3D terrain reconstruction from Mars elevation data."""
    
    def __init__(self, resolution: float = 100.0):
        """
        Initialize terrain reconstructor.
        
        Args:
            resolution: Spatial resolution in meters
        """
        self.resolution = resolution
        self.terrain_cache = {}
        
    def reconstruct_from_elevation_data(
        self,
        elevation_points: List[Tuple[float, float, float]],
        bounds: Optional[Tuple[float, float, float, float]] = None,
        method: str = "delaunay"
    ) -> Dict[str, Any]:
        """
        Reconstruct 3D terrain from elevation points.
        
        Args:
            elevation_points: List of (lon, lat, elevation) tuples
            bounds: Optional (min_lon, min_lat, max_lon, max_lat) bounds
            method: Interpolation method ('delaunay', 'rbf', 'grid')
            
        Returns:
            Dictionary containing terrain mesh data
        """
        if not NUMPY_AVAILABLE:
            logger.error("NumPy required for terrain reconstruction")
            return {}
        
        if not elevation_points:
            logger.error("No elevation points provided")
            return {}
        
        try:
            # Convert to numpy arrays
            points = np.array(elevation_points)
            x_coords = points[:, 0]  # longitude
            y_coords = points[:, 1]  # latitude
            z_values = points[:, 2]  # elevation
            
            # Determine bounds
            if bounds is None:
                bounds = (
                    float(np.min(x_coords)), float(np.min(y_coords)),
                    float(np.max(x_coords)), float(np.max(y_coords))
                )
            
            min_lon, min_lat, max_lon, max_lat = bounds
            
            # Create grid
            num_x = int((max_lon - min_lon) * 111000 / self.resolution)  # ~111km per degree
            num_y = int((max_lat - min_lat) * 111000 / self.resolution)
            
            num_x = max(10, min(num_x, 1000))  # Reasonable limits
            num_y = max(10, min(num_y, 1000))
            
            grid_x = np.linspace(min_lon, max_lon, num_x)
            grid_y = np.linspace(min_lat, max_lat, num_y)
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
            
            # Interpolate elevation
            if method == "delaunay" and SCIPY_AVAILABLE:
                grid_z = griddata(
                    (x_coords, y_coords), z_values,
                    (grid_xx, grid_yy), method='linear',
                    fill_value=np.mean(z_values)
                )
            elif method == "rbf" and SCIPY_AVAILABLE:
                # RBF interpolation for smoother results
                interpolator = RBFInterpolator(
                    np.column_stack((x_coords, y_coords)),
                    z_values, kernel='thin_plate_spline'
                )
                grid_z = interpolator(np.column_stack((
                    grid_xx.ravel(), grid_yy.ravel()
                ))).reshape(grid_xx.shape)
            else:
                # Simple grid interpolation fallback
                grid_z = self._simple_grid_interpolation(
                    x_coords, y_coords, z_values, grid_xx, grid_yy
                )
            
            # Generate mesh vertices and faces
            vertices, faces = self._generate_mesh(grid_xx, grid_yy, grid_z)
            
            # Calculate terrain statistics
            stats = self._calculate_terrain_stats(grid_z)
            
            terrain_data = {
                "method": method,
                "resolution": self.resolution,
                "bounds": bounds,
                "grid_shape": [num_y, num_x],
                "vertices": vertices.tolist() if NUMPY_AVAILABLE else [],
                "faces": faces.tolist() if NUMPY_AVAILABLE else [],
                "elevation_grid": grid_z.tolist() if NUMPY_AVAILABLE else [],
                "coordinates": {
                    "longitude": grid_x.tolist() if NUMPY_AVAILABLE else [],
                    "latitude": grid_y.tolist() if NUMPY_AVAILABLE else []
                },
                "statistics": stats,
                "metadata": {
                    "num_input_points": len(elevation_points),
                    "num_vertices": len(vertices) if NUMPY_AVAILABLE else 0,
                    "num_faces": len(faces) if NUMPY_AVAILABLE else 0
                }
            }
            
            return terrain_data
            
        except Exception as e:
            logger.error(f"Terrain reconstruction failed: {e}")
            return {}
    
    def _simple_grid_interpolation(
        self,
        x_coords: Any, y_coords: Any, z_values: Any,
        grid_xx: Any, grid_yy: Any
    ) -> Any:
        """Simple nearest neighbor interpolation fallback."""
        if not NUMPY_AVAILABLE:
            return []
        
        grid_z = np.zeros_like(grid_xx)
        
        for i in range(grid_xx.shape[0]):
            for j in range(grid_xx.shape[1]):
                # Find nearest point
                distances = ((x_coords - grid_xx[i, j]) ** 2 + 
                           (y_coords - grid_yy[i, j]) ** 2)
                nearest_idx = np.argmin(distances)
                grid_z[i, j] = z_values[nearest_idx]
        
        return grid_z
    
    def _generate_mesh(
        self,
        grid_xx: Any, grid_yy: Any, grid_z: Any
    ) -> Tuple[Any, Any]:
        """Generate mesh vertices and faces from elevation grid."""
        if not NUMPY_AVAILABLE:
            return [], []
        
        # Create vertices
        vertices = []
        for i in range(grid_xx.shape[0]):
            for j in range(grid_xx.shape[1]):
                vertices.append([
                    float(grid_xx[i, j]),  # longitude
                    float(grid_yy[i, j]),  # latitude
                    float(grid_z[i, j])    # elevation
                ])
        
        vertices = np.array(vertices)
        
        # Create faces (triangles)
        faces = []
        rows, cols = grid_xx.shape
        
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Two triangles per grid cell
                v1 = i * cols + j
                v2 = i * cols + (j + 1)
                v3 = (i + 1) * cols + j
                v4 = (i + 1) * cols + (j + 1)
                
                # Triangle 1
                faces.append([v1, v2, v3])
                # Triangle 2
                faces.append([v2, v4, v3])
        
        faces = np.array(faces)
        
        return vertices, faces
    
    def _calculate_terrain_stats(self, elevation_grid: Any) -> Dict[str, float]:
        """Calculate terrain statistics."""
        if not NUMPY_AVAILABLE or elevation_grid is None:
            return {}
        
        try:
            flat_elevations = elevation_grid.ravel()
            valid_elevations = flat_elevations[~np.isnan(flat_elevations)]
            
            if len(valid_elevations) == 0:
                return {}
            
            # Calculate gradients for roughness
            if elevation_grid.ndim == 2:
                grad_y, grad_x = np.gradient(elevation_grid)
                slope = np.sqrt(grad_x**2 + grad_y**2)
                roughness = np.std(slope)
            else:
                roughness = 0.0
            
            return {
                "min_elevation": float(np.min(valid_elevations)),
                "max_elevation": float(np.max(valid_elevations)),
                "mean_elevation": float(np.mean(valid_elevations)),
                "std_elevation": float(np.std(valid_elevations)),
                "elevation_range": float(np.max(valid_elevations) - np.min(valid_elevations)),
                "roughness": float(roughness)
            }
            
        except Exception as e:
            logger.error(f"Error calculating terrain stats: {e}")
            return {}
    
    def detect_terrain_features(
        self,
        terrain_data: Dict[str, Any],
        feature_types: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect terrain features from 3D data.
        
        Args:
            terrain_data: Terrain data from reconstruction
            feature_types: Types of features to detect
            
        Returns:
            Dictionary of detected features by type
        """
        if feature_types is None:
            feature_types = ["peaks", "valleys", "ridges", "depressions"]
        
        if not NUMPY_AVAILABLE or not terrain_data.get("elevation_grid"):
            return {ft: [] for ft in feature_types}
        
        try:
            elevation_grid = np.array(terrain_data["elevation_grid"])
            coordinates = terrain_data.get("coordinates", {})
            lon_coords = coordinates.get("longitude", [])
            lat_coords = coordinates.get("latitude", [])
            
            if not lon_coords or not lat_coords:
                return {ft: [] for ft in feature_types}
            
            features = {}
            
            # Detect peaks
            if "peaks" in feature_types:
                features["peaks"] = self._detect_peaks(
                    elevation_grid, lon_coords, lat_coords
                )
            
            # Detect valleys
            if "valleys" in feature_types:
                features["valleys"] = self._detect_valleys(
                    elevation_grid, lon_coords, lat_coords
                )
            
            # Detect ridges
            if "ridges" in feature_types:
                features["ridges"] = self._detect_ridges(
                    elevation_grid, lon_coords, lat_coords
                )
            
            # Detect depressions
            if "depressions" in feature_types:
                features["depressions"] = self._detect_depressions(
                    elevation_grid, lon_coords, lat_coords
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Feature detection failed: {e}")
            return {ft: [] for ft in feature_types}
    
    def _detect_peaks(
        self,
        elevation_grid: Any,
        lon_coords: List[float],
        lat_coords: List[float]
    ) -> List[Dict[str, Any]]:
        """Detect elevation peaks."""
        if not NUMPY_AVAILABLE:
            return []
        
        peaks = []
        rows, cols = elevation_grid.shape
        
        # Simple peak detection - local maxima
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center_elev = elevation_grid[i, j]
                
                # Check if center is higher than all neighbors
                is_peak = True
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if elevation_grid[i + di, j + dj] >= center_elev:
                            is_peak = False
                            break
                    if not is_peak:
                        break
                
                if is_peak:
                    peaks.append({
                        "longitude": lon_coords[j],
                        "latitude": lat_coords[i],
                        "elevation": float(center_elev),
                        "type": "peak",
                        "prominence": self._calculate_prominence(
                            elevation_grid, i, j
                        )
                    })
        
        return peaks
    
    def _detect_valleys(
        self,
        elevation_grid: Any,
        lon_coords: List[float],
        lat_coords: List[float]
    ) -> List[Dict[str, Any]]:
        """Detect valleys (local minima)."""
        if not NUMPY_AVAILABLE:
            return []
        
        valleys = []
        rows, cols = elevation_grid.shape
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center_elev = elevation_grid[i, j]
                
                # Check if center is lower than all neighbors
                is_valley = True
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if elevation_grid[i + di, j + dj] <= center_elev:
                            is_valley = False
                            break
                    if not is_valley:
                        break
                
                if is_valley:
                    valleys.append({
                        "longitude": lon_coords[j],
                        "latitude": lat_coords[i],
                        "elevation": float(center_elev),
                        "type": "valley",
                        "depth": self._calculate_depth(elevation_grid, i, j)
                    })
        
        return valleys
    
    def _detect_ridges(
        self,
        elevation_grid: Any,
        lon_coords: List[float],
        lat_coords: List[float]
    ) -> List[Dict[str, Any]]:
        """Detect ridges using gradient analysis."""
        if not NUMPY_AVAILABLE:
            return []
        
        ridges = []
        
        try:
            # Calculate gradients
            grad_y, grad_x = np.gradient(elevation_grid)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate second derivatives for ridge detection
            grad_xx = np.gradient(grad_x, axis=1)
            grad_yy = np.gradient(grad_y, axis=0)
            grad_xy = np.gradient(grad_x, axis=0)
            
            # Hessian determinant
            hessian_det = grad_xx * grad_yy - grad_xy**2
            
            # Ridge points have negative Hessian determinant and high gradient
            ridge_mask = (hessian_det < -0.1) & (grad_magnitude > np.percentile(grad_magnitude, 75))
            
            ridge_points = np.where(ridge_mask)
            
            for i, j in zip(ridge_points[0], ridge_points[1]):
                if 0 <= i < len(lat_coords) and 0 <= j < len(lon_coords):
                    ridges.append({
                        "longitude": lon_coords[j],
                        "latitude": lat_coords[i],
                        "elevation": float(elevation_grid[i, j]),
                        "type": "ridge",
                        "gradient_magnitude": float(grad_magnitude[i, j])
                    })
        
        except Exception as e:
            logger.error(f"Ridge detection error: {e}")
        
        return ridges
    
    def _detect_depressions(
        self,
        elevation_grid: Any,
        lon_coords: List[float],
        lat_coords: List[float]
    ) -> List[Dict[str, Any]]:
        """Detect depressions (crater-like features)."""
        if not NUMPY_AVAILABLE:
            return []
        
        depressions = []
        
        try:
            # Use morphological operations for depression detection
            # Simplified approach: look for circular low areas
            rows, cols = elevation_grid.shape
            
            for i in range(2, rows - 2):
                for j in range(2, cols - 2):
                    center_elev = elevation_grid[i, j]
                    
                    # Check circular pattern around center
                    surrounding_elevs = []
                    for radius in [1, 2]:
                        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                            di = int(radius * np.sin(angle))
                            dj = int(radius * np.cos(angle))
                            if 0 <= i + di < rows and 0 <= j + dj < cols:
                                surrounding_elevs.append(elevation_grid[i + di, j + dj])
                    
                    if surrounding_elevs:
                        mean_surrounding = np.mean(surrounding_elevs)
                        if center_elev < mean_surrounding - 10:  # Significant depression
                            depressions.append({
                                "longitude": lon_coords[j],
                                "latitude": lat_coords[i],
                                "elevation": float(center_elev),
                                "type": "depression",
                                "depth": float(mean_surrounding - center_elev),
                                "estimated_diameter": self._estimate_depression_diameter(
                                    elevation_grid, i, j
                                )
                            })
        
        except Exception as e:
            logger.error(f"Depression detection error: {e}")
        
        return depressions
    
    def _calculate_prominence(self, elevation_grid: Any, i: int, j: int) -> float:
        """Calculate peak prominence."""
        if not NUMPY_AVAILABLE:
            return 0.0
        
        try:
            center_elev = elevation_grid[i, j]
            rows, cols = elevation_grid.shape
            
            # Find minimum elevation in surrounding area
            min_elev = center_elev
            for radius in range(1, min(10, min(rows//2, cols//2))):
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            min_elev = min(min_elev, elevation_grid[ni, nj])
            
            return float(center_elev - min_elev)
            
        except Exception:
            return 0.0
    
    def _calculate_depth(self, elevation_grid: Any, i: int, j: int) -> float:
        """Calculate valley depth."""
        if not NUMPY_AVAILABLE:
            return 0.0
        
        try:
            center_elev = elevation_grid[i, j]
            rows, cols = elevation_grid.shape
            
            # Find maximum elevation in surrounding area
            max_elev = center_elev
            for radius in range(1, min(10, min(rows//2, cols//2))):
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            max_elev = max(max_elev, elevation_grid[ni, nj])
            
            return float(max_elev - center_elev)
            
        except Exception:
            return 0.0
    
    def _estimate_depression_diameter(
        self,
        elevation_grid: Any,
        center_i: int,
        center_j: int
    ) -> float:
        """Estimate depression diameter in grid units."""
        if not NUMPY_AVAILABLE:
            return 0.0
        
        try:
            center_elev = elevation_grid[center_i, center_j]
            rows, cols = elevation_grid.shape
            
            # Find where elevation rises significantly
            threshold = center_elev + 5  # 5m threshold
            max_radius = 0
            
            for radius in range(1, min(20, min(rows//2, cols//2))):
                rim_count = 0
                total_count = 0
                
                for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                    di = int(radius * np.sin(angle))
                    dj = int(radius * np.cos(angle))
                    ni, nj = center_i + di, center_j + dj
                    
                    if 0 <= ni < rows and 0 <= nj < cols:
                        total_count += 1
                        if elevation_grid[ni, nj] > threshold:
                            rim_count += 1
                
                if total_count > 0 and rim_count / total_count > 0.5:
                    max_radius = radius
                    break
            
            return float(max_radius * 2)  # Diameter in grid units
            
        except Exception:
            return 0.0
    
    def export_terrain_model(
        self,
        terrain_data: Dict[str, Any],
        output_path: Path,
        format: str = "obj"
    ) -> bool:
        """
        Export terrain model to file.
        
        Args:
            terrain_data: Terrain data to export
            output_path: Output file path
            format: Export format ('obj', 'stl', 'json')
            
        Returns:
            Success status
        """
        try:
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(terrain_data, f, indent=2)
                return True
            
            elif format.lower() == "obj":
                return self._export_obj(terrain_data, output_path)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def _export_obj(self, terrain_data: Dict[str, Any], output_path: Path) -> bool:
        """Export terrain as Wavefront OBJ file."""
        try:
            vertices = terrain_data.get("vertices", [])
            faces = terrain_data.get("faces", [])
            
            if not vertices or not faces:
                logger.error("No mesh data to export")
                return False
            
            with open(output_path, 'w') as f:
                f.write("# Mars terrain model generated by MARS-GIS\n")
                f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n\n")
                
                # Write vertices
                for vertex in vertices:
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                f.write("\n")
                
                # Write faces (OBJ uses 1-based indexing)
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            logger.info(f"Terrain model exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"OBJ export failed: {e}")
            return False
