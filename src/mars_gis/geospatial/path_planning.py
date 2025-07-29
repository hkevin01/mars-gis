"""Path planning algorithms for Mars rover operations."""

import heapq
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class MarsPathPlanner:
    """Advanced path planning for Mars rover navigation."""
    
    def __init__(
        self,
        terrain_data: Optional[Dict[str, Any]] = None,
        safety_margin: float = 2.0
    ):
        """
        Initialize path planner.
        
        Args:
            terrain_data: 3D terrain data for planning
            safety_margin: Safety margin in meters
        """
        self.terrain_data = terrain_data
        self.safety_margin = safety_margin
        self.hazard_zones = []
        self.path_cache = {}
        
        # Rover constraints
        self.max_slope = 25.0  # degrees
        self.max_step_height = 0.3  # meters
        self.min_clearance = 0.5  # meters
        self.energy_factor = 1.0  # energy consumption multiplier
    
    def set_rover_constraints(
        self,
        max_slope: float = 25.0,
        max_step_height: float = 0.3,
        min_clearance: float = 0.5,
        energy_factor: float = 1.0
    ):
        """Set rover physical constraints."""
        self.max_slope = max_slope
        self.max_step_height = max_step_height
        self.min_clearance = min_clearance
        self.energy_factor = energy_factor
    
    def add_hazard_zone(
        self,
        center: Tuple[float, float],
        radius: float,
        hazard_type: str = "obstacle"
    ):
        """Add hazard zone to avoid during planning."""
        self.hazard_zones.append({
            "center": center,
            "radius": radius,
            "type": hazard_type
        })
    
    def plan_path_astar(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        resolution: float = 1.0
    ) -> Dict[str, Any]:
        """
        Plan path using A* algorithm.
        
        Args:
            start: Start coordinates (lon, lat)
            goal: Goal coordinates (lon, lat)
            resolution: Grid resolution in meters
            
        Returns:
            Path planning result
        """
        try:
            # Convert coordinates to grid
            grid_bounds = self._get_planning_bounds(start, goal)
            grid = self._create_traversability_grid(grid_bounds, resolution)
            
            start_grid = self._coord_to_grid(start, grid_bounds, resolution)
            goal_grid = self._coord_to_grid(goal, grid_bounds, resolution)
            
            if not self._is_valid_grid_point(start_grid, grid) or \
               not self._is_valid_grid_point(goal_grid, grid):
                return {"error": "Invalid start or goal position"}
            
            # A* search
            path_grid = self._astar_search(start_grid, goal_grid, grid)
            
            if not path_grid:
                return {"error": "No path found"}
            
            # Convert back to coordinates
            path_coords = [
                self._grid_to_coord(point, grid_bounds, resolution)
                for point in path_grid
            ]
            
            # Calculate path metrics
            path_analysis = self._analyze_path(path_coords)
            
            return {
                "path": path_coords,
                "path_length": path_analysis["total_distance"],
                "estimated_time": path_analysis["estimated_time"],
                "energy_cost": path_analysis["energy_cost"],
                "max_slope": path_analysis["max_slope"],
                "difficulty": path_analysis["difficulty"],
                "waypoints": self._generate_waypoints(path_coords),
                "algorithm": "A*",
                "resolution": resolution,
                "grid_size": grid.shape if NUMPY_AVAILABLE else [0, 0]
            }
            
        except Exception as e:
            logger.error(f"A* path planning failed: {e}")
            return {"error": str(e)}
    
    def plan_path_rrt(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        max_iterations: int = 1000,
        step_size: float = 2.0
    ) -> Dict[str, Any]:
        """
        Plan path using RRT (Rapidly-exploring Random Tree).
        
        Args:
            start: Start coordinates
            goal: Goal coordinates
            max_iterations: Maximum RRT iterations
            step_size: Step size for tree expansion
            
        Returns:
            RRT path planning result
        """
        try:
            if not NUMPY_AVAILABLE:
                return {"error": "NumPy required for RRT planning"}
            
            # Initialize RRT
            tree = {0: {"pos": start, "parent": None, "cost": 0.0}}
            node_count = 1
            
            goal_threshold = 5.0  # meters
            
            for iteration in range(max_iterations):
                # Sample random point
                if np.random.random() < 0.1:  # 10% bias toward goal
                    sample = goal
                else:
                    sample = self._sample_random_point(start, goal)
                
                # Find nearest node
                nearest_id = self._find_nearest_node(sample, tree)
                nearest_pos = tree[nearest_id]["pos"]
                
                # Extend toward sample
                new_pos = self._extend_toward(nearest_pos, sample, step_size)
                
                # Check if new position is valid
                if self._is_valid_position(new_pos):
                    # Add new node
                    edge_cost = self._calculate_edge_cost(nearest_pos, new_pos)
                    tree[node_count] = {
                        "pos": new_pos,
                        "parent": nearest_id,
                        "cost": tree[nearest_id]["cost"] + edge_cost
                    }
                    
                    # Check if close to goal
                    if self._distance(new_pos, goal) < goal_threshold:
                        # Connect to goal
                        if self._is_valid_position(goal):
                            final_cost = self._calculate_edge_cost(new_pos, goal)
                            tree[node_count + 1] = {
                                "pos": goal,
                                "parent": node_count,
                                "cost": tree[node_count]["cost"] + final_cost
                            }
                            
                            # Extract path
                            path = self._extract_rrt_path(tree, node_count + 1)
                            path_analysis = self._analyze_path(path)
                            
                            return {
                                "path": path,
                                "path_length": path_analysis["total_distance"],
                                "estimated_time": path_analysis["estimated_time"],
                                "energy_cost": path_analysis["energy_cost"],
                                "iterations": iteration + 1,
                                "nodes_explored": len(tree),
                                "algorithm": "RRT",
                                "success": True
                            }
                    
                    node_count += 1
            
            return {"error": "RRT failed to find path within iterations"}
            
        except Exception as e:
            logger.error(f"RRT path planning failed: {e}")
            return {"error": str(e)}
    
    def plan_multi_waypoint_path(
        self,
        waypoints: List[Tuple[float, float]],
        optimize_order: bool = True
    ) -> Dict[str, Any]:
        """
        Plan path through multiple waypoints.
        
        Args:
            waypoints: List of waypoint coordinates
            optimize_order: Whether to optimize waypoint order
            
        Returns:
            Multi-waypoint path planning result
        """
        try:
            if len(waypoints) < 2:
                return {"error": "At least 2 waypoints required"}
            
            # Optimize waypoint order if requested
            if optimize_order and len(waypoints) > 2:
                waypoints = self._optimize_waypoint_order(waypoints)
            
            # Plan path segments
            segments = []
            total_distance = 0.0
            total_time = 0.0
            total_energy = 0.0
            
            for i in range(len(waypoints) - 1):
                start = waypoints[i]
                goal = waypoints[i + 1]
                
                segment_result = self.plan_path_astar(start, goal)
                
                if "error" in segment_result:
                    return {
                        "error": f"Failed to plan segment {i+1}: {segment_result['error']}"
                    }
                
                segments.append({
                    "start": start,
                    "goal": goal,
                    "path": segment_result["path"],
                    "distance": segment_result["path_length"],
                    "time": segment_result["estimated_time"],
                    "energy": segment_result["energy_cost"]
                })
                
                total_distance += segment_result["path_length"]
                total_time += segment_result["estimated_time"]
                total_energy += segment_result["energy_cost"]
            
            # Combine all paths
            combined_path = []
            for segment in segments:
                if combined_path:
                    # Skip first point of subsequent segments to avoid duplication
                    combined_path.extend(segment["path"][1:])
                else:
                    combined_path.extend(segment["path"])
            
            return {
                "waypoints": waypoints,
                "segments": segments,
                "combined_path": combined_path,
                "total_distance": total_distance,
                "total_time": total_time,
                "total_energy": total_energy,
                "num_segments": len(segments),
                "waypoint_optimized": optimize_order
            }
            
        except Exception as e:
            logger.error(f"Multi-waypoint planning failed: {e}")
            return {"error": str(e)}
    
    def _get_planning_bounds(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> Tuple[float, float, float, float]:
        """Get bounds for planning grid."""
        padding = 0.01  # degrees
        
        min_lon = min(start[0], goal[0]) - padding
        max_lon = max(start[0], goal[0]) + padding
        min_lat = min(start[1], goal[1]) - padding
        max_lat = max(start[1], goal[1]) + padding
        
        return (min_lon, min_lat, max_lon, max_lat)
    
    def _create_traversability_grid(
        self,
        bounds: Tuple[float, float, float, float],
        resolution: float
    ) -> Any:
        """Create traversability grid for planning."""
        if not NUMPY_AVAILABLE:
            return []
        
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Calculate grid dimensions
        width_m = (max_lon - min_lon) * 111000  # Approximate meters
        height_m = (max_lat - min_lat) * 111000
        
        grid_width = int(width_m / resolution)
        grid_height = int(height_m / resolution)
        
        # Initialize grid (0 = traversable, 1 = obstacle)
        grid = np.zeros((grid_height, grid_width), dtype=int)
        
        # Add hazard zones
        for hazard in self.hazard_zones:
            center_lon, center_lat = hazard["center"]
            radius = hazard["radius"]
            
            # Convert to grid coordinates
            center_grid = self._coord_to_grid(
                (center_lon, center_lat), bounds, resolution
            )
            
            radius_grid = int(radius / resolution)
            
            # Mark hazard area
            for i in range(max(0, center_grid[1] - radius_grid),
                          min(grid_height, center_grid[1] + radius_grid + 1)):
                for j in range(max(0, center_grid[0] - radius_grid),
                              min(grid_width, center_grid[0] + radius_grid + 1)):
                    if ((i - center_grid[1])**2 + (j - center_grid[0])**2)**0.5 <= radius_grid:
                        grid[i, j] = 1
        
        # Add terrain-based obstacles if terrain data available
        if self.terrain_data and "elevation_grid" in self.terrain_data:
            self._add_terrain_constraints(grid, bounds, resolution)
        
        return grid
    
    def _add_terrain_constraints(
        self,
        grid: Any,
        bounds: Tuple[float, float, float, float],
        resolution: float
    ):
        """Add terrain-based traversability constraints."""
        if not NUMPY_AVAILABLE:
            return
        
        try:
            elevation_grid = np.array(self.terrain_data["elevation_grid"])
            terrain_coords = self.terrain_data.get("coordinates", {})
            
            if not terrain_coords:
                return
            
            # Calculate slopes
            grad_y, grad_x = np.gradient(elevation_grid)
            slope_degrees = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
            
            # Mark steep areas as obstacles
            steep_mask = slope_degrees > self.max_slope
            
            # Map terrain grid to planning grid
            # This is a simplified mapping - in practice would need proper interpolation
            terrain_shape = elevation_grid.shape
            grid_shape = grid.shape
            
            for ti in range(terrain_shape[0]):
                for tj in range(terrain_shape[1]):
                    if steep_mask[ti, tj]:
                        # Map to planning grid coordinates
                        gi = int(ti * grid_shape[0] / terrain_shape[0])
                        gj = int(tj * grid_shape[1] / terrain_shape[1])
                        
                        if 0 <= gi < grid_shape[0] and 0 <= gj < grid_shape[1]:
                            grid[gi, gj] = 1
        
        except Exception as e:
            logger.error(f"Error adding terrain constraints: {e}")
    
    def _coord_to_grid(
        self,
        coord: Tuple[float, float],
        bounds: Tuple[float, float, float, float],
        resolution: float
    ) -> Tuple[int, int]:
        """Convert coordinates to grid indices."""
        min_lon, min_lat, max_lon, max_lat = bounds
        lon, lat = coord
        
        width_m = (max_lon - min_lon) * 111000
        height_m = (max_lat - min_lat) * 111000
        
        x_m = (lon - min_lon) * 111000
        y_m = (lat - min_lat) * 111000
        
        grid_x = int(x_m / resolution)
        grid_y = int(y_m / resolution)
        
        return (grid_x, grid_y)
    
    def _grid_to_coord(
        self,
        grid_point: Tuple[int, int],
        bounds: Tuple[float, float, float, float],
        resolution: float
    ) -> Tuple[float, float]:
        """Convert grid indices to coordinates."""
        min_lon, min_lat, max_lon, max_lat = bounds
        grid_x, grid_y = grid_point
        
        x_m = grid_x * resolution
        y_m = grid_y * resolution
        
        lon = min_lon + x_m / 111000
        lat = min_lat + y_m / 111000
        
        return (lon, lat)
    
    def _is_valid_grid_point(self, point: Tuple[int, int], grid: Any) -> bool:
        """Check if grid point is valid and traversable."""
        if not NUMPY_AVAILABLE:
            return True
        
        x, y = point
        if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
            return grid[y, x] == 0
        return False
    
    def _astar_search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid: Any
    ) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm."""
        if not NUMPY_AVAILABLE:
            return []
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            for neighbor in self._get_neighbors(current, grid):
                tentative_g = g_score[current] + self._distance_grid(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    
                    if (f_score[neighbor], neighbor) not in open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []
    
    def _get_neighbors(self, point: Tuple[int, int], grid: Any) -> List[Tuple[int, int]]:
        """Get valid neighboring grid points."""
        if not NUMPY_AVAILABLE:
            return []
        
        x, y = point
        neighbors = []
        
        # 8-connected grid
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                if self._is_valid_grid_point((nx, ny), grid):
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic function for A* (Euclidean distance)."""
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
    
    def _distance_grid(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Distance between grid points."""
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
    
    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Distance between coordinate points in meters."""
        lon_diff = (a[0] - b[0]) * 111000 * np.cos(np.radians((a[1] + b[1]) / 2)) if NUMPY_AVAILABLE else (a[0] - b[0]) * 111000
        lat_diff = (a[1] - b[1]) * 111000
        return (lon_diff**2 + lat_diff**2)**0.5
    
    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from A* search."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
    
    def _sample_random_point(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Sample random point for RRT."""
        if not NUMPY_AVAILABLE:
            return start
        
        # Sample within bounding box of start and goal
        min_lon = min(start[0], goal[0]) - 0.01
        max_lon = max(start[0], goal[0]) + 0.01
        min_lat = min(start[1], goal[1]) - 0.01
        max_lat = max(start[1], goal[1]) + 0.01
        
        random_lon = np.random.uniform(min_lon, max_lon)
        random_lat = np.random.uniform(min_lat, max_lat)
        
        return (random_lon, random_lat)
    
    def _find_nearest_node(
        self,
        point: Tuple[float, float],
        tree: Dict[int, Dict[str, Any]]
    ) -> int:
        """Find nearest node in RRT tree."""
        min_dist = float('inf')
        nearest_id = 0
        
        for node_id, node_data in tree.items():
            dist = self._distance(point, node_data["pos"])
            if dist < min_dist:
                min_dist = dist
                nearest_id = node_id
        
        return nearest_id
    
    def _extend_toward(
        self,
        from_pos: Tuple[float, float],
        to_pos: Tuple[float, float],
        step_size: float
    ) -> Tuple[float, float]:
        """Extend from one position toward another."""
        dist = self._distance(from_pos, to_pos)
        if dist <= step_size:
            return to_pos
        
        # Unit vector toward target
        lon_diff = to_pos[0] - from_pos[0]
        lat_diff = to_pos[1] - from_pos[1]
        
        # Normalize and scale
        scale = step_size / (dist * 111000)  # Convert to degrees
        
        new_lon = from_pos[0] + lon_diff * scale
        new_lat = from_pos[1] + lat_diff * scale
        
        return (new_lon, new_lat)
    
    def _is_valid_position(self, pos: Tuple[float, float]) -> bool:
        """Check if position is valid for rover."""
        # Check hazard zones
        for hazard in self.hazard_zones:
            center = hazard["center"]
            radius = hazard["radius"]
            
            if self._distance(pos, center) < radius + self.safety_margin:
                return False
        
        return True
    
    def _calculate_edge_cost(
        self,
        from_pos: Tuple[float, float],
        to_pos: Tuple[float, float]
    ) -> float:
        """Calculate cost of edge between positions."""
        base_cost = self._distance(from_pos, to_pos)
        
        # Add terrain-based costs if available
        if self.terrain_data:
            # Simplified terrain cost calculation
            terrain_cost_multiplier = 1.0
            
            # Check if path crosses difficult terrain
            # This would need proper interpolation in practice
            base_cost *= terrain_cost_multiplier
        
        return base_cost * self.energy_factor
    
    def _extract_rrt_path(
        self,
        tree: Dict[int, Dict[str, Any]],
        goal_id: int
    ) -> List[Tuple[float, float]]:
        """Extract path from RRT tree."""
        path = []
        current_id = goal_id
        
        while current_id is not None:
            path.append(tree[current_id]["pos"])
            current_id = tree[current_id]["parent"]
        
        return path[::-1]
    
    def _analyze_path(self, path: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze path characteristics."""
        if len(path) < 2:
            return {
                "total_distance": 0.0,
                "estimated_time": 0.0,
                "energy_cost": 0.0,
                "max_slope": 0.0,
                "difficulty": "easy"
            }
        
        total_distance = 0.0
        max_slope = 0.0
        
        for i in range(len(path) - 1):
            segment_distance = self._distance(path[i], path[i + 1])
            total_distance += segment_distance
            
            # Estimate slope (simplified)
            if self.terrain_data:
                # Would need proper elevation lookup
                estimated_slope = 0.0  # Placeholder
                max_slope = max(max_slope, estimated_slope)
        
        # Estimate time (assuming 0.5 m/s average speed)
        estimated_time = total_distance / 0.5  # seconds
        
        # Energy cost (simplified)
        energy_cost = total_distance * self.energy_factor
        
        # Difficulty assessment
        if max_slope > 20 or total_distance > 1000:
            difficulty = "hard"
        elif max_slope > 10 or total_distance > 500:
            difficulty = "medium"
        else:
            difficulty = "easy"
        
        return {
            "total_distance": total_distance,
            "estimated_time": estimated_time,
            "energy_cost": energy_cost,
            "max_slope": max_slope,
            "difficulty": difficulty
        }
    
    def _generate_waypoints(
        self,
        path: List[Tuple[float, float]],
        waypoint_spacing: float = 50.0
    ) -> List[Dict[str, Any]]:
        """Generate waypoints along path."""
        if not path:
            return []
        
        waypoints = [{"position": path[0], "type": "start", "distance": 0.0}]
        
        current_distance = 0.0
        last_waypoint_distance = 0.0
        
        for i in range(1, len(path)):
            segment_distance = self._distance(path[i-1], path[i])
            current_distance += segment_distance
            
            # Add waypoint if spacing exceeded
            if current_distance - last_waypoint_distance >= waypoint_spacing:
                waypoints.append({
                    "position": path[i],
                    "type": "navigation",
                    "distance": current_distance
                })
                last_waypoint_distance = current_distance
        
        # Add goal waypoint
        if len(path) > 1:
            waypoints.append({
                "position": path[-1],
                "type": "goal",
                "distance": current_distance
            })
        
        return waypoints
    
    def _optimize_waypoint_order(
        self,
        waypoints: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Optimize waypoint visiting order (simplified TSP)."""
        if len(waypoints) <= 3:
            return waypoints
        
        # Simple nearest neighbor heuristic
        optimized = [waypoints[0]]  # Start with first waypoint
        remaining = waypoints[1:]
        
        while remaining:
            current = optimized[-1]
            nearest_idx = 0
            nearest_dist = float('inf')
            
            for i, candidate in enumerate(remaining):
                dist = self._distance(current, candidate)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = i
            
            optimized.append(remaining.pop(nearest_idx))
        
        return optimized
    
    def export_path(
        self,
        path_result: Dict[str, Any],
        output_path: Path,
        format: str = "json"
    ) -> bool:
        """Export path planning result."""
        try:
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(path_result, f, indent=2)
                return True
            
            elif format.lower() == "gpx":
                return self._export_gpx(path_result, output_path)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Path export failed: {e}")
            return False
    
    def _export_gpx(self, path_result: Dict[str, Any], output_path: Path) -> bool:
        """Export path as GPX file."""
        try:
            path_coords = path_result.get("path", [])
            if not path_coords:
                return False
            
            with open(output_path, 'w') as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<gpx version="1.1" creator="MARS-GIS">\n')
                f.write('  <trk>\n')
                f.write('    <name>Mars Rover Path</name>\n')
                f.write('    <trkseg>\n')
                
                for lon, lat in path_coords:
                    f.write(f'      <trkpt lat="{lat:.8f}" lon="{lon:.8f}"></trkpt>\n')
                
                f.write('    </trkseg>\n')
                f.write('  </trk>\n')
                f.write('</gpx>\n')
            
            return True
            
        except Exception as e:
            logger.error(f"GPX export failed: {e}")
            return False
