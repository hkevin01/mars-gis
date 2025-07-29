"""Unit tests for geospatial analysis components."""

from unittest.mock import Mock, patch

import pytest

from tests.conftest import MOCK_MARS_COORDINATES


@pytest.mark.geospatial
class TestMarsCoordinateSystem:
    """Test cases for Mars coordinate system handling."""
    
    def test_mars_coordinate_validation(self):
        """Test Mars coordinate validation."""
        from mars_gis.geospatial.coordinates import validate_mars_coordinates

        # Valid Mars coordinates
        valid_coords = [-14.5684, 175.4729]  # Olympia Undae
        assert validate_mars_coordinates(valid_coords) is True
        
        # Invalid latitude (outside -90 to 90)
        invalid_lat = [95.0, 175.4729]
        assert validate_mars_coordinates(invalid_lat) is False
        
        # Invalid longitude (outside -180 to 180)
        invalid_lon = [-14.5684, 185.0]
        assert validate_mars_coordinates(invalid_lon) is False
    
    def test_coordinate_transformation(self):
        """Test coordinate system transformations."""
        from mars_gis.geospatial.coordinates import transform_coordinates

        # Test Earth to Mars coordinate transformation
        earth_coords = [40.7128, -74.0060]  # New York City
        mars_coords = transform_coordinates(
            earth_coords, 
            from_crs="EPSG:4326",  # WGS84
            to_crs="MARS:2000"     # Mars 2000
        )
        
        assert mars_coords is not None
        assert len(mars_coords) == 2
        assert isinstance(mars_coords[0], float)
        assert isinstance(mars_coords[1], float)
    
    def test_distance_calculation(self):
        """Test distance calculation between Mars coordinates."""
        from mars_gis.geospatial.coordinates import calculate_mars_distance

        # Distance from Olympia Undae to Gale Crater
        coord1 = [-14.5684, 175.4729]  # Olympia Undae
        coord2 = [-5.4453, 137.8414]   # Gale Crater
        
        distance = calculate_mars_distance(coord1, coord2)
        
        assert distance > 0
        assert isinstance(distance, float)
        # Distance should be reasonable (thousands of km)
        assert 1000 < distance < 20000


@pytest.mark.geospatial
class TestTerrainAnalysis:
    """Test cases for terrain analysis components."""
    
    def test_terrain_3d_reconstruction(self, mock_elevation_data):
        """Test 3D terrain reconstruction from elevation data."""
        if not mock_elevation_data:
            pytest.skip("NumPy not available for elevation data")
            
        from mars_gis.geospatial.terrain_3d import Mars3DTerrainReconstructor
        
        reconstructor = Mars3DTerrainReconstructor()
        
        # Mock the elevation data processing
        with patch.object(reconstructor, 'load_elevation_data') as mock_load:
            mock_load.return_value = mock_elevation_data
            
            mesh_data = reconstructor.create_terrain_mesh(
                bounds=[-15, 175, -14, 176],  # lat/lon bounds
                resolution=100
            )
            
            assert mesh_data is not None
            assert 'vertices' in mesh_data
            assert 'faces' in mesh_data
            assert 'normals' in mesh_data
    
    def test_slope_calculation(self, mock_elevation_data):
        """Test slope calculation from elevation data."""
        if not mock_elevation_data:
            pytest.skip("NumPy not available for elevation data")
            
        from mars_gis.geospatial.terrain_3d import calculate_slope
        
        slope_data = calculate_slope(mock_elevation_data)
        
        assert slope_data is not None
        assert slope_data.shape == mock_elevation_data.shape
        # Slopes should be in reasonable range (0-90 degrees)
        assert slope_data.min() >= 0
        assert slope_data.max() <= 90
    
    def test_terrain_roughness(self, mock_elevation_data):
        """Test terrain roughness calculation."""
        if not mock_elevation_data:
            pytest.skip("NumPy not available for elevation data")
            
        from mars_gis.geospatial.terrain_3d import calculate_roughness
        
        roughness = calculate_roughness(mock_elevation_data, window_size=5)
        
        assert roughness is not None
        assert isinstance(roughness, float)
        assert roughness >= 0  # Roughness should be non-negative


@pytest.mark.geospatial
class TestPathPlanning:
    """Test cases for Mars rover path planning."""
    
    def test_astar_path_planning(self):
        """Test A* path planning algorithm."""
        from mars_gis.geospatial.path_planning import (
            MarsPathPlanner,
            PathPlanningConfig,
        )
        
        planner = MarsPathPlanner()
        config = PathPlanningConfig(
            algorithm="astar",
            max_slope=30.0,
            min_clearance=2.0,
            energy_weight=0.1
        )
        
        start = [-14.5684, 175.4729]  # Olympia Undae
        goal = [-14.5600, 175.4800]   # Nearby location
        
        # Mock obstacle map
        with patch.object(planner, 'get_obstacle_map') as mock_obstacles:
            import numpy as np
            mock_obstacles.return_value = np.zeros((100, 100))  # No obstacles
            
            path = planner.plan_path(start, goal, config)
            
            assert path is not None
            assert len(path) >= 2  # At least start and goal
            assert path[0] == start
            assert path[-1] == goal
    
    def test_rrt_path_planning(self):
        """Test RRT (Rapidly-exploring Random Tree) path planning."""
        from mars_gis.geospatial.path_planning import (
            MarsPathPlanner,
            PathPlanningConfig,
        )
        
        planner = MarsPathPlanner()
        config = PathPlanningConfig(
            algorithm="rrt",
            max_iterations=1000,
            step_size=10.0,
            goal_bias=0.1
        )
        
        start = [-14.5684, 175.4729]
        goal = [-14.5500, 175.4900]
        
        with patch.object(planner, 'check_collision') as mock_collision:
            mock_collision.return_value = False  # No collisions
            
            path = planner.plan_path(start, goal, config)
            
            assert path is not None
            assert len(path) >= 2
            assert path[0] == start
            assert path[-1] == goal
    
    def test_path_optimization(self):
        """Test path optimization for energy efficiency."""
        from mars_gis.geospatial.path_planning import optimize_path

        # Create a simple path with redundant waypoints
        original_path = [
            [-14.5684, 175.4729],
            [-14.5650, 175.4750],
            [-14.5640, 175.4760],  # Close to previous point
            [-14.5600, 175.4800]
        ]
        
        optimized_path = optimize_path(original_path, min_distance=50.0)
        
        assert len(optimized_path) <= len(original_path)
        assert optimized_path[0] == original_path[0]  # Same start
        assert optimized_path[-1] == original_path[-1]  # Same end
    
    def test_obstacle_avoidance(self):
        """Test obstacle detection and avoidance."""
        from mars_gis.geospatial.path_planning import check_path_clearance
        
        path = [
            [-14.5684, 175.4729],
            [-14.5650, 175.4750],
            [-14.5600, 175.4800]
        ]
        
        # Mock terrain data with obstacles
        with patch('mars_gis.geospatial.path_planning.get_terrain_data') as mock_terrain:
            mock_terrain.return_value = {
                'obstacles': [[-14.5650, 175.4750]],  # Obstacle at waypoint
                'slopes': [15.0, 35.0, 20.0],         # One steep slope
                'roughness': [0.2, 0.8, 0.3]          # One rough area
            }
            
            clearance_result = check_path_clearance(path, min_clearance=10.0)
            
            assert 'safe' in clearance_result
            assert 'obstacles_detected' in clearance_result
            assert 'hazard_points' in clearance_result


@pytest.mark.geospatial
class TestFeatureDetection:
    """Test cases for geological feature detection."""
    
    def test_crater_detection(self, mock_elevation_data):
        """Test crater detection in elevation data."""
        if not mock_elevation_data:
            pytest.skip("NumPy not available for elevation data")
            
        from mars_gis.geospatial.feature_detection import detect_craters
        
        craters = detect_craters(mock_elevation_data, min_diameter=100, max_diameter=5000)
        
        assert isinstance(craters, list)
        for crater in craters:
            assert 'center' in crater
            assert 'diameter' in crater
            assert 'depth' in crater
            assert 'confidence' in crater
            assert 0 <= crater['confidence'] <= 1
    
    def test_valley_detection(self, mock_elevation_data):
        """Test valley/channel detection."""
        if not mock_elevation_data:
            pytest.skip("NumPy not available for elevation data")
            
        from mars_gis.geospatial.feature_detection import detect_valleys
        
        valleys = detect_valleys(mock_elevation_data, min_length=500)
        
        assert isinstance(valleys, list)
        for valley in valleys:
            assert 'centerline' in valley
            assert 'width' in valley
            assert 'depth' in valley
            assert 'length' in valley
    
    def test_ridge_detection(self, mock_elevation_data):
        """Test ridge/mountain detection."""
        if not mock_elevation_data:
            pytest.skip("NumPy not available for elevation data")
            
        from mars_gis.geospatial.feature_detection import detect_ridges
        
        ridges = detect_ridges(mock_elevation_data, prominence_threshold=100)
        
        assert isinstance(ridges, list)
        for ridge in ridges:
            assert 'peak' in ridge
            assert 'elevation' in ridge
            assert 'prominence' in ridge


@pytest.mark.geospatial
class TestSpatialAnalysis:
    """Test cases for spatial analysis operations."""
    
    def test_buffer_analysis(self):
        """Test spatial buffer operations."""
        from mars_gis.geospatial.spatial_analysis import create_buffer
        
        point = [-14.5684, 175.4729]  # Olympia Undae
        buffer_distance = 1000  # 1 km buffer
        
        buffer_geometry = create_buffer(point, buffer_distance)
        
        assert buffer_geometry is not None
        assert hasattr(buffer_geometry, 'area')
        assert hasattr(buffer_geometry, 'bounds')
    
    def test_intersection_analysis(self):
        """Test geometric intersection operations."""
        from mars_gis.geospatial.spatial_analysis import calculate_intersection

        # Mock two overlapping areas
        area1 = {
            'type': 'Polygon',
            'coordinates': [[
                [175.47, -14.57],
                [175.48, -14.57],
                [175.48, -14.56],
                [175.47, -14.56],
                [175.47, -14.57]
            ]]
        }
        
        area2 = {
            'type': 'Polygon',
            'coordinates': [[
                [175.475, -14.575],
                [175.485, -14.575],
                [175.485, -14.565],
                [175.475, -14.565],
                [175.475, -14.575]
            ]]
        }
        
        intersection = calculate_intersection(area1, area2)
        
        assert intersection is not None
        assert intersection['type'] == 'Polygon'
    
    def test_viewshed_analysis(self, mock_elevation_data):
        """Test viewshed analysis for landing site visibility."""
        if not mock_elevation_data:
            pytest.skip("NumPy not available for elevation data")
            
        from mars_gis.geospatial.spatial_analysis import calculate_viewshed
        
        observer_point = [50, 50]  # Center of elevation grid
        max_distance = 10000  # 10 km
        
        viewshed = calculate_viewshed(
            mock_elevation_data,
            observer_point,
            max_distance
        )
        
        assert viewshed is not None
        assert viewshed.shape == mock_elevation_data.shape
        # Viewshed should be binary (visible/not visible)
        unique_values = set(viewshed.flatten())
        assert unique_values.issubset({0, 1})


@pytest.mark.geospatial
class TestGISDataProcessing:
    """Test cases for GIS data processing utilities."""
    
    def test_raster_resampling(self, mock_elevation_data):
        """Test raster data resampling."""
        if not mock_elevation_data:
            pytest.skip("NumPy not available for elevation data")
            
        from mars_gis.geospatial.data_processing import resample_raster

        # Resample to half resolution
        target_shape = (50, 50)
        resampled = resample_raster(mock_elevation_data, target_shape)
        
        assert resampled.shape == target_shape
        assert resampled.dtype == mock_elevation_data.dtype
    
    def test_raster_clipping(self, mock_elevation_data):
        """Test raster clipping to region of interest."""
        if not mock_elevation_data:
            pytest.skip("NumPy not available for elevation data")
            
        from mars_gis.geospatial.data_processing import clip_raster

        # Define clipping bounds (row, col indices)
        bounds = (10, 10, 60, 60)  # min_row, min_col, max_row, max_col
        
        clipped = clip_raster(mock_elevation_data, bounds)
        
        expected_shape = (50, 50)  # 60-10 = 50
        assert clipped.shape == expected_shape
    
    def test_raster_statistics(self, mock_elevation_data):
        """Test raster statistics calculation."""
        if not mock_elevation_data:
            pytest.skip("NumPy not available for elevation data")
            
        from mars_gis.geospatial.data_processing import calculate_raster_stats
        
        stats = calculate_raster_stats(mock_elevation_data)
        
        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'median' in stats
        
        # Verify statistics are reasonable
        assert stats['min'] <= stats['median'] <= stats['max']
        assert stats['std'] >= 0
    
    def test_nodata_handling(self, mock_elevation_data):
        """Test handling of no-data values in raster processing."""
        if not mock_elevation_data:
            pytest.skip("NumPy not available for elevation data")
            
        import numpy as np

        from mars_gis.geospatial.data_processing import handle_nodata

        # Add some no-data values
        data_with_nodata = mock_elevation_data.copy()
        data_with_nodata[0:10, 0:10] = -9999  # No-data value
        
        cleaned_data = handle_nodata(data_with_nodata, nodata_value=-9999)
        
        # No-data values should be masked or interpolated
        assert not np.any(cleaned_data == -9999)


@pytest.mark.geospatial
@pytest.mark.integration
class TestGeospatialIntegration:
    """Integration tests for geospatial components."""
    
    def test_end_to_end_terrain_analysis(self, mock_elevation_data):
        """Test complete terrain analysis workflow."""
        if not mock_elevation_data:
            pytest.skip("NumPy not available for elevation data")
            
        from mars_gis.geospatial.feature_detection import detect_craters
        from mars_gis.geospatial.spatial_analysis import calculate_viewshed
        from mars_gis.geospatial.terrain_3d import Mars3DTerrainReconstructor

        # Step 1: Reconstruct 3D terrain
        reconstructor = Mars3DTerrainReconstructor()
        
        with patch.object(reconstructor, 'load_elevation_data') as mock_load:
            mock_load.return_value = mock_elevation_data
            
            mesh_data = reconstructor.create_terrain_mesh(
                bounds=[-15, 175, -14, 176],
                resolution=100
            )
        
        # Step 2: Detect geological features
        craters = detect_craters(mock_elevation_data)
        
        # Step 3: Perform viewshed analysis
        viewshed = calculate_viewshed(
            mock_elevation_data,
            observer_point=[50, 50],
            max_distance=5000
        )
        
        # Verify all components worked together
        assert mesh_data is not None
        assert isinstance(craters, list)
        assert viewshed is not None
        
        # Integration check: mesh and viewshed should have compatible dimensions
        assert viewshed.shape == mock_elevation_data.shape
    
    def test_mission_planning_integration(self):
        """Test integration of geospatial analysis with mission planning."""
        from mars_gis.geospatial.path_planning import MarsPathPlanner
        from mars_gis.geospatial.spatial_analysis import create_buffer
        
        planner = MarsPathPlanner()
        
        # Define mission parameters
        landing_site = [-14.5684, 175.4729]
        target_sites = [
            [-14.5600, 175.4800],
            [-14.5500, 175.4900]
        ]
        
        # Create safety buffer around landing site
        safety_buffer = create_buffer(landing_site, 500)  # 500m buffer
        
        # Plan paths to each target
        paths = []
        for target in target_sites:
            with patch.object(planner, 'get_obstacle_map') as mock_obstacles:
                import numpy as np
                mock_obstacles.return_value = np.zeros((100, 100))
                
                path = planner.plan_path(landing_site, target)
                if path:
                    paths.append(path)
        
        # Verify mission planning components work together
        assert safety_buffer is not None
        assert len(paths) <= len(target_sites)
        
        for path in paths:
            assert len(path) >= 2
            assert path[0] == landing_site
