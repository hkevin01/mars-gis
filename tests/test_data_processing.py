"""Unit tests for data processing components."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from tests.conftest import MOCK_MARS_COORDINATES


@pytest.mark.data_processing
class TestDataIngestion:
    """Test cases for Mars data ingestion pipelines."""
    
    def test_mro_data_ingestion(self):
        """Test Mars Reconnaissance Orbiter data ingestion."""
        from mars_gis.data.ingestion import MRODataIngester
        
        ingester = MRODataIngester()
        
        # Mock MRO data file
        mock_data_path = "/mock/path/mro_hirise_image.img"
        
        with patch.object(ingester, 'validate_file_format') as mock_validate:
            with patch.object(ingester, 'parse_metadata') as mock_metadata:
                mock_validate.return_value = True
                mock_metadata.return_value = {
                    'instrument': 'HiRISE',
                    'acquisition_date': '2023-01-15',
                    'target_coordinates': [-14.5684, 175.4729],
                    'resolution': 0.25,  # meters per pixel
                    'bands': ['RED', 'IR', 'BG']
                }
                
                result = ingester.ingest_file(mock_data_path)
                
                assert result is not None
                assert result['status'] == 'success'
                assert result['metadata']['instrument'] == 'HiRISE'
                assert result['metadata']['target_coordinates'] == [-14.5684, 175.4729]
    
    def test_mgs_data_ingestion(self):
        """Test Mars Global Surveyor data ingestion."""
        from mars_gis.data.ingestion import MGSDataIngester
        
        ingester = MGSDataIngester()
        
        mock_data_path = "/mock/path/mgs_mola_elevation.dat"
        
        with patch.object(ingester, 'read_binary_data') as mock_read:
            with patch.object(ingester, 'convert_coordinates') as mock_coords:
                mock_read.return_value = {
                    'elevation_data': [1000, 1100, 1200, 1150],
                    'coordinates': [[-14.5, 175.4], [-14.5, 175.5], [-14.6, 175.4], [-14.6, 175.5]]
                }
                mock_coords.return_value = MOCK_MARS_COORDINATES
                
                result = ingester.ingest_file(mock_data_path)
                
                assert result is not None
                assert result['status'] == 'success'
                assert 'elevation_data' in result
                assert len(result['coordinates']) > 0
    
    def test_batch_data_processing(self):
        """Test batch processing of multiple Mars data files."""
        from mars_gis.data.ingestion import BatchDataProcessor
        
        processor = BatchDataProcessor()
        
        mock_file_list = [
            "/mock/path/file1.img",
            "/mock/path/file2.img",
            "/mock/path/file3.dat"
        ]
        
        with patch.object(processor, 'process_single_file') as mock_process:
            mock_process.return_value = {'status': 'success', 'file_id': 'mock_id'}
            
            results = processor.process_batch(mock_file_list, max_workers=2)
            
            assert len(results) == len(mock_file_list)
            assert all(r['status'] == 'success' for r in results)
            assert mock_process.call_count == len(mock_file_list)
    
    def test_data_validation(self):
        """Test validation of ingested Mars data."""
        from mars_gis.data.validation import MarsDataValidator
        
        validator = MarsDataValidator()
        
        # Valid Mars data sample
        valid_data = {
            'coordinates': [-14.5684, 175.4729],
            'elevation': 1250.5,
            'temperature': -80.0,
            'pressure': 610.0,
            'dust_opacity': 0.3,
            'acquisition_date': '2023-01-15T10:30:00Z'
        }
        
        validation_result = validator.validate_data_point(valid_data)
        
        assert validation_result['is_valid'] is True
        assert len(validation_result['errors']) == 0
        
        # Invalid data sample
        invalid_data = {
            'coordinates': [95.0, 185.0],  # Invalid lat/lon
            'elevation': -50000,           # Unrealistic elevation
            'temperature': 50.0,           # Too warm for Mars
            'pressure': -100.0,            # Negative pressure
        }
        
        validation_result = validator.validate_data_point(invalid_data)
        
        assert validation_result['is_valid'] is False
        assert len(validation_result['errors']) > 0


@pytest.mark.data_processing
class TestDataTransformation:
    """Test cases for Mars data transformation and processing."""
    
    def test_coordinate_reprojection(self):
        """Test coordinate system reprojection for Mars data."""
        from mars_gis.data.transformation import MarsCoordinateTransformer
        
        transformer = MarsCoordinateTransformer()
        
        # Test Mars geographic to Mars cartesian
        mars_geographic = [-14.5684, 175.4729, 1250.5]  # lat, lon, elevation
        
        cartesian_coords = transformer.geographic_to_cartesian(mars_geographic)
        
        assert len(cartesian_coords) == 3  # x, y, z
        assert all(isinstance(coord, float) for coord in cartesian_coords)
        
        # Test reverse transformation
        back_to_geographic = transformer.cartesian_to_geographic(cartesian_coords)
        
        # Should be close to original (within tolerance)
        lat_diff = abs(back_to_geographic[0] - mars_geographic[0])
        lon_diff = abs(back_to_geographic[1] - mars_geographic[1])
        elev_diff = abs(back_to_geographic[2] - mars_geographic[2])
        
        assert lat_diff < 0.001  # Within 0.001 degrees
        assert lon_diff < 0.001
        assert elev_diff < 1.0   # Within 1 meter
    
    def test_atmospheric_data_processing(self, mock_atmospheric_data):
        """Test processing of Mars atmospheric data."""
        if not mock_atmospheric_data:
            pytest.skip("Mock atmospheric data not available")
            
        from mars_gis.data.transformation import AtmosphericDataProcessor
        
        processor = AtmosphericDataProcessor()
        
        processed_data = processor.process_atmospheric_profile(mock_atmospheric_data)
        
        assert 'temperature_profile' in processed_data
        assert 'pressure_profile' in processed_data
        assert 'density_profile' in processed_data
        assert 'altitude_levels' in processed_data
        
        # Check data consistency
        temp_profile = processed_data['temperature_profile']
        pressure_profile = processed_data['pressure_profile']
        
        assert len(temp_profile) == len(pressure_profile)
        assert all(t < 0 for t in temp_profile)  # Mars temperatures are below freezing
        assert all(p > 0 for p in pressure_profile)  # Positive pressures
    
    def test_image_preprocessing(self):
        """Test Mars image preprocessing pipeline."""
        from mars_gis.data.transformation import MarsImageProcessor
        
        processor = MarsImageProcessor()
        
        # Mock image data
        with patch('mars_gis.data.transformation.load_image') as mock_load:
            import numpy as np
            mock_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            mock_load.return_value = mock_image
            
            processed_image = processor.preprocess_mars_image(
                "/mock/path/mars_image.jpg",
                operations=['noise_reduction', 'contrast_enhancement', 'color_correction']
            )
            
            assert processed_image is not None
            assert processed_image.shape == mock_image.shape
            assert processed_image.dtype == np.uint8
    
    def test_temporal_data_alignment(self):
        """Test alignment of temporal Mars data series."""
        from mars_gis.data.transformation import TemporalDataAligner
        
        aligner = TemporalDataAligner()
        
        # Mock time series data with different timestamps
        data_series_1 = {
            'timestamps': ['2023-01-01T00:00:00Z', '2023-01-02T00:00:00Z', '2023-01-03T00:00:00Z'],
            'values': [100, 105, 98],
            'parameter': 'temperature'
        }
        
        data_series_2 = {
            'timestamps': ['2023-01-01T12:00:00Z', '2023-01-02T12:00:00Z', '2023-01-03T12:00:00Z'],
            'values': [610, 615, 608],
            'parameter': 'pressure'
        }
        
        aligned_data = aligner.align_time_series([data_series_1, data_series_2])
        
        assert 'aligned_timestamps' in aligned_data
        assert 'aligned_series' in aligned_data
        assert len(aligned_data['aligned_series']) == 2
        
        # All series should have the same number of time points
        series_lengths = [len(series['values']) for series in aligned_data['aligned_series']]
        assert len(set(series_lengths)) == 1  # All lengths should be the same


@pytest.mark.data_processing
class TestDataStorage:
    """Test cases for Mars data storage and retrieval."""
    
    def test_database_connection(self):
        """Test database connection for Mars data storage."""
        from mars_gis.data.storage import MarsDataDatabase
        
        with patch('mars_gis.data.storage.create_engine') as mock_engine:
            mock_engine.return_value = Mock()
            
            db = MarsDataDatabase("postgresql://test:test@localhost/mars_gis")
            connection = db.get_connection()
            
            assert connection is not None
            mock_engine.assert_called_once()
    
    def test_mission_data_storage(self, mock_mission_data):
        """Test storing mission data in the database."""
        if not mock_mission_data:
            pytest.skip("Mock mission data not available")
            
        from mars_gis.data.storage import MarsDataDatabase
        
        with patch('mars_gis.data.storage.create_engine') as mock_engine:
            mock_connection = Mock()
            mock_engine.return_value.connect.return_value = mock_connection
            
            db = MarsDataDatabase("postgresql://test:test@localhost/mars_gis")
            
            result = db.store_mission_data(mock_mission_data)
            
            assert result['status'] == 'success'
            assert 'mission_id' in result
            mock_connection.execute.assert_called()
    
    def test_data_retrieval_by_coordinates(self):
        """Test retrieving Mars data by coordinate bounds."""
        from mars_gis.data.storage import MarsDataDatabase
        
        with patch('mars_gis.data.storage.create_engine') as mock_engine:
            mock_connection = Mock()
            mock_result = Mock()
            mock_result.fetchall.return_value = [
                (1, -14.5684, 175.4729, 1250.5, -80.0, '2023-01-15'),
                (2, -14.5700, 175.4750, 1245.2, -82.5, '2023-01-16')
            ]
            mock_connection.execute.return_value = mock_result
            mock_engine.return_value.connect.return_value = mock_connection
            
            db = MarsDataDatabase("postgresql://test:test@localhost/mars_gis")
            
            bounds = {
                'min_lat': -15.0,
                'max_lat': -14.0,
                'min_lon': 175.0,
                'max_lon': 176.0
            }
            
            data = db.get_data_by_bounds(bounds)
            
            assert len(data) == 2
            assert all('latitude' in point for point in data)
            assert all('longitude' in point for point in data)
    
    def test_data_caching(self):
        """Test caching mechanism for frequently accessed data."""
        from mars_gis.data.storage import MarsDataCache
        
        cache = MarsDataCache(max_size_mb=100)
        
        # Test cache storage
        test_data = {'elevation': [1000, 1100, 1200], 'coordinates': MOCK_MARS_COORDINATES}
        cache_key = "test_region_elevation"
        
        cache.store(cache_key, test_data)
        
        # Test cache retrieval
        retrieved_data = cache.get(cache_key)
        
        assert retrieved_data is not None
        assert retrieved_data['elevation'] == test_data['elevation']
        assert retrieved_data['coordinates'] == test_data['coordinates']
        
        # Test cache eviction
        cache.clear()
        assert cache.get(cache_key) is None


@pytest.mark.data_processing
class TestDataQuality:
    """Test cases for Mars data quality assessment."""
    
    def test_data_completeness_check(self):
        """Test checking data completeness for Mars datasets."""
        from mars_gis.data.quality import DataCompletenessChecker
        
        checker = DataCompletenessChecker()
        
        # Complete dataset
        complete_dataset = {
            'coordinates': [-14.5684, 175.4729],
            'elevation': 1250.5,
            'temperature': -80.0,
            'pressure': 610.0,
            'dust_opacity': 0.3,
            'wind_speed': 15.2,
            'acquisition_date': '2023-01-15T10:30:00Z'
        }
        
        completeness_score = checker.assess_completeness(complete_dataset)
        
        assert completeness_score == 1.0  # 100% complete
        
        # Incomplete dataset
        incomplete_dataset = {
            'coordinates': [-14.5684, 175.4729],
            'elevation': 1250.5,
            'temperature': None,  # Missing value
            'pressure': 610.0,
            # Missing dust_opacity, wind_speed, acquisition_date
        }
        
        completeness_score = checker.assess_completeness(incomplete_dataset)
        
        assert 0.0 < completeness_score < 1.0  # Partially complete
    
    def test_data_accuracy_validation(self):
        """Test data accuracy validation against known Mars parameters."""
        from mars_gis.data.quality import DataAccuracyValidator
        
        validator = DataAccuracyValidator()
        
        # Accurate Mars data
        accurate_data = {
            'surface_temperature': -80.0,    # Reasonable Mars surface temp
            'atmospheric_pressure': 610.0,   # Typical Mars pressure (Pa)
            'elevation': 1250.5,             # Reasonable elevation
            'dust_opacity': 0.3,             # Reasonable dust level
            'coordinates': [-14.5684, 175.4729]  # Valid Mars coordinates
        }
        
        accuracy_assessment = validator.validate_accuracy(accurate_data)
        
        assert accuracy_assessment['overall_score'] > 0.9  # High accuracy
        assert len(accuracy_assessment['warnings']) == 0
        
        # Inaccurate data
        inaccurate_data = {
            'surface_temperature': 50.0,     # Too warm for Mars
            'atmospheric_pressure': 101325,  # Earth-like pressure
            'elevation': -20000,             # Unrealistic elevation
            'dust_opacity': 5.0,             # Impossible dust opacity
            'coordinates': [95.0, 185.0]     # Invalid coordinates
        }
        
        accuracy_assessment = validator.validate_accuracy(inaccurate_data)
        
        assert accuracy_assessment['overall_score'] < 0.5  # Low accuracy
        assert len(accuracy_assessment['warnings']) > 0
    
    def test_outlier_detection(self):
        """Test outlier detection in Mars data series."""
        from mars_gis.data.quality import MarsDataOutlierDetector
        
        detector = MarsDataOutlierDetector()
        
        # Temperature data with outliers
        temperature_data = [-80, -82, -79, -81, 20, -83, -78, -85, -77]  # 20°C is an outlier
        
        outliers = detector.detect_outliers(temperature_data, parameter='temperature')
        
        assert len(outliers) > 0
        assert 20 in [outliers[i]['value'] for i in range(len(outliers))]
        
        # Elevation data with outliers  
        elevation_data = [1200, 1250, 1180, 1300, -50000, 1220, 1280]  # -50000m is an outlier
        
        outliers = detector.detect_outliers(elevation_data, parameter='elevation')
        
        assert len(outliers) > 0
        assert any(outlier['value'] == -50000 for outlier in outliers)
    
    def test_temporal_consistency(self):
        """Test temporal consistency in Mars data time series."""
        from mars_gis.data.quality import TemporalConsistencyChecker
        
        checker = TemporalConsistencyChecker()
        
        # Consistent time series
        consistent_series = {
            'timestamps': ['2023-01-01T00:00:00Z', '2023-01-01T01:00:00Z', '2023-01-01T02:00:00Z'],
            'values': [-80.0, -79.5, -81.0],  # Gradual temperature changes
            'parameter': 'temperature'
        }
        
        consistency_score = checker.check_consistency(consistent_series)
        
        assert consistency_score > 0.8  # High consistency
        
        # Inconsistent time series
        inconsistent_series = {
            'timestamps': ['2023-01-01T00:00:00Z', '2023-01-01T01:00:00Z', '2023-01-01T02:00:00Z'],
            'values': [-80.0, 50.0, -81.0],  # Sudden temperature spike
            'parameter': 'temperature'
        }
        
        consistency_score = checker.check_consistency(inconsistent_series)
        
        assert consistency_score < 0.5  # Low consistency


@pytest.mark.data_processing
@pytest.mark.integration
class TestDataProcessingIntegration:
    """Integration tests for Mars data processing pipeline."""
    
    def test_end_to_end_data_pipeline(self):
        """Test complete data processing pipeline from ingestion to storage."""
        from mars_gis.data.ingestion import MRODataIngester
        from mars_gis.data.storage import MarsDataDatabase
        from mars_gis.data.transformation import MarsCoordinateTransformer
        from mars_gis.data.validation import MarsDataValidator

        # Mock components
        ingester = MRODataIngester()
        transformer = MarsCoordinateTransformer()
        validator = MarsDataValidator()
        
        with patch.object(ingester, 'ingest_file') as mock_ingest:
            with patch.object(transformer, 'geographic_to_cartesian') as mock_transform:
                with patch.object(validator, 'validate_data_point') as mock_validate:
                    with patch('mars_gis.data.storage.create_engine') as mock_engine:
                        
                        # Setup mocks
                        mock_ingest.return_value = {
                            'status': 'success',
                            'data': {'coordinates': [-14.5684, 175.4729], 'elevation': 1250.5}
                        }
                        mock_transform.return_value = [1000000, 2000000, 3000000]
                        mock_validate.return_value = {'is_valid': True, 'errors': []}
                        
                        mock_connection = Mock()
                        mock_engine.return_value.connect.return_value = mock_connection
                        
                        # Run pipeline
                        db = MarsDataDatabase("postgresql://test:test@localhost/mars_gis")
                        
                        # Step 1: Ingest
                        raw_data = ingester.ingest_file("/mock/path/data.img")
                        
                        # Step 2: Transform
                        cartesian_coords = transformer.geographic_to_cartesian(
                            raw_data['data']['coordinates'] + [raw_data['data']['elevation']]
                        )
                        
                        # Step 3: Validate
                        validation_result = validator.validate_data_point(raw_data['data'])
                        
                        # Step 4: Store (if valid)
                        if validation_result['is_valid']:
                            storage_result = db.store_mission_data(raw_data['data'])
                        
                        # Verify pipeline execution
                        assert raw_data['status'] == 'success'
                        assert len(cartesian_coords) == 3
                        assert validation_result['is_valid'] is True
                        mock_connection.execute.assert_called()
    
    def test_batch_processing_workflow(self):
        """Test batch processing of multiple Mars data files."""
        from mars_gis.data.ingestion import BatchDataProcessor
        from mars_gis.data.quality import DataCompletenessChecker
        
        processor = BatchDataProcessor()
        quality_checker = DataCompletenessChecker()
        
        mock_files = ["/mock/file1.img", "/mock/file2.dat", "/mock/file3.img"]
        
        with patch.object(processor, 'process_single_file') as mock_process:
            with patch.object(quality_checker, 'assess_completeness') as mock_quality:
                
                # Setup mocks for successful processing
                mock_process.return_value = {
                    'status': 'success',
                    'data': {'coordinates': [-14.5684, 175.4729], 'elevation': 1250.5}
                }
                mock_quality.return_value = 0.95  # High quality score
                
                # Process batch
                results = processor.process_batch(mock_files)
                
                # Check quality for each result
                quality_scores = []
                for result in results:
                    if result['status'] == 'success':
                        score = quality_checker.assess_completeness(result['data'])
                        quality_scores.append(score)
                
                # Verify batch processing
                assert len(results) == len(mock_files)
                assert all(r['status'] == 'success' for r in results)
                assert all(score > 0.9 for score in quality_scores)
    
    def test_real_time_data_streaming(self):
        """Test real-time Mars data streaming and processing."""
        from mars_gis.data.streaming import MarsDataStreamProcessor
        
        processor = MarsDataStreamProcessor()
        
        # Mock real-time data stream
        mock_data_stream = [
            {'timestamp': '2023-01-15T10:00:00Z', 'temperature': -80.0, 'pressure': 610.0},
            {'timestamp': '2023-01-15T10:01:00Z', 'temperature': -80.5, 'pressure': 612.0},
            {'timestamp': '2023-01-15T10:02:00Z', 'temperature': -79.8, 'pressure': 608.0}
        ]
        
        with patch.object(processor, 'validate_stream_data') as mock_validate:
            with patch.object(processor, 'buffer_data') as mock_buffer:
                
                mock_validate.return_value = True
                mock_buffer.return_value = {'buffered_points': 3}
                
                # Process stream
                processed_count = 0
                for data_point in mock_data_stream:
                    result = processor.process_data_point(data_point)
                    if result['status'] == 'success':
                        processed_count += 1
                
                # Verify streaming processing
                assert processed_count == len(mock_data_stream)
                assert mock_validate.call_count == len(mock_data_stream)
                assert mock_buffer.call_count == len(mock_data_stream)
