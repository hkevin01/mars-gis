"""
Mars-GIS Backend Unit Tests
Test-Driven Development for Foundation Models and API Endpoints

Following TDD Red-Green-Refactor cycle for comprehensive backend testing
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mars_gis.api.routes import router
from mars_gis.core.config import settings
from mars_gis.models.comparative import ComparativePlanetaryAnalyzer

# Import our modules to test
from mars_gis.models.foundation import MarsEarthTransferModel
from mars_gis.models.multimodal import MultiModalMarsProcessor
from mars_gis.models.optimization import MarsLandingSiteOptimizer
from mars_gis.models.planetary_scale import PlanetaryScaleEmbeddingGenerator
from mars_gis.models.self_supervised import SelfSupervisedMarsLearning


class TestMarsEarthTransferModel:
    """
    TDD Unit Tests for Earth-Mars Transfer Learning Model
    RED phase: Write failing tests for expected functionality
    """

    @pytest.fixture
    def model(self):
        """Create model instance for testing"""
        return MarsEarthTransferModel(
            earth_pretrained_path="test_model.pth",
            mars_adaptation_layers=3,
            feature_dim=512
        )

    @pytest.fixture
    def sample_earth_data(self):
        """Sample Earth observation data"""
        return torch.randn(1, 3, 224, 224)

    @pytest.fixture
    def sample_mars_data(self):
        """Sample Mars observation data"""
        return torch.randn(1, 3, 224, 224)

    def test_model_initialization(self, model):
        """Test model initializes with correct architecture"""
        # RED: This test will fail initially
        assert hasattr(model, 'earth_encoder')
        assert hasattr(model, 'mars_adapter')
        assert hasattr(model, 'feature_extractor')
        assert model.feature_dim == 512
        assert model.mars_adaptation_layers == 3

    def test_earth_feature_extraction(self, model, sample_earth_data):
        """Test Earth feature extraction functionality"""
        # RED: Will fail until we implement the forward pass
        model.eval()
        with torch.no_grad():
            features = model.extract_earth_features(sample_earth_data)

        assert features.shape == (1, 512)
        assert not torch.isnan(features).any()
        assert features.dtype == torch.float32

    def test_mars_adaptation(self, model, sample_mars_data):
        """Test Mars domain adaptation"""
        # RED: Will fail until adaptation layer is implemented
        model.eval()
        with torch.no_grad():
            adapted_features = model.adapt_to_mars(sample_mars_data)

        assert adapted_features.shape == (1, 512)
        assert not torch.isnan(adapted_features).any()

    def test_transfer_learning_training(self, model, sample_earth_data, sample_mars_data):
        """Test transfer learning training process"""
        # RED: Will fail until training loop is implemented
        model.train()

        # Mock optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Forward pass
        earth_features = model.extract_earth_features(sample_earth_data)
        mars_features = model.adapt_to_mars(sample_mars_data)

        # Compute transfer loss (domain adaptation loss)
        transfer_loss = model.compute_transfer_loss(earth_features, mars_features)

        assert transfer_loss.requires_grad
        assert transfer_loss.item() > 0

        # Backward pass
        optimizer.zero_grad()
        transfer_loss.backward()
        optimizer.step()

    def test_model_save_load(self, model, tmp_path):
        """Test model serialization and deserialization"""
        # RED: Will fail until save/load methods are implemented
        save_path = tmp_path / "test_model.pth"

        # Save model
        model.save_model(str(save_path))
        assert save_path.exists()

        # Load model
        loaded_model = MarsEarthTransferModel.load_model(str(save_path))
        assert loaded_model.feature_dim == model.feature_dim
        assert loaded_model.mars_adaptation_layers == model.mars_adaptation_layers

    def test_error_handling_invalid_input(self, model):
        """Test error handling for invalid inputs"""
        # RED: Will fail until error handling is implemented
        with pytest.raises(ValueError):
            model.extract_earth_features(torch.randn(1, 4, 224, 224))  # Wrong channels

        with pytest.raises(ValueError):
            model.adapt_to_mars(torch.randn(2, 3, 128, 128))  # Wrong dimensions


class TestMultiModalMarsProcessor:
    """
    TDD Unit Tests for Multi-Modal Mars Data Processing
    """

    @pytest.fixture
    def processor(self):
        """Create processor instance for testing"""
        return MultiModalMarsProcessor(
            modalities=['visual', 'spectral', 'thermal'],
            fusion_method='attention',
            output_dim=256
        )

    @pytest.fixture
    def sample_visual_data(self):
        """Sample visual Mars data"""
        return torch.randn(1, 3, 256, 256)

    @pytest.fixture
    def sample_spectral_data(self):
        """Sample spectral Mars data"""
        return torch.randn(1, 128, 256, 256)

    @pytest.fixture
    def sample_thermal_data(self):
        """Sample thermal Mars data"""
        return torch.randn(1, 1, 256, 256)

    def test_processor_initialization(self, processor):
        """Test processor initializes correctly"""
        # RED: Will fail until MultiModalMarsProcessor is implemented
        assert processor.modalities == ['visual', 'spectral', 'thermal']
        assert processor.fusion_method == 'attention'
        assert processor.output_dim == 256
        assert hasattr(processor, 'visual_encoder')
        assert hasattr(processor, 'spectral_encoder')
        assert hasattr(processor, 'thermal_encoder')
        assert hasattr(processor, 'fusion_layer')

    def test_visual_processing(self, processor, sample_visual_data):
        """Test visual data processing"""
        # RED: Will fail until visual encoder is implemented
        processor.eval()
        with torch.no_grad():
            visual_features = processor.process_visual(sample_visual_data)

        assert visual_features.shape[0] == 1  # Batch size
        assert visual_features.shape[1] == 256  # Feature dimension
        assert not torch.isnan(visual_features).any()

    def test_spectral_processing(self, processor, sample_spectral_data):
        """Test spectral data processing"""
        # RED: Will fail until spectral encoder is implemented
        processor.eval()
        with torch.no_grad():
            spectral_features = processor.process_spectral(sample_spectral_data)

        assert spectral_features.shape[0] == 1
        assert spectral_features.shape[1] == 256
        assert not torch.isnan(spectral_features).any()

    def test_thermal_processing(self, processor, sample_thermal_data):
        """Test thermal data processing"""
        # RED: Will fail until thermal encoder is implemented
        processor.eval()
        with torch.no_grad():
            thermal_features = processor.process_thermal(sample_thermal_data)

        assert thermal_features.shape[0] == 1
        assert thermal_features.shape[1] == 256
        assert not torch.isnan(thermal_features).any()

    def test_multimodal_fusion(self, processor, sample_visual_data,
                              sample_spectral_data, sample_thermal_data):
        """Test multi-modal data fusion"""
        # RED: Will fail until fusion mechanism is implemented
        processor.eval()
        with torch.no_grad():
            fused_features = processor.fuse_modalities(
                visual=sample_visual_data,
                spectral=sample_spectral_data,
                thermal=sample_thermal_data
            )

        assert fused_features.shape == (1, 256)
        assert not torch.isnan(fused_features).any()

    def test_attention_weights(self, processor, sample_visual_data,
                              sample_spectral_data, sample_thermal_data):
        """Test attention mechanism produces valid weights"""
        # RED: Will fail until attention mechanism is implemented
        processor.eval()
        with torch.no_grad():
            _, attention_weights = processor.fuse_modalities(
                visual=sample_visual_data,
                spectral=sample_spectral_data,
                thermal=sample_thermal_data,
                return_attention=True
            )

        assert attention_weights.shape == (1, 3)  # 3 modalities
        assert torch.allclose(attention_weights.sum(dim=1), torch.ones(1))
        assert (attention_weights >= 0).all()


class TestComparativePlanetaryAnalyzer:
    """
    TDD Unit Tests for Comparative Planetary Analysis
    """

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        return ComparativePlanetaryAnalyzer(
            feature_dim=512,
            similarity_threshold=0.7,
            comparison_method='cosine'
        )

    @pytest.fixture
    def sample_mars_features(self):
        """Sample Mars feature vectors"""
        return torch.randn(10, 512)

    @pytest.fixture
    def sample_earth_features(self):
        """Sample Earth feature vectors"""
        return torch.randn(20, 512)

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initializes correctly"""
        # RED: Will fail until ComparativePlanetaryAnalyzer is implemented
        assert analyzer.feature_dim == 512
        assert analyzer.similarity_threshold == 0.7
        assert analyzer.comparison_method == 'cosine'
        assert hasattr(analyzer, 'similarity_computer')

    def test_cosine_similarity_computation(self, analyzer, sample_mars_features, sample_earth_features):
        """Test cosine similarity computation"""
        # RED: Will fail until similarity computation is implemented
        similarities = analyzer.compute_similarity(sample_mars_features, sample_earth_features)

        assert similarities.shape == (10, 20)  # Mars x Earth comparisons
        assert (similarities >= -1).all() and (similarities <= 1).all()
        assert not torch.isnan(similarities).any()

    def test_find_analogs(self, analyzer, sample_mars_features, sample_earth_features):
        """Test finding Earth analogs for Mars features"""
        # RED: Will fail until analog finding is implemented
        analogs = analyzer.find_earth_analogs(sample_mars_features, sample_earth_features)

        assert isinstance(analogs, dict)
        assert len(analogs) <= 10  # At most one analog per Mars feature
        for mars_idx, earth_idx in analogs.items():
            assert 0 <= mars_idx < 10
            assert 0 <= earth_idx < 20

    def test_clustering_analysis(self, analyzer, sample_mars_features, sample_earth_features):
        """Test clustering for feature grouping"""
        # RED: Will fail until clustering is implemented
        mars_clusters = analyzer.cluster_features(sample_mars_features, n_clusters=3)
        earth_clusters = analyzer.cluster_features(sample_earth_features, n_clusters=5)

        assert len(mars_clusters) == 10
        assert len(earth_clusters) == 20
        assert all(0 <= cluster < 3 for cluster in mars_clusters)
        assert all(0 <= cluster < 5 for cluster in earth_clusters)

    def test_cross_planetary_comparison_report(self, analyzer, sample_mars_features, sample_earth_features):
        """Test generation of comparison report"""
        # RED: Will fail until report generation is implemented
        report = analyzer.generate_comparison_report(sample_mars_features, sample_earth_features)

        assert isinstance(report, dict)
        assert 'similarity_stats' in report
        assert 'analog_matches' in report
        assert 'cluster_analysis' in report
        assert 'confidence_scores' in report


class TestMarsLandingSiteOptimizer:
    """
    TDD Unit Tests for Mars Landing Site Optimization
    """

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing"""
        return MarsLandingSiteOptimizer(
            optimization_method='genetic_algorithm',
            population_size=100,
            max_generations=50
        )

    @pytest.fixture
    def sample_terrain_data(self):
        """Sample Mars terrain data"""
        return {
            'elevation': np.random.randn(100, 100),
            'slope': np.random.uniform(0, 45, (100, 100)),
            'roughness': np.random.uniform(0, 1, (100, 100)),
            'coordinates': np.mgrid[0:100, 0:100]
        }

    @pytest.fixture
    def mission_constraints(self):
        """Sample mission constraints"""
        return {
            'max_slope': 15.0,
            'min_flat_area': 100.0,  # square meters
            'safety_buffer': 50.0,   # meters
            'elevation_range': (-1000, 5000),  # meters
            'avoid_craters': True
        }

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly"""
        # RED: Will fail until MarsLandingSiteOptimizer is implemented
        assert optimizer.optimization_method == 'genetic_algorithm'
        assert optimizer.population_size == 100
        assert optimizer.max_generations == 50
        assert hasattr(optimizer, 'genetic_algorithm')
        assert hasattr(optimizer, 'fitness_evaluator')

    def test_fitness_evaluation(self, optimizer, sample_terrain_data, mission_constraints):
        """Test fitness function for landing sites"""
        # RED: Will fail until fitness function is implemented
        sample_sites = np.random.randint(0, 100, (10, 2))  # 10 candidate sites

        fitness_scores = optimizer.evaluate_fitness(
            sites=sample_sites,
            terrain_data=sample_terrain_data,
            constraints=mission_constraints
        )

        assert len(fitness_scores) == 10
        assert all(0 <= score <= 1 for score in fitness_scores)
        assert not np.isnan(fitness_scores).any()

    def test_constraint_checking(self, optimizer, sample_terrain_data, mission_constraints):
        """Test constraint validation for landing sites"""
        # RED: Will fail until constraint checking is implemented
        valid_site = np.array([50, 50])  # Center of terrain
        invalid_site = np.array([0, 0])   # Edge (likely invalid)

        is_valid_valid = optimizer.check_constraints(
            site=valid_site,
            terrain_data=sample_terrain_data,
            constraints=mission_constraints
        )

        is_valid_invalid = optimizer.check_constraints(
            site=invalid_site,
            terrain_data=sample_terrain_data,
            constraints=mission_constraints
        )

        assert isinstance(is_valid_valid, bool)
        assert isinstance(is_valid_invalid, bool)

    def test_genetic_algorithm_optimization(self, optimizer, sample_terrain_data, mission_constraints):
        """Test genetic algorithm optimization process"""
        # RED: Will fail until genetic algorithm is implemented
        optimal_sites = optimizer.optimize_landing_sites(
            terrain_data=sample_terrain_data,
            constraints=mission_constraints,
            num_sites=3
        )

        assert len(optimal_sites) == 3
        assert all(isinstance(site, np.ndarray) for site in optimal_sites)
        assert all(len(site) == 2 for site in optimal_sites)

        # Check that sites satisfy basic constraints
        for site in optimal_sites:
            assert optimizer.check_constraints(site, sample_terrain_data, mission_constraints)

    def test_multi_objective_optimization(self, optimizer, sample_terrain_data, mission_constraints):
        """Test multi-objective optimization (safety vs scientific value)"""
        # RED: Will fail until multi-objective optimization is implemented
        pareto_front = optimizer.multi_objective_optimize(
            terrain_data=sample_terrain_data,
            constraints=mission_constraints,
            objectives=['safety', 'scientific_value']
        )

        assert isinstance(pareto_front, list)
        assert len(pareto_front) > 0
        assert all('site' in solution and 'objectives' in solution for solution in pareto_front)


class TestAPIEndpoints:
    """
    TDD Unit Tests for FastAPI Endpoints
    """

    @pytest.fixture
    def client(self):
        """Create test client"""
        from mars_gis.main import app
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers"""
        return {"Authorization": "Bearer test_token"}

    def test_health_check_endpoint(self, client):
        """Test health check endpoint"""
        # RED: Will fail until endpoint is implemented
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "timestamp" in response.json()

    def test_models_list_endpoint(self, client, auth_headers):
        """Test models listing endpoint"""
        # RED: Will fail until endpoint is implemented
        response = client.get("/api/v1/models", headers=auth_headers)

        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        assert len(models) == 6  # Our 6 foundation models

        expected_models = [
            'foundation', 'multimodal', 'comparative',
            'optimization', 'self_supervised', 'planetary_scale'
        ]
        model_names = [model['name'] for model in models]
        assert all(name in model_names for name in expected_models)

    def test_model_inference_endpoint(self, client, auth_headers):
        """Test model inference endpoint"""
        # RED: Will fail until endpoint is implemented
        test_data = {
            "model_name": "foundation",
            "input_data": {
                "image": "base64_encoded_image_data",
                "coordinates": {"lat": -14.5684, "lon": 175.4726}
            }
        }

        response = client.post("/api/v1/models/infer",
                              json=test_data, headers=auth_headers)

        assert response.status_code == 200
        result = response.json()
        assert "predictions" in result
        assert "confidence" in result
        assert "processing_time" in result

    def test_terrain_analysis_endpoint(self, client, auth_headers):
        """Test terrain analysis endpoint"""
        # RED: Will fail until endpoint is implemented
        test_request = {
            "region": {
                "lat_min": -15.0, "lat_max": -14.0,
                "lon_min": 175.0, "lon_max": 176.0
            },
            "analysis_type": "slope_calculation"
        }

        response = client.post("/api/v1/terrain/analyze",
                              json=test_request, headers=auth_headers)

        assert response.status_code == 200
        result = response.json()
        assert "analysis_id" in result
        assert "status" in result
        assert result["status"] == "processing"

    def test_mission_planning_endpoint(self, client, auth_headers):
        """Test mission planning endpoint"""
        # RED: Will fail until endpoint is implemented
        mission_data = {
            "mission_name": "Test Mission",
            "target_coordinates": {"lat": -14.5684, "lon": 175.4726},
            "mission_duration": 687,
            "constraints": {
                "max_slope": 15.0,
                "min_elevation": -1000
            }
        }

        response = client.post("/api/v1/missions/plan",
                              json=mission_data, headers=auth_headers)

        assert response.status_code == 201
        result = response.json()
        assert "mission_id" in result
        assert "optimal_sites" in result
        assert "risk_assessment" in result

    def test_data_upload_endpoint(self, client, auth_headers):
        """Test data upload endpoint"""
        # RED: Will fail until endpoint is implemented
        test_file_data = b"mock_mars_image_data"

        response = client.post("/api/v1/data/upload",
                              files={"file": ("test_mars.tif", test_file_data, "image/tiff")},
                              headers=auth_headers)

        assert response.status_code == 201
        result = response.json()
        assert "file_id" in result
        assert "upload_status" in result
        assert result["upload_status"] == "success"

    def test_authentication_required(self, client):
        """Test endpoints require authentication"""
        # RED: Will fail until authentication middleware is implemented
        response = client.get("/api/v1/models")
        assert response.status_code == 401

        response = client.post("/api/v1/models/infer", json={})
        assert response.status_code == 401

    def test_rate_limiting(self, client, auth_headers):
        """Test API rate limiting"""
        # RED: Will fail until rate limiting is implemented
        # Make multiple rapid requests
        responses = []
        for _ in range(100):  # Exceed rate limit
            response = client.get("/api/v1/models", headers=auth_headers)
            responses.append(response)

        # Should have some 429 (Too Many Requests) responses
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes

    def test_input_validation(self, client, auth_headers):
        """Test input validation for endpoints"""
        # RED: Will fail until validation is implemented
        # Test invalid data
        invalid_data = {
            "model_name": "nonexistent_model",
            "input_data": {"invalid": "data"}
        }

        response = client.post("/api/v1/models/infer",
                              json=invalid_data, headers=auth_headers)

        assert response.status_code == 422  # Validation error
        error_detail = response.json()
        assert "detail" in error_detail


class TestDatabaseIntegration:
    """
    TDD Unit Tests for Database Operations
    """

    def test_database_connection(self, test_session):
        """Test database connection establishment"""
        # RED: Will fail until database connection is properly configured
        assert test_session is not None
        assert test_session.is_active

    def test_mars_data_storage(self, test_session):
        """Test storing Mars data in database"""
        # RED: Will fail until Mars data model is implemented
        from mars_gis.database.models import MarsData

        mars_data = MarsData(
            latitude=-14.5684,
            longitude=175.4726,
            elevation=1500.0,
            geological_type="crater",
            data_source="HiRISE"
        )

        test_session.add(mars_data)
        test_session.commit()

        # Verify data was stored
        stored_data = test_session.query(MarsData).filter_by(
            latitude=-14.5684, longitude=175.4726
        ).first()

        assert stored_data is not None
        assert stored_data.elevation == 1500.0
        assert stored_data.geological_type == "crater"

    def test_mission_data_storage(self, test_session):
        """Test storing mission data in database"""
        # RED: Will fail until Mission data model is implemented
        from mars_gis.database.models import Mission

        mission = Mission(
            name="Test Mars Mission",
            target_latitude=-14.5684,
            target_longitude=175.4726,
            mission_type="landing",
            status="planned"
        )

        test_session.add(mission)
        test_session.commit()

        # Verify mission was stored
        stored_mission = test_session.query(Mission).filter_by(
            name="Test Mars Mission"
        ).first()

        assert stored_mission is not None
        assert stored_mission.mission_type == "landing"
        assert stored_mission.status == "planned"

    def test_spatial_queries(self, test_session):
        """Test spatial database queries"""
        # RED: Will fail until spatial queries are implemented
        from mars_gis.database.models import MarsData
        from mars_gis.database.spatial import get_data_in_region

        # Add test data
        test_points = [
            MarsData(latitude=-14.5, longitude=175.4, elevation=1000),
            MarsData(latitude=-14.6, longitude=175.5, elevation=1100),
            MarsData(latitude=-14.7, longitude=175.6, elevation=1200),
        ]

        for point in test_points:
            test_session.add(point)
        test_session.commit()

        # Query data in region
        region_data = get_data_in_region(
            test_session,
            lat_min=-14.8, lat_max=-14.4,
            lon_min=175.3, lon_max=175.7
        )

        assert len(region_data) == 3
        assert all(point.elevation >= 1000 for point in region_data)


# Performance Tests
class TestPerformance:
    """Performance testing for critical components"""

    @pytest.mark.performance
    def test_model_inference_performance(self, mock_ai_model, sample_mars_image):
        """Test model inference performance"""
        import time

        # Warm up
        for _ in range(5):
            mock_ai_model.predict(sample_mars_image)

        # Measure performance
        start_time = time.time()
        for _ in range(100):
            result = mock_ai_model.predict(sample_mars_image)
        end_time = time.time()

        avg_inference_time = (end_time - start_time) / 100

        # Should be under 50ms per inference
        assert avg_inference_time < 0.05

    @pytest.mark.performance
    def test_database_query_performance(self, test_session, performance_thresholds):
        """Test database query performance"""
        import time

        # Insert test data
        from mars_gis.database.models import MarsData
        test_data = [
            MarsData(latitude=i*0.1, longitude=j*0.1, elevation=i*j)
            for i in range(100) for j in range(100)
        ]

        for data in test_data:
            test_session.add(data)
        test_session.commit()

        # Measure query performance
        start_time = time.time()
        results = test_session.query(MarsData).filter(
            MarsData.elevation > 500
        ).all()
        end_time = time.time()

        query_time = end_time - start_time

        # Should be under performance threshold
        assert query_time < performance_thresholds["api_response_time"]
        assert len(results) > 0


# Integration Tests
class TestSystemIntegration:
    """Integration tests for end-to-end workflows"""

    @pytest.mark.integration
    def test_complete_analysis_workflow(self, client, auth_headers):
        """Test complete Mars analysis workflow"""
        # 1. Upload data
        test_file_data = b"mock_mars_image_data"
        upload_response = client.post("/api/v1/data/upload",
                                     files={"file": ("test_mars.tif", test_file_data)},
                                     headers=auth_headers)
        assert upload_response.status_code == 201

        # 2. Run analysis
        file_id = upload_response.json()["file_id"]
        analysis_request = {
            "file_id": file_id,
            "analysis_types": ["terrain", "geological"]
        }

        analysis_response = client.post("/api/v1/analysis/run",
                                       json=analysis_request,
                                       headers=auth_headers)
        assert analysis_response.status_code == 200

        # 3. Get results
        analysis_id = analysis_response.json()["analysis_id"]
        results_response = client.get(f"/api/v1/analysis/{analysis_id}/results",
                                     headers=auth_headers)
        assert results_response.status_code == 200

    @pytest.mark.integration
    def test_mission_planning_workflow(self, client, auth_headers):
        """Test complete mission planning workflow"""
        # 1. Create mission plan
        mission_data = {
            "mission_name": "Integration Test Mission",
            "target_region": {
                "lat_min": -15.0, "lat_max": -14.0,
                "lon_min": 175.0, "lon_max": 176.0
            },
            "objectives": ["geological_survey", "sample_collection"]
        }

        plan_response = client.post("/api/v1/missions/create",
                                   json=mission_data,
                                   headers=auth_headers)
        assert plan_response.status_code == 201

        # 2. Optimize landing sites
        mission_id = plan_response.json()["mission_id"]
        optimize_response = client.post(f"/api/v1/missions/{mission_id}/optimize",
                                       headers=auth_headers)
        assert optimize_response.status_code == 200

        # 3. Get mission details
        details_response = client.get(f"/api/v1/missions/{mission_id}",
                                     headers=auth_headers)
        assert details_response.status_code == 200
        mission_details = details_response.json()
        assert "optimal_sites" in mission_details
        assert len(mission_details["optimal_sites"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
