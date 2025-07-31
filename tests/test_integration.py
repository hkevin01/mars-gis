"""
Mars GIS Integration Tests

Comprehensive tests verifying that all modules work together correctly.
"""

import logging
from typing import Any, Dict

import numpy as np

from mars_gis.models.comparative import ComparativePlanetaryAnalyzer

# Import all major components
from mars_gis.models.foundation import MarsEarthTransferModel
from mars_gis.models.multimodal import MultiModalMarsProcessor
from mars_gis.models.optimization import MarsLandingSiteOptimizer
from mars_gis.models.planetary_scale import PlanetaryScaleEmbeddingGenerator
from mars_gis.models.self_supervised import SelfSupervisedMarsLearning
from mars_gis.visualization.analysis_dashboard import MarsAnalysisDashboard
from mars_gis.visualization.interactive_mapping import InteractiveMarsMap
from mars_gis.visualization.mars_3d_globe import Mars3DGlobeGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarsGISIntegrationTester:
    """Integration tester for the complete Mars GIS platform."""
    
    def __init__(self):
        """Initialize the integration tester."""
        self.test_data = self._generate_test_data()
        self.results = {}
        
    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate test data for integration tests."""
        return {
            'mars_image': np.random.rand(256, 256, 3),
            'earth_image': np.random.rand(256, 256, 3),
            'elevation_data': np.random.rand(100, 100),
            'coordinates': (15.0, -30.0),  # lat, lon
            'region_size': (50, 50),
            'spectral_data': np.random.rand(128, 128, 10),
            'thermal_data': np.random.rand(64, 64)
        }
    
    def test_foundation_models_integration(self) -> bool:
        """Test that all foundation models work together."""
        logger.info("Testing foundation models integration...")
        
        try:
            # Test Earth-Mars transfer
            transfer_model = MarsEarthTransferModel()
            earth_features = transfer_model.extract_earth_features(
                self.test_data['earth_image']
            )
            mars_features = transfer_model.extract_mars_features(
                self.test_data['mars_image']
            )
            
            # Test multimodal processing
            multimodal = MultiModalMarsProcessor()
            multimodal_features = multimodal.process_multimodal_data(
                visual=self.test_data['mars_image'],
                spectral=self.test_data['spectral_data'],
                thermal=self.test_data['thermal_data']
            )
            
            # Test comparative analysis
            comparative = ComparativePlanetaryAnalyzer()
            comparative_results = comparative.compare_planetary_features(
                mars_features, earth_features
            )
            
            # Test landing site optimization
            optimizer = MarsLandingSiteOptimizer()
            optimization_results = optimizer.optimize_landing_sites(
                multimodal_features,
                self.test_data['elevation_data']
            )
            
            # Test self-supervised learning
            ssl_model = SelfSupervisedMarsLearning()
            ssl_features = ssl_model.learn_mars_representations(
                [self.test_data['mars_image']]
            )
            
            # Test planetary-scale embeddings
            embedding_gen = PlanetaryScaleEmbeddingGenerator()
            embeddings = embedding_gen.generate_embeddings(
                [self.test_data['mars_image']]
            )
            
            # Verify all outputs are valid
            assert earth_features is not None
            assert mars_features is not None
            assert multimodal_features is not None
            assert comparative_results is not None
            assert optimization_results is not None
            assert ssl_features is not None
            assert embeddings is not None
            
            self.results['foundation_models'] = True
            logger.info("✅ Foundation models integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Foundation models integration test failed: {e}")
            self.results['foundation_models'] = False
            return False
    
    def test_visualization_integration(self) -> bool:
        """Test that all visualization components work together."""
        logger.info("Testing visualization integration...")
        
        try:
            # Test 3D globe generation
            globe_gen = Mars3DGlobeGenerator()
            globe_data = globe_gen.generate_3d_globe(
                elevation_data=self.test_data['elevation_data'],
                color_scheme='realistic'
            )
            
            # Test interactive mapping
            interactive_map = InteractiveMarsMap()
            map_tiles = interactive_map.generate_map_tiles(
                center_lat=self.test_data['coordinates'][0],
                center_lon=self.test_data['coordinates'][1],
                zoom_level=5
            )
            
            # Test analysis dashboard
            dashboard = MarsAnalysisDashboard()
            dashboard_config = dashboard.initialize_dashboard()
            
            # Verify all outputs are valid
            assert globe_data is not None
            assert hasattr(globe_data, 'vertices')
            assert hasattr(globe_data, 'faces')
            assert map_tiles is not None
            assert dashboard_config is not None
            
            self.results['visualization'] = True
            logger.info("✅ Visualization integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Visualization integration test failed: {e}")
            self.results['visualization'] = False
            return False
    
    def test_end_to_end_workflow(self) -> bool:
        """Test a complete end-to-end Mars analysis workflow."""
        logger.info("Testing end-to-end workflow...")
        
        try:
            # Step 1: Process Mars data with foundation models
            transfer_model = MarsEarthTransferModel()
            mars_features = transfer_model.extract_mars_features(
                self.test_data['mars_image']
            )
            
            multimodal = MultiModalMarsProcessor()
            enhanced_features = multimodal.process_multimodal_data(
                visual=self.test_data['mars_image'],
                spectral=self.test_data['spectral_data'],
                thermal=self.test_data['thermal_data']
            )
            
            # Step 2: Optimize landing sites
            optimizer = MarsLandingSiteOptimizer()
            landing_sites = optimizer.optimize_landing_sites(
                enhanced_features,
                self.test_data['elevation_data']
            )
            
            # Step 3: Visualize results
            globe_gen = Mars3DGlobeGenerator()
            globe_visualization = globe_gen.generate_3d_globe(
                elevation_data=self.test_data['elevation_data'],
                color_scheme='scientific'
            )
            
            interactive_map = InteractiveMarsMap()
            interactive_map.add_landing_sites(landing_sites)
            
            # Step 4: Create analysis dashboard
            dashboard = MarsAnalysisDashboard()
            dashboard.start_real_time_analysis()
            
            # Verify workflow completion
            assert mars_features is not None
            assert enhanced_features is not None
            assert landing_sites is not None
            assert globe_visualization is not None
            
            self.results['end_to_end'] = True
            logger.info("✅ End-to-end workflow test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ End-to-end workflow test failed: {e}")
            self.results['end_to_end'] = False
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks for key operations."""
        logger.info("Testing performance benchmarks...")
        
        import time
        
        try:
            # Benchmark foundation model processing
            start_time = time.time()
            transfer_model = MarsEarthTransferModel()
            _ = transfer_model.extract_mars_features(
                self.test_data['mars_image']
            )
            foundation_time = time.time() - start_time
            
            # Benchmark visualization generation
            start_time = time.time()
            globe_gen = Mars3DGlobeGenerator()
            _ = globe_gen.generate_3d_globe(
                elevation_data=self.test_data['elevation_data']
            )
            visualization_time = time.time() - start_time
            
            # Benchmark dashboard initialization
            start_time = time.time()
            dashboard = MarsAnalysisDashboard()
            _ = dashboard.initialize_dashboard()
            dashboard_time = time.time() - start_time
            
            # Check performance thresholds
            # Should complete in under 30 seconds
            assert foundation_time < 30.0
            # Should complete in under 15 seconds
            assert visualization_time < 15.0
            assert dashboard_time < 5.0  # Should complete in under 5 seconds
            
            self.results['performance'] = {
                'foundation_time': foundation_time,
                'visualization_time': visualization_time,
                'dashboard_time': dashboard_time
            }
            
            logger.info("✅ Performance benchmark test passed")
            logger.info(f"  Foundation model: {foundation_time:.2f}s")
            logger.info(f"  Visualization: {visualization_time:.2f}s")
            logger.info(f"  Dashboard: {dashboard_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark test failed: {e}")
            self.results['performance'] = False
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("Starting Mars GIS integration tests...")
        
        tests = [
            self.test_foundation_models_integration,
            self.test_visualization_integration,
            self.test_end_to_end_workflow,
            self.test_performance_benchmarks
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
        
        success_rate = (passed / total) * 100
        
        logger.info(f"\n{'='*60}")
        logger.info("INTEGRATION TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Tests passed: {passed}/{total} ({success_rate:.1f}%)")
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name:20}: {status}")
        
        if success_rate == 100:
            logger.info("\n🎉 All integration tests passed! Mars GIS is ready!")
        else:
            warning_msg = f"\n⚠️  Some tests failed. " \
                         f"Success rate: {success_rate:.1f}%"
            logger.warning(warning_msg)
        
        return {
            'passed': passed,
            'total': total,
            'success_rate': success_rate,
            'results': self.results
        }


def run_integration_tests():
    """Main function to run integration tests."""
    tester = MarsGISIntegrationTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    results = run_integration_tests()
    exit(0 if results['success_rate'] == 100 else 1)
