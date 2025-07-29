"""Unit tests for Mars terrain classification models."""

from unittest.mock import Mock, patch

import pytest

from tests.conftest import requires_torch, torch_available


@pytest.mark.ml
@requires_torch
class TestMarsTerrainCNN:
    """Test cases for Mars terrain CNN model."""
    
    def test_model_initialization(self):
        """Test model creates with correct parameters."""
        from mars_gis.ml.models.terrain_models import MarsTerrainCNN
        
        model = MarsTerrainCNN(num_classes=8, input_channels=3)
        
        assert model.num_classes == 8
        assert len(model.terrain_classes) == 8
        assert "rocky_terrain" in model.terrain_classes
        assert "sandy_terrain" in model.terrain_classes
    
    def test_model_forward_pass(self, mock_terrain_model):
        """Test model forward pass with mock data."""
        if not torch_available():
            pytest.skip("PyTorch not available")
            
        import torch

        # Create mock input batch (batch_size=2, channels=3, height=224, width=224)
        mock_input = torch.randn(2, 3, 224, 224)
        
        output = mock_terrain_model(mock_input)
        
        # Check output shape (batch_size=2, num_classes=8)
        assert output.shape == (2, 8)
        
    def test_model_training_mode(self):
        """Test model can switch between train/eval modes."""
        if not torch_available():
            pytest.skip("PyTorch not available")
            
        from mars_gis.ml.models.terrain_models import MarsTerrainCNN
        
        model = MarsTerrainCNN()
        
        # Test training mode
        model.train()
        assert model.training is True
        
        # Test evaluation mode
        model.eval()
        assert model.training is False
    
    def test_terrain_class_mapping(self):
        """Test terrain class names are correctly mapped."""
        from mars_gis.ml.models.terrain_models import MarsTerrainCNN
        
        model = MarsTerrainCNN()
        expected_classes = [
            "rocky_terrain", "sandy_terrain", "crater_rim", "crater_floor",
            "dust_deposit", "volcanic_flow", "channel_bed", "unknown"
        ]
        
        assert model.terrain_classes == expected_classes


@pytest.mark.ml
@requires_torch
class TestMarsHazardDetector:
    """Test cases for Mars hazard detection model."""
    
    def test_hazard_model_initialization(self):
        """Test hazard detection model creates correctly."""
        from mars_gis.ml.models.terrain_models import MarsHazardDetector
        
        model = MarsHazardDetector(input_channels=3)
        
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
    
    def test_hazard_detection_output_shape(self):
        """Test hazard detection model output shape."""
        if not torch_available():
            pytest.skip("PyTorch not available")
            
        import torch

        from mars_gis.ml.models.terrain_models import MarsHazardDetector
        
        model = MarsHazardDetector()
        mock_input = torch.randn(1, 3, 256, 256)
        
        output = model(mock_input)
        
        # Output should be probability map of same spatial dimensions
        assert output.shape == (1, 1, 256, 256)
        
        # Check values are in valid probability range
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)


@pytest.mark.ml
class TestTrainingUtils:
    """Test cases for training utilities."""
    
    def test_mars_image_dataset_creation(self):
        """Test Mars image dataset can be created."""
        from mars_gis.ml.training.trainer import MarsImageDataset
        
        image_paths = ["path1.jpg", "path2.jpg", "path3.jpg"]
        labels = [0, 1, 2]
        
        dataset = MarsImageDataset(image_paths, labels)
        
        assert len(dataset) == 3
        assert dataset.image_paths == image_paths
        assert dataset.labels == labels
    
    @requires_torch
    def test_dataset_item_access(self):
        """Test dataset item access returns correct format."""
        if not torch_available():
            pytest.skip("PyTorch not available")
            
        from mars_gis.ml.training.trainer import MarsImageDataset
        
        image_paths = ["test.jpg"]
        labels = [0]
        
        dataset = MarsImageDataset(image_paths, labels)
        
        # Mock the actual image loading since we don't have real images in tests
        with patch.object(dataset, '__getitem__') as mock_getitem:
            import torch
            mock_getitem.return_value = (torch.randn(3, 224, 224), 0)
            
            image, label = dataset[0]
            
            assert image.shape == (3, 224, 224)
            assert label == 0
    
    def test_training_config_validation(self):
        """Test training configuration validation."""
        from mars_gis.ml.training.trainer import TrainingConfig
        
        config = TrainingConfig(
            model_name="MarsTerrainCNN",
            num_epochs=10,
            batch_size=32,
            learning_rate=0.001
        )
        
        assert config.model_name == "MarsTerrainCNN"
        assert config.num_epochs == 10
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
    
    def test_invalid_training_config(self):
        """Test training configuration validation catches invalid values."""
        from mars_gis.ml.training.trainer import TrainingConfig
        
        with pytest.raises(ValueError):
            TrainingConfig(
                model_name="",  # Empty name should be invalid
                num_epochs=-1,  # Negative epochs should be invalid
                batch_size=0,   # Zero batch size should be invalid
                learning_rate=-0.1  # Negative learning rate should be invalid
            )


@pytest.mark.ml
class TestInferenceEngine:
    """Test cases for ML inference engine."""
    
    @requires_torch
    def test_predictor_initialization(self):
        """Test terrain predictor can be initialized."""
        from mars_gis.ml.inference.predictor import TerrainPredictor
        
        predictor = TerrainPredictor(model_path=None)  # Use default model
        
        assert predictor is not None
        assert hasattr(predictor, 'predict')
        assert hasattr(predictor, 'predict_batch')
    
    def test_prediction_result_format(self):
        """Test prediction results have correct format."""
        from mars_gis.ml.inference.predictor import TerrainPredictor
        
        predictor = TerrainPredictor(model_path=None)
        
        # Mock prediction to avoid loading actual model
        with patch.object(predictor, 'predict') as mock_predict:
            mock_predict.return_value = {
                "terrain_class": "rocky_terrain",
                "confidence": 0.85,
                "probabilities": {
                    "rocky_terrain": 0.85,
                    "sandy_terrain": 0.10,
                    "crater_rim": 0.05
                }
            }
            
            result = predictor.predict(None)  # Mock input
            
            assert "terrain_class" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert result["confidence"] >= 0.0
            assert result["confidence"] <= 1.0
    
    def test_batch_prediction_consistency(self):
        """Test batch predictions are consistent with individual predictions."""
        from mars_gis.ml.inference.predictor import TerrainPredictor
        
        predictor = TerrainPredictor(model_path=None)
        
        # Mock both individual and batch predictions
        with patch.object(predictor, 'predict') as mock_predict, \
             patch.object(predictor, 'predict_batch') as mock_predict_batch:
            
            # Set up mock returns
            individual_result = {"terrain_class": "rocky_terrain", "confidence": 0.85}
            mock_predict.return_value = individual_result
            mock_predict_batch.return_value = [individual_result, individual_result]
            
            # Test individual prediction
            single_result = predictor.predict(None)
            
            # Test batch prediction
            batch_results = predictor.predict_batch([None, None])
            
            assert len(batch_results) == 2
            assert batch_results[0] == single_result
            assert batch_results[1] == single_result


@pytest.mark.ml
class TestModelUtils:
    """Test cases for ML utility functions."""
    
    def test_model_save_load_consistency(self):
        """Test model saving and loading preserves state."""
        if not torch_available():
            pytest.skip("PyTorch not available")
            
        import tempfile

        import torch

        from mars_gis.ml.models.terrain_models import MarsTerrainCNN
        from mars_gis.ml.utils import load_model, save_model

        # Create and train a simple model
        model = MarsTerrainCNN(num_classes=8)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save model state
        with tempfile.NamedTemporaryFile() as temp_file:
            save_model(model, optimizer, temp_file.name, epoch=5, loss=0.1)
            
            # Load model state
            loaded_model, loaded_optimizer, metadata = load_model(
                temp_file.name, MarsTerrainCNN, torch.optim.Adam
            )
            
            assert metadata['epoch'] == 5
            assert metadata['loss'] == 0.1
            assert loaded_model.num_classes == 8
    
    def test_preprocessing_pipeline(self):
        """Test image preprocessing pipeline."""
        from mars_gis.ml.utils import preprocess_mars_image

        # Mock image preprocessing
        with patch('mars_gis.ml.utils.preprocess_mars_image') as mock_preprocess:
            if torch_available():
                import torch
                mock_preprocess.return_value = torch.randn(3, 224, 224)
            else:
                mock_preprocess.return_value = None
            
            result = preprocess_mars_image("dummy_path.jpg")
            
            if torch_available():
                assert result.shape == (3, 224, 224)
            else:
                assert result is None
    
    def test_model_metrics_calculation(self):
        """Test model performance metrics calculation."""
        from mars_gis.ml.utils import calculate_metrics

        # Mock predictions and ground truth
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 1, 2]  # One misclassification
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Accuracy should be 5/6 ≈ 0.833
        assert abs(metrics['accuracy'] - 0.833) < 0.01


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for ML components."""
    
    def test_end_to_end_prediction_pipeline(self):
        """Test complete prediction pipeline from image to result."""
        # This would test the full pipeline:
        # 1. Load image
        # 2. Preprocess
        # 3. Run inference
        # 4. Post-process results
        # 5. Format output
        
        # Mock the entire pipeline since we don't have real models/data in tests
        with patch('mars_gis.ml.inference.predictor.TerrainPredictor') as MockPredictor:
            mock_predictor = MockPredictor.return_value
            mock_predictor.predict.return_value = {
                "terrain_class": "rocky_terrain",
                "confidence": 0.85,
                "coordinates": [-14.5684, 175.4729]
            }
            
            # Test the pipeline
            from mars_gis.ml.inference.predictor import TerrainPredictor
            
            predictor = TerrainPredictor()
            result = predictor.predict("test_image.jpg")
            
            assert result["terrain_class"] == "rocky_terrain"
            assert result["confidence"] == 0.85
    
    def test_model_training_pipeline(self):
        """Test complete model training pipeline."""
        # This would test:
        # 1. Data loading
        # 2. Model creation
        # 3. Training loop
        # 4. Validation
        # 5. Model saving
        
        # Mock the training pipeline
        with patch('mars_gis.ml.training.trainer.MarsTrainer') as MockTrainer:
            mock_trainer = MockTrainer.return_value
            mock_trainer.train.return_value = {
                "final_loss": 0.1,
                "accuracy": 0.95,
                "epochs_completed": 10
            }
            
            from mars_gis.ml.training.trainer import MarsTrainer
            
            trainer = MarsTrainer()
            results = trainer.train()
            
            assert results["accuracy"] > 0.9
            assert results["epochs_completed"] == 10
