"""Inference utilities for Mars ML models."""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    NUMPY_AVAILABLE = False
    torch = F = np = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


class MarsModelPredictor:
    """Unified predictor for Mars ML models."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize Mars model predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on (auto/cuda/cpu)
        """
        self.model_path = model_path
        self.models = {}
        self.device = self._setup_device(device)
        
        # Terrain classification labels
        self.terrain_labels = [
            "plains", "craters", "ridges", "valleys",
            "dunes", "polar_ice", "volcanic", "rocky"
        ]
        
        # Hazard detection labels
        self.hazard_labels = ["safe", "hazardous"]
        
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if not TORCH_AVAILABLE:
            return "cpu"
            
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_terrain_model(self, model_path: str) -> bool:
        """Load terrain classification model."""
        if not TORCH_AVAILABLE:
            print("PyTorch not available for model loading")
            return False
        
        try:
            from ..models.terrain_models import MarsTerrainCNN
            
            model = MarsTerrainCNN()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            self.models["terrain"] = model
            print(f"Terrain model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load terrain model: {e}")
            return False
    
    def load_hazard_model(self, model_path: str) -> bool:
        """Load hazard detection model."""
        if not TORCH_AVAILABLE:
            print("PyTorch not available for model loading")
            return False
        
        try:
            from ..models.terrain_models import MarsHazardDetector
            
            model = MarsHazardDetector()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            self.models["hazard"] = model
            print(f"Hazard model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load hazard model: {e}")
            return False
    
    def load_atmosphere_model(self, model_path: str) -> bool:
        """Load atmosphere analysis model."""
        if not TORCH_AVAILABLE:
            print("PyTorch not available for model loading")
            return False
        
        try:
            from ..models.terrain_models import MarsAtmosphereAnalyzer
            
            model = MarsAtmosphereAnalyzer()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            self.models["atmosphere"] = model
            print(f"Atmosphere model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load atmosphere model: {e}")
            return False
    
    def preprocess_image(self, image_path: str) -> Optional[Any]:
        """Preprocess image for model inference."""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            if PIL_AVAILABLE:
                # Load and preprocess real image
                image = Image.open(image_path).convert('RGB')
                image = image.resize((224, 224))
                
                # Convert to tensor
                image_tensor = torch.from_numpy(
                    np.array(image).transpose(2, 0, 1)
                ).float() / 255.0
            else:
                # Create dummy tensor for demonstration
                image_tensor = torch.randn(3, 224, 224)
            
            # Normalize (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            # Add batch dimension
            return image_tensor.unsqueeze(0).to(self.device)
            
        except Exception as e:
            print(f"Image preprocessing failed: {e}")
            return None
    
    def predict_terrain(
        self,
        image_input: Union[str, Any],
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Predict terrain type from Mars image.
        
        Args:
            image_input: Path to image or preprocessed tensor
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Prediction results dictionary
        """
        if "terrain" not in self.models:
            return {"error": "Terrain model not loaded"}
        
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        try:
            # Preprocess input
            if isinstance(image_input, str):
                image_tensor = self.preprocess_image(image_input)
                if image_tensor is None:
                    return {"error": "Image preprocessing failed"}
            else:
                image_tensor = image_input
            
            # Run inference
            with torch.no_grad():
                outputs = self.models["terrain"](image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            results = {
                "predicted_class": self.terrain_labels[predicted_class],
                "class_index": predicted_class,
                "confidence": confidence,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if return_probabilities:
                results["probabilities"] = {
                    label: probabilities[0, i].item()
                    for i, label in enumerate(self.terrain_labels)
                }
            
            return results
            
        except Exception as e:
            return {"error": f"Terrain prediction failed: {e}"}
    
    def predict_hazards(
        self,
        image_input: Union[str, Any],
        safety_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Predict landing hazards from Mars image.
        
        Args:
            image_input: Path to image or preprocessed tensor
            safety_threshold: Minimum safety score for safe landing
            
        Returns:
            Hazard prediction results
        """
        if "hazard" not in self.models:
            return {"error": "Hazard model not loaded"}
        
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        try:
            # Preprocess input
            if isinstance(image_input, str):
                image_tensor = self.preprocess_image(image_input)
                if image_tensor is None:
                    return {"error": "Image preprocessing failed"}
            else:
                image_tensor = image_input
            
            # Run inference
            with torch.no_grad():
                hazard_logits, safety_score = self.models["hazard"](image_tensor)
                hazard_probs = F.softmax(hazard_logits, dim=1)
                hazard_class = torch.argmax(hazard_probs, dim=1).item()
                safety_value = safety_score.item()
            
            is_safe = safety_value >= safety_threshold
            
            results = {
                "hazard_detected": hazard_class == 1,
                "hazard_class": self.hazard_labels[hazard_class],
                "hazard_confidence": hazard_probs[0, hazard_class].item(),
                "safety_score": safety_value,
                "is_safe_landing": is_safe,
                "safety_threshold": safety_threshold,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return results
            
        except Exception as e:
            return {"error": f"Hazard prediction failed: {e}"}
    
    def predict_atmosphere(
        self,
        atmospheric_data: List[float],
        sequence_length: int = 24
    ) -> Dict[str, Any]:
        """
        Predict atmospheric conditions and dust storms.
        
        Args:
            atmospheric_data: Time series of atmospheric measurements
            sequence_length: Length of input sequence
            
        Returns:
            Atmospheric prediction results
        """
        if "atmosphere" not in self.models:
            return {"error": "Atmosphere model not loaded"}
        
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        try:
            # Prepare input tensor
            if len(atmospheric_data) < sequence_length:
                # Pad with zeros if needed
                padded_data = atmospheric_data + [0.0] * (
                    sequence_length - len(atmospheric_data)
                )
            else:
                padded_data = atmospheric_data[-sequence_length:]
            
            input_tensor = torch.tensor(
                padded_data, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(-1).to(self.device)
            
            # Run inference
            with torch.no_grad():
                dust_storm_prob, visibility_pred = self.models["atmosphere"](
                    input_tensor
                )
                
                dust_storm_risk = torch.sigmoid(dust_storm_prob).item()
                predicted_visibility = visibility_pred.item()
            
            results = {
                "dust_storm_probability": dust_storm_risk,
                "dust_storm_risk": "high" if dust_storm_risk > 0.7 else 
                                  "medium" if dust_storm_risk > 0.4 else "low",
                "predicted_visibility_km": max(0, predicted_visibility),
                "atmospheric_conditions": (
                    "poor" if dust_storm_risk > 0.6 else
                    "moderate" if dust_storm_risk > 0.3 else "good"
                ),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return results
            
        except Exception as e:
            return {"error": f"Atmosphere prediction failed: {e}"}
    
    def batch_predict_terrain(
        self,
        image_paths: List[str],
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """Batch prediction for multiple terrain images."""
        if "terrain" not in self.models:
            return [{"error": "Terrain model not loaded"}] * len(image_paths)
        
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_results = []
            
            for path in batch_paths:
                result = self.predict_terrain(path, return_probabilities=False)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def analyze_mars_scene(
        self,
        image_path: str,
        include_atmosphere: bool = False,
        atmospheric_data: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive Mars scene analysis.
        
        Args:
            image_path: Path to Mars image
            include_atmosphere: Whether to include atmospheric analysis
            atmospheric_data: Atmospheric measurements for analysis
            
        Returns:
            Complete scene analysis results
        """
        analysis = {
            "image_path": image_path,
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "terrain_analysis": {},
            "hazard_analysis": {},
            "atmospheric_analysis": {}
        }
        
        # Terrain analysis
        if "terrain" in self.models:
            analysis["terrain_analysis"] = self.predict_terrain(image_path)
        
        # Hazard analysis
        if "hazard" in self.models:
            analysis["hazard_analysis"] = self.predict_hazards(image_path)
        
        # Atmospheric analysis
        if include_atmosphere and "atmosphere" in self.models and atmospheric_data:
            analysis["atmospheric_analysis"] = self.predict_atmosphere(
                atmospheric_data
            )
        
        # Generate summary
        terrain_safe = analysis["terrain_analysis"].get("predicted_class") in [
            "plains", "valleys"
        ]
        hazard_safe = analysis["hazard_analysis"].get("is_safe_landing", False)
        
        analysis["summary"] = {
            "overall_safety": "safe" if terrain_safe and hazard_safe else "unsafe",
            "terrain_suitable": terrain_safe,
            "landing_safe": hazard_safe,
            "recommendation": (
                "Suitable for landing operations" if terrain_safe and hazard_safe
                else "Caution advised - detailed survey recommended"
            )
        }
        
        return analysis


def create_inference_pipeline(
    models_dir: Path,
    config: Optional[Dict[str, Any]] = None
) -> MarsModelPredictor:
    """
    Create complete inference pipeline.
    
    Args:
        models_dir: Directory containing trained models
        config: Optional configuration parameters
        
    Returns:
        Configured predictor instance
    """
    predictor = MarsModelPredictor()
    
    # Load available models
    terrain_model = models_dir / "terrain_model.pth"
    if terrain_model.exists():
        predictor.load_terrain_model(str(terrain_model))
    
    hazard_model = models_dir / "hazard_model.pth"
    if hazard_model.exists():
        predictor.load_hazard_model(str(hazard_model))
    
    atmosphere_model = models_dir / "atmosphere_model.pth"
    if atmosphere_model.exists():
        predictor.load_atmosphere_model(str(atmosphere_model))
    
    return predictor


def batch_process_mars_images(
    image_directory: Path,
    output_file: Path,
    predictor: MarsModelPredictor
) -> bool:
    """
    Process batch of Mars images for analysis.
    
    Args:
        image_directory: Directory containing Mars images
        output_file: Path to save results JSON
        predictor: Configured predictor instance
        
    Returns:
        Success status
    """
    try:
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        image_paths = [
            str(p) for p in image_directory.glob("*")
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            print(f"No images found in {image_directory}")
            return False
        
        print(f"Processing {len(image_paths)} images...")
        
        results = []
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            analysis = predictor.analyze_mars_scene(image_path)
            results.append(analysis)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return False
