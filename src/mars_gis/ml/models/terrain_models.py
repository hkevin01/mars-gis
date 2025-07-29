"""Mars terrain classification models using PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

try:
    import torchvision.models as models
    from torchvision.transforms import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    models = None
    transforms = None


class MarsTerrainCNN(nn.Module):
    """Convolutional Neural Network for Mars terrain classification."""
    
    def __init__(
        self, 
        num_classes: int = 8,
        input_channels: int = 3,
        dropout_rate: float = 0.5
    ):
        """
        Initialize Mars terrain classification model.
        
        Args:
            num_classes: Number of terrain classes to predict
            input_channels: Number of input channels (RGB=3, multispectral>3)
            dropout_rate: Dropout rate for regularization
        """
        super(MarsTerrainCNN, self).__init__()
        
        self.num_classes = num_classes
        self.terrain_classes = [
            "rocky_terrain", "sandy_terrain", "crater_rim", "crater_floor",
            "dust_deposit", "volcanic_flow", "channel_bed", "unknown"
        ]
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Convolutional layers with activation and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(F.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def predict_terrain_type(self, x: torch.Tensor) -> List[str]:
        """Predict terrain types from input tensor."""
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            
            return [self.terrain_classes[idx] for idx in predicted.cpu().numpy()]


class MarsHazardDetector(nn.Module):
    """Neural network for Mars landing hazard detection."""
    
    def __init__(self, input_channels: int = 3):
        """
        Initialize hazard detection model.
        
        Args:
            input_channels: Number of input channels
        """
        super(MarsHazardDetector, self).__init__()
        
        self.hazard_types = [
            "safe_zone", "steep_slope", "boulder_field", 
            "crater_edge", "dust_storm", "rough_terrain"
        ]
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
        )
        
        # Detection head for hazard classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(self.hazard_types))
        )
        
        # Regression head for safety score
        self.safety_scorer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output safety score between 0 and 1
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning hazard classification and safety score."""
        features = self.backbone(x)
        hazard_logits = self.classifier(features)
        safety_score = self.safety_scorer(features)
        
        return hazard_logits, safety_score
    
    def assess_landing_safety(
        self, 
        x: torch.Tensor
    ) -> Dict[str, Any]:
        """Assess landing safety for input imagery."""
        with torch.no_grad():
            hazard_logits, safety_scores = self.forward(x)
            hazard_probs = F.softmax(hazard_logits, dim=1)
            
            # Get most likely hazard type
            _, predicted_hazards = torch.max(hazard_probs, 1)
            
            results = []
            for i in range(x.size(0)):
                hazard_type = self.hazard_types[predicted_hazards[i]]
                safety_score = safety_scores[i].item()
                hazard_confidence = hazard_probs[i][predicted_hazards[i]].item()
                
                results.append({
                    "hazard_type": hazard_type,
                    "safety_score": safety_score,
                    "hazard_confidence": hazard_confidence,
                    "recommended_action": self._get_recommendation(
                        hazard_type, safety_score
                    )
                })
            
            return results
    
    def _get_recommendation(self, hazard_type: str, safety_score: float) -> str:
        """Get landing recommendation based on hazard type and safety score."""
        if safety_score > 0.8:
            return "safe_to_land"
        elif safety_score > 0.6:
            return "proceed_with_caution"
        elif hazard_type in ["steep_slope", "boulder_field", "crater_edge"]:
            return "avoid_area"
        else:
            return "requires_detailed_analysis"


class MarsAtmosphereAnalyzer(nn.Module):
    """LSTM model for Mars atmospheric analysis and dust storm prediction."""
    
    def __init__(
        self, 
        input_features: int = 10,
        hidden_size: int = 128,
        num_layers: int = 3,
        sequence_length: int = 24
    ):
        """
        Initialize atmospheric analyzer.
        
        Args:
            input_features: Number of atmospheric measurement features
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            sequence_length: Length of input time sequences
        """
        super(MarsAtmosphereAnalyzer, self).__init__()
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Atmospheric parameters
        self.feature_names = [
            "temperature", "pressure", "humidity", "wind_speed",
            "wind_direction", "dust_opacity", "co2_concentration",
            "solar_radiation", "surface_albedo", "atmospheric_density"
        ]
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_features, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Output layers
        self.dust_storm_predictor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.weather_forecaster = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, input_features)  # Predict next timestep
        )
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for atmospheric analysis."""
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Predictions
        dust_storm_prob = self.dust_storm_predictor(last_output)
        weather_forecast = self.weather_forecaster(last_output)
        
        return dust_storm_prob, weather_forecast
    
    def predict_dust_storm(
        self, 
        atmospheric_sequence: torch.Tensor
    ) -> Dict[str, float]:
        """Predict dust storm probability from atmospheric data."""
        with torch.no_grad():
            dust_prob, weather_pred = self.forward(atmospheric_sequence)
            
            return {
                "dust_storm_probability": dust_prob.squeeze().item(),
                "risk_level": self._get_risk_level(dust_prob.squeeze().item()),
                "predicted_weather": {
                    feature: weather_pred[0][i].item()
                    for i, feature in enumerate(self.feature_names)
                }
            }
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert dust storm probability to risk level."""
        if probability > 0.8:
            return "extreme"
        elif probability > 0.6:
            return "high"
        elif probability > 0.4:
            return "moderate"
        elif probability > 0.2:
            return "low"
        else:
            return "minimal"


# Model factory functions
def create_terrain_classifier(
    num_classes: int = 8,
    pretrained: bool = False
) -> MarsTerrainCNN:
    """Create a Mars terrain classification model."""
    model = MarsTerrainCNN(num_classes=num_classes)
    
    if pretrained and TORCHVISION_AVAILABLE:
        # Load pretrained weights if available
        # This would typically load from a saved checkpoint
        print("Note: Pretrained weights not yet available for Mars terrain model")
    
    return model


def create_hazard_detector() -> MarsHazardDetector:
    """Create a Mars landing hazard detection model."""
    return MarsHazardDetector()


def create_atmosphere_analyzer(
    input_features: int = 10,
    sequence_length: int = 24
) -> MarsAtmosphereAnalyzer:
    """Create a Mars atmospheric analysis model."""
    return MarsAtmosphereAnalyzer(
        input_features=input_features,
        sequence_length=sequence_length
    )


# Utility functions for model management
def save_model(model: nn.Module, filepath: str, metadata: Dict[str, Any] = None):
    """Save PyTorch model with metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "metadata": metadata or {}
    }
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str, model_class: nn.Module) -> nn.Module:
    """Load PyTorch model from checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu")
    
    model = model_class
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model
