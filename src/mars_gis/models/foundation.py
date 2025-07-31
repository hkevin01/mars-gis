"""
Earth-Mars Transfer Learning Foundation Models

This module implements AlphaEarth-inspired architecture for cross-planetary
transfer learning, enabling Mars analysis using Earth geological knowledge.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MarsSpectraConfig:
    """Configuration for Mars-specific spectral processing."""
    mars_spectral_bands: int = 12
    earth_spectral_bands: int = 10
    embedding_dim: int = 64
    adapter_hidden_dim: int = 128


class MarsSpecificAdapter(nn.Module):
    """
    Adapter module for converting Earth foundation model features
    to Mars-specific representations.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        mars_spectral_bands: int = 12,
        output_dim: int = 128,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mars_spectral_bands = mars_spectral_bands
        self.output_dim = output_dim
        
        # Mars atmospheric correction layers
        self.atmospheric_correction = nn.Sequential(
            nn.Linear(mars_spectral_bands, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, mars_spectral_bands)
        )
        
        # Cross-domain adaptation network
        self.adaptation_network = nn.Sequential(
            nn.Linear(input_dim + mars_spectral_bands, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Mars-specific geological feature enhancer
        self.geological_enhancer = nn.Sequential(
            nn.Conv2d(output_dim, output_dim * 2, 3, padding=1),
            nn.BatchNorm2d(output_dim * 2),
            nn.ReLU(),
            nn.Conv2d(output_dim * 2, output_dim, 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        )
        
    def forward(
        self,
        mars_imagery: torch.Tensor,
        earth_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Mars adapter.
        
        Args:
            mars_imagery: Mars spectral imagery tensor
            earth_features: Optional Earth foundation model features
            
        Returns:
            Mars-adapted feature representation
        """
        batch_size = mars_imagery.shape[0]
        
        # Apply atmospheric correction specific to Mars
        corrected_mars = self.atmospheric_correction(mars_imagery)
        
        if earth_features is not None:
            # Fuse Earth knowledge with Mars observations
            if earth_features.dim() == 2:
                earth_features = earth_features.unsqueeze(-1).unsqueeze(-1)
            earth_features_flat = earth_features.view(batch_size, -1)
            mars_features_flat = corrected_mars.view(batch_size, -1)
            
            combined_features = torch.cat([
                earth_features_flat, mars_features_flat
            ], dim=1)
            adapted_features = self.adaptation_network(combined_features)
        else:
            # Mars-only processing
            mars_features_flat = corrected_mars.view(batch_size, -1)
            # Pad to match expected input size
            padding_size = self.input_dim
            padded_features = torch.cat([
                torch.zeros(
                    batch_size, padding_size, device=mars_imagery.device
                ),
                mars_features_flat
            ], dim=1)
            adapted_features = self.adaptation_network(padded_features)
        
        # Reshape for geological enhancement if needed
        if adapted_features.dim() == 2:
            # Assume square spatial dimensions
            feature_size = adapted_features.shape[1]
            spatial_dim = int(np.sqrt(feature_size // self.output_dim))
            expected_size = spatial_dim * spatial_dim * self.output_dim
            if expected_size == feature_size:
                adapted_features = adapted_features.view(
                    batch_size, self.output_dim, spatial_dim, spatial_dim
                )
                enhanced_features = self.geological_enhancer(adapted_features)
                return enhanced_features
        
        return adapted_features


class EarthFoundationEncoder(nn.Module):
    """
    Earth foundation model encoder inspired by AlphaEarth architecture.
    """
    
    def __init__(self, pretrained_path: Optional[str] = None):
        super().__init__()
        
        # Vision Transformer backbone (AlphaEarth-inspired)
        self.patch_embed = nn.Conv2d(10, 768, kernel_size=16, stride=16)
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, 768))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        
        # Earth-specific geological knowledge embedding
        self.earth_geo_embedding = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        
        if pretrained_path and Path(pretrained_path).exists():
            self.load_pretrained_weights(pretrained_path)
            
    def load_pretrained_weights(self, path: str):
        """Load pretrained Earth foundation model weights."""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except Exception as e:
            msg = f"Warning: Could not load pretrained weights " \
                  f"from {path}: {e}"
            print(msg)
    
    def forward(self, earth_imagery: torch.Tensor) -> torch.Tensor:
        """
        Encode Earth imagery into geological feature representations.
        
        Args:
            earth_imagery: Earth multispectral imagery (B, 10, H, W)
            
        Returns:
            Earth geological feature embeddings (B, 64)
        """
        B, C, H, W = earth_imagery.shape
        
        # Patch embedding
        x = self.patch_embed(earth_imagery)  # (B, 768, H//16, W//16)
        x = x.flatten(2).transpose(1, 2)  # (B, N, 768)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Transformer encoding
        x = self.transformer(x)  # (B, N, 768)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, 768)
        
        # Earth geological embedding
        geo_features = self.earth_geo_embedding(x)  # (B, 64)
        
        return geo_features


class MarsEarthTransferModel(nn.Module):
    """
    Main Earth-Mars transfer learning foundation model.
    Implements AlphaEarth-inspired architecture for Mars analysis.
    """
    
    def __init__(
        self,
        earth_pretrained_path: Optional[str] = None,
        config: Optional[MarsSpectraConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = MarsSpectraConfig()
        
        self.config = config
        
        # Load AlphaEarth-inspired Earth encoder
        self.earth_encoder = EarthFoundationEncoder(earth_pretrained_path)
        
        # Mars-specific adaptation layer
        self.mars_adapter = MarsSpecificAdapter(
            input_dim=config.embedding_dim,
            mars_spectral_bands=config.mars_spectral_bands,
            output_dim=config.adapter_hidden_dim
        )
        
        # Cross-planetary geological similarity network
        self.similarity_network = nn.Sequential(
            nn.Linear(config.adapter_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Mars terrain classification head
        self.terrain_classifier = nn.Sequential(
            nn.Linear(config.adapter_hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 8)  # 8 Mars terrain types
        )
        
    def extract_earth_features(self, earth_imagery: np.ndarray) -> np.ndarray:
        """Extract features from Earth imagery."""
        with torch.no_grad():
            earth_tensor = torch.from_numpy(earth_imagery).float()
            if earth_tensor.dim() == 3:
                earth_tensor = earth_tensor.unsqueeze(0)
            features = self.earth_encoder(earth_tensor)
            return features.cpu().numpy()
    
    def extract_mars_features(self, mars_imagery: np.ndarray) -> np.ndarray:
        """Extract features from Mars imagery."""
        with torch.no_grad():
            mars_tensor = torch.from_numpy(mars_imagery).float()
            if mars_tensor.dim() == 3:
                mars_tensor = mars_tensor.unsqueeze(0)
            features = self.mars_adapter(mars_tensor)
            if features.dim() > 2:
                features = F.adaptive_avg_pool2d(features, 1)
                features = features.squeeze(-1).squeeze(-1)
            return features.cpu().numpy()
    
    def forward(
        self,
        mars_imagery: torch.Tensor,
        earth_reference: Optional[torch.Tensor] = None,
        task: str = "classification"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Earth-Mars foundation model.
        """
        outputs = {}
        
        # Process Earth reference if available
        earth_features = None
        if earth_reference is not None:
            earth_features = self.earth_encoder(earth_reference)
            outputs['earth_features'] = earth_features
        
        # Mars adaptation
        mars_features = self.mars_adapter(mars_imagery, earth_features)
        outputs['mars_features'] = mars_features
        
        # Task-specific outputs
        if task in ["classification", "all"]:
            # Flatten features for classification if needed
            if mars_features.dim() > 2:
                mars_features_flat = mars_features.view(
                    mars_features.shape[0], -1
                )
                # Use adaptive pooling to get fixed size
                mars_features_flat = F.adaptive_avg_pool1d(
                    mars_features_flat.unsqueeze(1),
                    self.config.adapter_hidden_dim
                ).squeeze(1)
            else:
                mars_features_flat = mars_features
                
            terrain_logits = self.terrain_classifier(mars_features_flat)
            outputs['terrain_classification'] = terrain_logits
        
        if task in ["similarity", "all"] and earth_features is not None:
            if mars_features.dim() > 2:
                mars_features_flat = mars_features.view(
                    mars_features.shape[0], -1
                )
                mars_features_flat = F.adaptive_avg_pool1d(
                    mars_features_flat.unsqueeze(1),
                    self.config.adapter_hidden_dim
                ).squeeze(1)
            else:
                mars_features_flat = mars_features
                
            similarity_score = self.similarity_network(mars_features_flat)
            outputs['earth_mars_similarity'] = similarity_score
        
        return outputs


def create_mars_earth_transfer_model(
    pretrained_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> MarsEarthTransferModel:
    """
    Factory function to create Earth-Mars foundation model.
    """
    config = MarsSpectraConfig()
    model = MarsEarthTransferModel(pretrained_path, config)
    model.to(device)
    return model
