"""
Earth-Mars Transfer Learning Foundation Models

This module implements AlphaEarth-inspired architecture for cross-planetary
transfer learning, enabling Mars analysis using Earth geological knowledge.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mars_gis.core.types import EarthCoordinate, MarsCoordinate


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
            
            combined_features = torch.cat([earth_features_flat, mars_features_flat], dim=1)
            adapted_features = self.adaptation_network(combined_features)
        else:
            # Mars-only processing
            mars_features_flat = corrected_mars.view(batch_size, -1)
            # Pad to match expected input size
            padding_size = self.input_dim
            padded_features = torch.cat([
                torch.zeros(batch_size, padding_size, device=mars_imagery.device),
                mars_features_flat
            ], dim=1)
            adapted_features = self.adaptation_network(padded_features)
        
        # Reshape for geological enhancement if needed
        if adapted_features.dim() == 2:
            # Assume square spatial dimensions
            spatial_dim = int(np.sqrt(adapted_features.shape[1] // self.output_dim))
            if spatial_dim * spatial_dim * self.output_dim == adapted_features.shape[1]:
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
            print(f"Warning: Could not load pretrained weights from {path}: {e}")
    
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


class EarthMarsFoundationModel(nn.Module):
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
        
    def forward(
        self,
        mars_imagery: torch.Tensor,
        earth_reference: Optional[torch.Tensor] = None,
        task: str = "classification"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Earth-Mars foundation model.
        
        Args:
            mars_imagery: Mars spectral imagery (B, mars_spectral_bands, H, W)
            earth_reference: Optional Earth reference imagery (B, 10, H, W)
            task: Task type ("classification", "similarity", "embedding")
            
        Returns:
            Dictionary containing task-specific outputs
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
                mars_features_flat = mars_features.view(mars_features.shape[0], -1)
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
                mars_features_flat = mars_features.view(mars_features.shape[0], -1)
                mars_features_flat = F.adaptive_avg_pool1d(
                    mars_features_flat.unsqueeze(1), 
                    self.config.adapter_hidden_dim
                ).squeeze(1)
            else:
                mars_features_flat = mars_features
                
            similarity_score = self.similarity_network(mars_features_flat)
            outputs['earth_mars_similarity'] = similarity_score
        
        return outputs
    
    def get_mars_embedding(self, mars_imagery: torch.Tensor) -> torch.Tensor:
        """
        Get Mars embedding for similarity search and analysis.
        
        Args:
            mars_imagery: Mars spectral imagery
            
        Returns:
            Normalized Mars embedding vector
        """
        with torch.no_grad():
            mars_features = self.mars_adapter(mars_imagery)
            if mars_features.dim() > 2:
                # Global average pooling for spatial features
                mars_embedding = F.adaptive_avg_pool2d(mars_features, 1).squeeze(-1).squeeze(-1)
            else:
                mars_embedding = mars_features
            
            # L2 normalization for cosine similarity
            mars_embedding = F.normalize(mars_embedding, p=2, dim=1)
            
        return mars_embedding
    
    def find_earth_analogs(
        self,
        mars_region: torch.Tensor,
        earth_database: List[Tuple[torch.Tensor, Dict]],
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Find Earth analog regions for given Mars area.
        
        Args:
            mars_region: Mars imagery tensor
            earth_database: List of (earth_imagery, metadata) tuples
            top_k: Number of top analogs to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of Earth analog dictionaries with similarity scores
        """
        mars_embedding = self.get_mars_embedding(mars_region)
        
        analogs = []
        for earth_imagery, metadata in earth_database:
            earth_features = self.earth_encoder(earth_imagery.unsqueeze(0))
            
            # Calculate similarity
            similarity = F.cosine_similarity(
                mars_embedding, 
                earth_features, 
                dim=1
            ).item()
            
            if similarity >= similarity_threshold:
                analogs.append({
                    'earth_location': metadata.get('location', 'Unknown'),
                    'similarity_score': similarity,
                    'geological_features': metadata.get('geological_features', []),
                    'coordinates': metadata.get('coordinates', None),
                    'earth_imagery': earth_imagery
                })
        
        # Sort by similarity and return top-k
        analogs.sort(key=lambda x: x['similarity_score'], reverse=True)
        return analogs[:top_k]


class MarsTerrainClassifier(nn.Module):
    """
    Mars terrain classifier using foundation model features.
    """
    
    def __init__(self, foundation_model: EarthMarsFoundationModel):
        super().__init__()
        self.foundation_model = foundation_model
        
        # Freeze foundation model during training
        for param in self.foundation_model.parameters():
            param.requires_grad = False
    
    def forward(self, mars_imagery: torch.Tensor) -> torch.Tensor:
        """Classify Mars terrain using foundation model features."""
        with torch.no_grad():
            outputs = self.foundation_model(mars_imagery, task="classification")
        
        return outputs['terrain_classification']


def create_earth_mars_foundation_model(
    pretrained_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> EarthMarsFoundationModel:
    """
    Factory function to create Earth-Mars foundation model.
    
    Args:
        pretrained_path: Path to pretrained Earth foundation model
        device: Device to load model on
        
    Returns:
        Initialized EarthMarsFoundationModel
    """
    config = MarsSpectraConfig()
    model = EarthMarsFoundationModel(pretrained_path, config)
    model.to(device)
    
    return model


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    mars_imagery = torch.randn(2, 12, 64, 64)  # Batch of Mars images
    earth_imagery = torch.randn(2, 10, 64, 64)  # Batch of Earth images
    
    # Initialize model
    model = create_earth_mars_foundation_model()
    
    # Test forward pass
    outputs = model(mars_imagery, earth_imagery, task="all")
    
    print("Model outputs:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
    
    # Test embedding generation
    embedding = model.get_mars_embedding(mars_imagery)
    print(f"Mars embedding shape: {embedding.shape}")
