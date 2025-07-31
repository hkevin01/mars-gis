"""
Multi-Modal Data Fusion for Mars Analysis

This module implements multi-modal data fusion for creating unified
embeddings from diverse Mars data sources.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class DataAvailability(Enum):
    """Data availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    DEGRADED = "degraded"


class DataUnavailableError(Exception):
    """Raised when required data is not available."""
    pass


@dataclass
class MarsDataSample:
    """Container for Mars multi-modal data sample."""
    optical: Optional[torch.Tensor] = None
    thermal: Optional[torch.Tensor] = None
    elevation: Optional[torch.Tensor] = None
    spectral: Optional[torch.Tensor] = None
    radar: Optional[torch.Tensor] = None
    atmospheric: Optional[torch.Tensor] = None
    location: Optional[tuple] = None
    timestamp: Optional[str] = None
    quality_scores: Optional[Dict[str, float]] = None


class OpticalProcessor(nn.Module):
    """Process HiRISE and CTX optical imagery."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        # HiRISE RGB processing
        self.hirise_processor = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim)
        )

        # CTX grayscale processing
        self.ctx_processor = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, optical_data: torch.Tensor) -> torch.Tensor:
        """Process optical imagery."""
        if optical_data.shape[1] == 3:
            return self.hirise_processor(optical_data)
        elif optical_data.shape[1] == 1:
            return self.ctx_processor(optical_data)
        else:
            msg = f"Unsupported optical channels: {optical_data.shape[1]}"
            raise ValueError(msg)


class ThermalProcessor(nn.Module):
    """Process THEMIS thermal infrared data."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        # THEMIS thermal processing
        self.thermal_net = nn.Sequential(
            nn.Conv2d(5, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, thermal_data: torch.Tensor) -> torch.Tensor:
        """Process thermal imagery."""
        return self.thermal_net(thermal_data)


class MultiModalMarsProcessor(nn.Module):
    """
    Multi-modal Mars data processor for unified embeddings.
    """

    def __init__(self, embedding_dim: int = 64, fusion_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fusion_dim = fusion_dim

        # Individual modality processors
        self.optical_processor = OpticalProcessor(embedding_dim)
        self.thermal_processor = ThermalProcessor(embedding_dim)

        # Elevation processor
        self.elevation_processor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, embedding_dim)
        )

        # Spectral processor
        self.spectral_processor = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embedding_dim)
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            batch_first=True
        )

        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(embedding_dim * 4, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, embedding_dim)
        )

    def process_multimodal_data(
        self,
        visual: np.ndarray,
        spectral: np.ndarray,
        thermal: np.ndarray
    ) -> np.ndarray:
        """Process multi-modal data and return unified embedding."""
        with torch.no_grad():
            # Convert to tensors
            visual_tensor = torch.from_numpy(visual).float()
            spectral_tensor = torch.from_numpy(spectral).float()
            thermal_tensor = torch.from_numpy(thermal).float()

            # Add batch dimension if needed
            if visual_tensor.dim() == 3:
                visual_tensor = visual_tensor.unsqueeze(0)
            if spectral_tensor.dim() == 3:
                spectral_tensor = spectral_tensor.unsqueeze(0)
            if thermal_tensor.dim() == 3:
                thermal_tensor = thermal_tensor.unsqueeze(0)

            # Process each modality
            visual_features = self.optical_processor(visual_tensor)
            spectral_features = self.spectral_processor(spectral_tensor)
            thermal_features = self.thermal_processor(thermal_tensor)

            # Create dummy elevation features for fusion
            elevation_features = torch.zeros_like(visual_features)

            # Fuse modalities
            fused_features = self.fuse_modalities([
                visual_features, spectral_features,
                thermal_features, elevation_features
            ])

            return fused_features.cpu().numpy()

    def fuse_modalities(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple modality features."""
        # Concatenate all features
        concatenated = torch.cat(features, dim=1)

        # Apply fusion network
        fused = self.fusion_network(concatenated)

        return fused

    def forward(
        self,
        data_sample: Union[MarsDataSample, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Forward pass through multi-modal processor."""
        if isinstance(data_sample, dict):
            # Convert dict to MarsDataSample
            data_sample = MarsDataSample(**data_sample)

        features = []

        # Process available modalities
        if data_sample.optical is not None:
            optical_feat = self.optical_processor(data_sample.optical)
            features.append(optical_feat)
        else:
            features.append(torch.zeros(1, self.embedding_dim))

        if data_sample.thermal is not None:
            thermal_feat = self.thermal_processor(data_sample.thermal)
            features.append(thermal_feat)
        else:
            features.append(torch.zeros(1, self.embedding_dim))

        if data_sample.elevation is not None:
            elevation_feat = self.elevation_processor(data_sample.elevation)
            features.append(elevation_feat)
        else:
            features.append(torch.zeros(1, self.embedding_dim))

        if data_sample.spectral is not None:
            spectral_feat = self.spectral_processor(data_sample.spectral)
            features.append(spectral_feat)
        else:
            features.append(torch.zeros(1, self.embedding_dim))

        # Fuse all features
        fused_features = self.fuse_modalities(features)

        return fused_features


def create_multimodal_mars_processor(
    embedding_dim: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> MultiModalMarsProcessor:
    """Factory function to create multi-modal Mars processor."""
    processor = MultiModalMarsProcessor(embedding_dim)
    processor.to(device)
    return processor
