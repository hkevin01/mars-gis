"""
Self-Supervised Learning for Mars Representations

This module implements self-supervised learning techniques for learning
Mars-specific representations without manual labels.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


class SelfSupervisedMarsLearning(nn.Module):
    """
    Self-supervised learning system for Mars data.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Encoder for Mars imagery
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embedding_dim)
        )

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 64 * 64),  # Reconstruct to 64x64x3
            nn.Tanh()
        )

    def learn_mars_representations(
        self,
        mars_images: List[np.ndarray]
    ) -> np.ndarray:
        """Learn representations from Mars images."""
        with torch.no_grad():
            representations = []

            for img in mars_images:
                # Convert to tensor
                img_tensor = torch.from_numpy(img).float()
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)

                # Encode
                encoding = self.encoder(img_tensor)
                representations.append(encoding.cpu().numpy())

            return np.vstack(representations)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the self-supervised model."""
        # Encode
        encoding = self.encoder(x)

        # Project for contrastive learning
        projection = self.projection_head(encoding)

        # Reconstruct
        reconstruction = self.decoder(encoding)
        reconstruction = reconstruction.view(-1, 3, 64, 64)

        return {
            'encoding': encoding,
            'projection': projection,
            'reconstruction': reconstruction
        }

    def contrastive_loss(
        self,
        projections: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """Compute contrastive loss for self-supervised learning."""
        batch_size = projections.shape[0]

        # Normalize projections
        projections = torch.nn.functional.normalize(projections, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T)
        similarity_matrix = similarity_matrix / temperature

        # Create positive pairs mask (assuming consecutive pairs are positive)
        mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        for i in range(0, batch_size - 1, 2):
            mask[i, i + 1] = True
            mask[i + 1, i] = True

        # Compute contrastive loss
        positive_samples = similarity_matrix[mask]
        negative_samples = similarity_matrix[~mask]

        positive_loss = -torch.log(torch.exp(positive_samples).mean())
        negative_loss = torch.log(torch.exp(negative_samples).mean())

        return positive_loss + negative_loss


def create_self_supervised_learner(
    embedding_dim: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> SelfSupervisedMarsLearning:
    """Factory function to create self-supervised learner."""
    learner = SelfSupervisedMarsLearning(embedding_dim)
    learner.to(device)
    return learner
