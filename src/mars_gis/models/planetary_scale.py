"""
Planetary-Scale Embedding Generation

This module implements large-scale embedding generation for
Mars planetary data analysis.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn


class PlanetaryScaleEmbeddingGenerator(nn.Module):
    """
    Generator for planetary-scale embeddings.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Main embedding network
        self.embedding_network = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim)
        )

        # Normalization layer for embeddings
        self.normalizer = nn.LayerNorm(embedding_dim)

    def generate_embeddings(self, images: List[np.ndarray]) -> np.ndarray:
        """Generate embeddings for a list of images."""
        with torch.no_grad():
            embeddings = []

            for img in images:
                # Convert to tensor
                img_tensor = torch.from_numpy(img).float()
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)

                # Generate embedding
                embedding = self.embedding_network(img_tensor)
                embedding = self.normalizer(embedding)
                embeddings.append(embedding.cpu().numpy())

            return np.vstack(embeddings)

    def find_similar_regions(
        self,
        query_embedding: np.ndarray,
        database_embeddings: np.ndarray,
        top_k: int = 10
    ) -> List[int]:
        """Find similar regions using cosine similarity."""
        # Convert to tensors
        query_tensor = torch.from_numpy(query_embedding).float()
        database_tensor = torch.from_numpy(database_embeddings).float()

        # Ensure query is 2D
        if query_tensor.dim() == 1:
            query_tensor = query_tensor.unsqueeze(0)

        # Calculate similarities
        similarities = torch.nn.functional.cosine_similarity(
            query_tensor, database_tensor, dim=1
        )

        # Get top-k indices
        max_k = min(top_k, len(similarities))
        _, top_indices = torch.topk(similarities, max_k)

        return top_indices.tolist()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through embedding generator."""
        embedding = self.embedding_network(x)
        normalized_embedding = self.normalizer(embedding)
        return normalized_embedding


def create_planetary_embedding_generator(
    embedding_dim: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> PlanetaryScaleEmbeddingGenerator:
    """Factory function to create planetary embedding generator."""
    generator = PlanetaryScaleEmbeddingGenerator(embedding_dim)
    generator.to(device)
    return generator
