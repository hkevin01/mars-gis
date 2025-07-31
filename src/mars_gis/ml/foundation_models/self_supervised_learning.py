"""
Self-Supervised Learning Framework for Mars Foundation Models

This module implements self-supervised learning techniques including masked
autoencoding and contrastive learning for Mars geospatial data.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .earth_mars_transfer import EarthMarsFoundationModel


@dataclass
class MaskingConfig:
    """Configuration for masked autoencoding."""
    mask_ratio: float = 0.75  # Percentage of patches to mask
    patch_size: int = 16  # Size of image patches
    min_mask_patches: int = 4  # Minimum number of patches to mask
    max_mask_patches: int = 48  # Maximum number of patches to mask
    random_masking: bool = True  # Use random vs. block masking


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning."""
    temperature: float = 0.1   # Temperature for contrastive loss
    projection_dim: int = 256  # Dimension of projection head
    augmentation_strength: float = 0.5  # Strength of data augmentations
    negative_samples: int = 1024  # Number of negative samples


class PatchMasker(nn.Module):
    """
    Implements patch-based masking for masked autoencoding.
    """
    
    def __init__(self, config: MaskingConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply patch masking to input tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            mask_ratio: Override default mask ratio
            
        Returns:
            Tuple of (masked_input, mask, unmasked_patches)
        """
        B, C, H, W = x.shape
        patch_size = self.config.patch_size
        
        # Calculate number of patches
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        total_patches = num_patches_h * num_patches_w
        # Determine number of patches to mask
        mask_ratio = mask_ratio or self.config.mask_ratio
        num_masked = int(total_patches * mask_ratio)
        num_masked = max(self.config.min_mask_patches,
                         min(num_masked, self.config.max_mask_patches))
        
        # Create patch indices
        patch_indices = list(range(total_patches))
        
        if self.config.random_masking:
            # Random masking
            masked_indices = random.sample(patch_indices, num_masked)
        else:
            # Block masking (center region)
            center_h = num_patches_h // 2
            center_w = num_patches_w // 2
            block_size = int(math.sqrt(num_masked))
            
            masked_indices = []
            for i in range(max(0, center_h - block_size//2),
                           min(num_patches_h, center_h + block_size//2)):
                for j in range(max(0, center_w - block_size//2),
                               min(num_patches_w, center_w + block_size//2)):
                    masked_indices.append(i * num_patches_w + j)
        
        # Create mask tensor
        mask = torch.zeros(B, total_patches, device=x.device, dtype=torch.bool)
        for i in range(B):
            mask[i, masked_indices] = True
        
        # Reshape input to patches
        patches = x.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size)
        patches = patches.contiguous().view(
            B, C, total_patches, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(B, total_patches, -1)
        
        # Apply masking
        masked_patches = patches.clone()
        masked_patches[mask] = 0  # Zero out masked patches
        
        # Get unmasked patches for reconstruction target
        unmasked_patches = patches[~mask].view(B, -1, patches.shape[-1])
        
        # Reconstruct masked input
        masked_input = self._patches_to_image(
            masked_patches, B, C, H, W, patch_size,
            num_patches_h, num_patches_w
        )
        
        return masked_input, mask, unmasked_patches
    
    def _patches_to_image(
        self,
        patches: torch.Tensor,
        B: int, C: int, H: int, W: int,
        patch_size: int,
        num_patches_h: int,
        num_patches_w: int
    ) -> torch.Tensor:
        """Reconstruct image from patches."""
        patches = patches.view(
            B, num_patches_h, num_patches_w, C, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        image = patches.view(B, C, H, W)
        return image


class MaskedAutoEncoder(nn.Module):
    """
    Masked autoencoder for self-supervised pretraining.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        patch_size: int = 16,
        in_channels: int = 12
    ):
        super().__init__()
        
        self.encoder = encoder
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Decoder transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=decoder_depth
        )
        
        # Projection layers
        encoder_dim = (encoder.embed_dim if hasattr(encoder, 'embed_dim')
                       else 768)
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Reconstruction head
        self.reconstruction_head = nn.Linear(
            decoder_dim,
            patch_size * patch_size * in_channels
        )
        
        # Position embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, (224 // patch_size) ** 2, decoder_dim)
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize weights."""
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of masked autoencoder.
        
        Args:
            x: Input tensor (B, C, H, W)
            mask: Mask tensor (B, num_patches)
            
        Returns:
            Tuple of (reconstructed_patches, encoded_features)
        """
        # Encode visible patches
        encoded = self.encode_visible_patches(x, mask)
        
        # Decode with mask tokens
        reconstructed = self.decode_with_mask_tokens(encoded, mask)
        
        return reconstructed, encoded
    
    def encode_visible_patches(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode only visible (non-masked) patches."""
        # Convert to patches
        B, C, H, W = x.shape
        patch_size = self.patch_size
        
        patches = x.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(B, -1, patch_size * patch_size * C)
        
        # Select visible patches
        visible_patches = patches[~mask].view(B, -1, patches.shape[-1])
        
        # Encode visible patches (simplified - would use actual encoder)
        # For now, we'll use a simple linear projection
        if not hasattr(self, 'patch_encoder'):
            self.patch_encoder = nn.Linear(
                patch_size * patch_size * C,
                self.encoder_to_decoder.in_features
            ).to(x.device)
        
        encoded = self.patch_encoder(visible_patches)
        return encoded
    
    def decode_with_mask_tokens(
        self, 
        encoded: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Decode with mask tokens for masked patches."""
        B, num_patches = mask.shape
        
        # Project encoded features to decoder dimension
        encoded_decoder = self.encoder_to_decoder(encoded)
        
        # Create full sequence with mask tokens
        full_sequence = torch.zeros(
            B, num_patches, self.encoder_to_decoder.out_features,
            device=encoded.device
        )
        
        # Fill visible positions with encoded features
        visible_count = (~mask).sum(dim=1)
        for i in range(B):
            visible_idx = 0
            for j in range(num_patches):
                if not mask[i, j]:
                    full_sequence[i, j] = encoded_decoder[i, visible_idx]
                    visible_idx += 1
                else:
                    full_sequence[i, j] = self.mask_token.squeeze()
        
        # Add position embeddings
        if self.decoder_pos_embed.shape[1] >= num_patches:
            pos_embed = self.decoder_pos_embed[:, :num_patches, :]
        else:
            # Interpolate if needed
            pos_embed = self.decoder_pos_embed
        
        full_sequence = full_sequence + pos_embed
        
        # Apply decoder transformer
        decoded = self.decoder(full_sequence)
        
        # Reconstruct patches
        reconstructed = self.reconstruction_head(decoded)
        
        return reconstructed


class ContrastiveLearner(nn.Module):
    """
    Contrastive learning module for Mars geospatial data.
    """
    
    def __init__(
        self, 
        encoder: nn.Module,
        config: ContrastiveConfig
    ):
        super().__init__()
        
        self.encoder = encoder
        self.config = config
        
        # Projection head for contrastive learning
        encoder_dim = encoder.embed_dim if hasattr(encoder, 'embed_dim') else 768
        
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, config.projection_dim)
        )
        
        # Data augmentation modules
        self.augmentation = MarsDataAugmentation(config.augmentation_strength)
        
    def forward(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive learning.
        
        Args:
            x1: First view of the data
            x2: Second view of the data
            
        Returns:
            Tuple of (z1, z2) projected embeddings
        """
        # Encode both views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Project to contrastive space
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        
        # Normalize for cosine similarity
        z1 = nn.functional.normalize(z1, dim=-1)
        z2 = nn.functional.normalize(z2, dim=-1)
        
        return z1, z2
    
    def contrastive_loss(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss (InfoNCE).
        
        Args:
            z1: First set of embeddings
            z2: Second set of embeddings
            
        Returns:
            Contrastive loss value
        """
        batch_size = z1.shape[0]
        temperature = self.config.temperature
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z1, z2.T) / temperature
        
        # Create positive pairs (diagonal elements)
        positive_mask = torch.eye(batch_size, device=z1.device, dtype=torch.bool)
        
        # Compute loss
        numerator = torch.exp(sim_matrix[positive_mask])
        denominator = torch.exp(sim_matrix).sum(dim=1)
        
        loss = -torch.log(numerator / denominator).mean()
        
        return loss


class MarsDataAugmentation(nn.Module):
    """
    Data augmentation module for Mars imagery.
    """
    
    def __init__(self, strength: float = 0.5):
        super().__init__()
        self.strength = strength
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to Mars data."""
        # Random rotation
        if random.random() < self.strength:
            angle = random.uniform(-30, 30)
            x = self._rotate(x, angle)
        
        # Random flip
        if random.random() < self.strength:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])  # Horizontal flip
            if random.random() < 0.5:
                x = torch.flip(x, dims=[3])  # Vertical flip
        
        # Random brightness/contrast
        if random.random() < self.strength:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            x = x * contrast_factor + (brightness_factor - 1.0)
        
        # Channel-wise noise
        if random.random() < self.strength:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        return x
    
    def _rotate(self, x: torch.Tensor, angle: float) -> torch.Tensor:
        """Apply rotation to tensor (simplified implementation)."""
        # This is a placeholder - would use proper rotation in practice
        return x


class SelfSupervisedTrainer:
    """
    Trainer for self-supervised learning on Mars data.
    """
    
    def __init__(
        self,
        model: EarthMarsFoundationModel,
        masking_config: MaskingConfig,
        contrastive_config: ContrastiveConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.device = device
        
        # Initialize components
        self.patch_masker = PatchMasker(masking_config)
        self.masked_autoencoder = MaskedAutoEncoder(model.mars_encoder)
        self.contrastive_learner = ContrastiveLearner(model.mars_encoder, contrastive_config)
        
        # Move to device
        self.model.to(device)
        self.masked_autoencoder.to(device)
        self.contrastive_learner.to(device)
        
    def train_masked_autoencoding(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        lr: float = 1e-4
    ) -> List[float]:
        """
        Train using masked autoencoding.
        
        Args:
            dataloader: Mars imagery dataloader
            num_epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            List of training losses
        """
        optimizer = torch.optim.AdamW(
            self.masked_autoencoder.parameters(), 
            lr=lr, 
            weight_decay=0.05
        )
        
        losses = []
        
        self.masked_autoencoder.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                # Apply masking
                masked_input, mask, target_patches = self.patch_masker(x)
                
                # Forward pass
                reconstructed, _ = self.masked_autoencoder(masked_input, mask)
                
                # Compute reconstruction loss (only on masked patches)
                masked_targets = target_patches[mask]
                masked_predictions = reconstructed[mask]
                
                loss = nn.MSELoss()(masked_predictions, masked_targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            print(f"Epoch {epoch} completed. Average loss: {avg_loss:.6f}")
        
        return losses
    
    def train_contrastive(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        lr: float = 1e-4
    ) -> List[float]:
        """
        Train using contrastive learning.
        
        Args:
            dataloader: Mars imagery dataloader
            num_epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            List of training losses
        """
        optimizer = torch.optim.AdamW(
            self.contrastive_learner.parameters(),
            lr=lr,
            weight_decay=0.05
        )
        
        losses = []
        
        self.contrastive_learner.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                # Create two augmented views
                x1 = self.contrastive_learner.augmentation(x)
                x2 = self.contrastive_learner.augmentation(x)
                
                # Forward pass
                z1, z2 = self.contrastive_learner(x1, x2)
                
                # Compute contrastive loss
                loss = self.contrastive_learner.contrastive_loss(z1, z2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            print(f"Epoch {epoch} completed. Average loss: {avg_loss:.6f}")
        
        return losses


def create_self_supervised_trainer(
    foundation_model: EarthMarsFoundationModel,
    mask_ratio: float = 0.75,
    contrastive_temperature: float = 0.1,
    device: str = "cuda"
) -> SelfSupervisedTrainer:
    """
    Factory function to create self-supervised trainer.
    
    Args:
        foundation_model: Earth-Mars foundation model
        mask_ratio: Ratio of patches to mask
        contrastive_temperature: Temperature for contrastive loss
        device: Device to use for training
        
    Returns:
        Initialized SelfSupervisedTrainer
    """
    masking_config = MaskingConfig(mask_ratio=mask_ratio)
    contrastive_config = ContrastiveConfig(temperature=contrastive_temperature)
    
    return SelfSupervisedTrainer(
        foundation_model,
        masking_config,
        contrastive_config,
        device
    )


# Example usage and testing
if __name__ == "__main__":
    from .earth_mars_transfer import create_earth_mars_foundation_model

    # Create foundation model
    foundation_model = create_earth_mars_foundation_model()
    
    # Create self-supervised trainer
    trainer = create_self_supervised_trainer(foundation_model)
    
    # Create dummy dataloader for testing
    dummy_data = torch.randn(32, 12, 224, 224)  # 32 samples, 12 channels, 224x224
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data)
    dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=8, shuffle=True)
    
    print("Self-Supervised Learning Framework Initialized")
    print("=" * 50)
    
    # Test masked autoencoding
    print("\nTesting Masked Autoencoding...")
    mae_losses = trainer.train_masked_autoencoding(dataloader, num_epochs=2)
    print(f"MAE Training completed. Final loss: {mae_losses[-1]:.6f}")
    
    # Test contrastive learning
    print("\nTesting Contrastive Learning...")
    contrastive_losses = trainer.train_contrastive(dataloader, num_epochs=2)
    print(f"Contrastive Training completed. Final loss: {contrastive_losses[-1]:.6f}")
    
    print("\nSelf-supervised learning framework ready for Mars data training!")
