"""Training utilities for Mars ML models."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
    
    # Type aliases for when torch is available
    TorchDataLoader = DataLoader
    TorchDataset = Dataset
    TorchModule = nn.Module
    TorchOptimizer = optim.Optimizer
    
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = None
    
    # Fallback types when torch is not available
    TorchDataLoader = Any
    TorchDataset = object
    TorchModule = Any
    TorchOptimizer = Any

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class MarsImageDataset(TorchDataset):
    """Dataset class for Mars imagery data."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None
    ):
        """
        Initialize Mars image dataset.
        
        Args:
            image_paths: List of paths to Mars images
            labels: List of corresponding labels
            transform: Optional data transformations
        """
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Get item from dataset."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        # Load image (placeholder - would use PIL/OpenCV in real implementation)
        image_path = self.image_paths[idx]
        # image = load_mars_image(image_path)  # Placeholder
        
        # For now, create dummy tensor
        image = torch.randn(3, 224, 224)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


class ModelTrainer:
    """Trainer class for Mars ML models."""
    
    def __init__(
        self,
        model: Any,
        device: Optional[str] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on (cuda/cpu)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for training")
        
        self.model = model
        if device is None:
            if TORCH_AVAILABLE and torch and torch.cuda and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        
        self.training_history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "epochs": []
        }
    
    def train_terrain_classifier(
        self,
        train_loader: Any,  # TorchDataLoader
        val_loader: Any,   # TorchDataLoader
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train terrain classification model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            save_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        if not TORCH_AVAILABLE:
            return {}
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Save training history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_accuracy"].append(train_acc)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_accuracy"].append(val_acc)
            self.training_history["epochs"].append(epoch + 1)
            
            # Save best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print("-" * 50)
        
        return self.training_history
    
    def _train_epoch(
        self,
        train_loader: Any,
        criterion: Any,
        optimizer: Any
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        if not TORCH_AVAILABLE:
            return 0.0, 0.0
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(
        self,
        val_loader: Any,
        criterion: Any
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        if not TORCH_AVAILABLE:
            return 0.0, 0.0
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def save_training_history(self, filepath: str):
        """Save training history to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.training_history, f, indent=2)


class ModelEvaluator:
    """Evaluator for trained Mars ML models."""
    
    def __init__(self, model: Any, device: str = "cpu"):
        """Initialize model evaluator."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate_terrain_model(
        self,
        test_loader: Any
    ) -> Dict[str, Any]:
        """Evaluate terrain classification model."""
        if not TORCH_AVAILABLE:
            return {}
        
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions for detailed analysis
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_correct[label] = class_correct.get(label, 0) + (
                        predicted[i] == labels[i]
                    ).item()
                    class_total[label] = class_total.get(label, 0) + 1
        
        # Calculate metrics
        overall_accuracy = correct / total
        class_accuracies = {
            cls: class_correct[cls] / class_total[cls]
            for cls in class_total.keys()
        }
        
        return {
            "overall_accuracy": overall_accuracy,
            "class_accuracies": class_accuracies,
            "total_samples": total,
            "correct_predictions": correct,
            "predictions": predictions,
            "true_labels": true_labels
        }
    
    def evaluate_hazard_detector(
        self,
        test_loader: Any
    ) -> Dict[str, Any]:
        """Evaluate hazard detection model."""
        if not TORCH_AVAILABLE:
            return {}
        
        hazard_correct = 0
        total = 0
        safety_scores = []
        true_safety_scores = []
        
        with torch.no_grad():
            for images, (hazard_labels, safety_labels) in test_loader:
                images = images.to(self.device)
                hazard_labels = hazard_labels.to(self.device)
                safety_labels = safety_labels.to(self.device)
                
                hazard_logits, safety_preds = self.model(images)
                _, hazard_predicted = torch.max(hazard_logits, 1)
                
                total += hazard_labels.size(0)
                hazard_correct += (hazard_predicted == hazard_labels).sum().item()
                
                safety_scores.extend(safety_preds.cpu().numpy())
                true_safety_scores.extend(safety_labels.cpu().numpy())
        
        # Calculate safety score MSE
        if NUMPY_AVAILABLE:
            safety_mse = np.mean(
                (np.array(safety_scores) - np.array(true_safety_scores)) ** 2
            )
        else:
            safety_mse = 0.0
        
        return {
            "hazard_accuracy": hazard_correct / total,
            "safety_score_mse": safety_mse,
            "total_samples": total
        }


# Training pipeline functions
def create_training_pipeline(
    model_type: str,
    data_dir: Path,
    model_save_dir: Path
) -> Dict[str, Any]:
    """Create complete training pipeline for Mars models."""
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    pipeline_config = {
        "model_type": model_type,
        "data_directory": str(data_dir),
        "model_save_directory": str(model_save_dir),
        "training_parameters": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 50,
            "validation_split": 0.2
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create directories
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save pipeline configuration
    config_file = model_save_dir / "training_config.json"
    with open(config_file, "w") as f:
        json.dump(pipeline_config, f, indent=2)
    
    return pipeline_config


def run_model_training(config: Dict[str, Any]) -> bool:
    """Run model training based on configuration."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available for training")
        return False
    
    try:
        print(f"Starting training for {config['model_type']} model...")
        
        # Create dummy data for demonstration
        # In real implementation, this would load actual Mars imagery
        dummy_images = [f"image_{i}.jpg" for i in range(100)]
        dummy_labels = [i % 8 for i in range(100)]  # 8 terrain classes
        
        dataset = MarsImageDataset(dummy_images, dummy_labels)
        dataloader = TorchDataLoader(dataset, batch_size=16, shuffle=True)
        
        print(f"Created dataset with {len(dataset)} samples")
        print("Note: Using dummy data for demonstration")
        print("In production, replace with real Mars imagery data")
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False
