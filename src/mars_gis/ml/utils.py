"""ML utilities for Mars GIS system."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    classification_report = confusion_matrix = train_test_split = None


class ModelMetrics:
    """Utility class for computing model performance metrics."""
    
    @staticmethod
    def compute_classification_metrics(
        y_true: List[int],
        y_pred: List[int],
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names for reporting
            
        Returns:
            Dictionary containing all metrics
        """
        if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("scikit-learn or numpy not available for metrics")
            return {}
        
        try:
            # Basic accuracy
            accuracy = np.mean(np.array(y_true) == np.array(y_pred))
            
            # Classification report
            report = classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Per-class metrics
            per_class_metrics = {}
            if class_names:
                for i, class_name in enumerate(class_names):
                    if str(i) in report:
                        per_class_metrics[class_name] = report[str(i)]
            
            return {
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "per_class_metrics": per_class_metrics,
                "macro_avg": report.get("macro avg", {}),
                "weighted_avg": report.get("weighted avg", {})
            }
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {}
    
    @staticmethod
    def compute_regression_metrics(
        y_true: List[float],
        y_pred: List[float]
    ) -> Dict[str, float]:
        """
        Compute regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing regression metrics
        """
        if not NUMPY_AVAILABLE:
            logger.warning("numpy not available for metrics")
            return {}
        
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Mean Squared Error
            mse = np.mean((y_true - y_pred) ** 2)
            
            # Root Mean Squared Error
            rmse = np.sqrt(mse)
            
            # Mean Absolute Error
            mae = np.mean(np.abs(y_true - y_pred))
            
            # R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2)
            }
            
        except Exception as e:
            logger.error(f"Error computing regression metrics: {e}")
            return {}


class DatasetSplitter:
    """Utility for splitting datasets for ML training."""
    
    @staticmethod
    def split_mars_dataset(
        image_paths: List[str],
        labels: List[int],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]:
        """
        Split Mars dataset into train/validation/test sets.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify splits
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for dataset splitting")
            # Return simple splits without stratification
            n_samples = len(image_paths)
            n_test = int(n_samples * test_size)
            n_val = int(n_samples * val_size)
            
            test_data = {
                "image_paths": image_paths[:n_test],
                "labels": labels[:n_test]
            }
            val_data = {
                "image_paths": image_paths[n_test:n_test + n_val],
                "labels": labels[n_test:n_test + n_val]
            }
            train_data = {
                "image_paths": image_paths[n_test + n_val:],
                "labels": labels[n_test + n_val:]
            }
            
            return train_data, val_data, test_data
        
        try:
            stratify_labels = labels if stratify else None
            
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                image_paths, labels,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels
            )
            
            # Second split: separate train and validation
            val_ratio = val_size / (1 - test_size)
            stratify_temp = y_temp if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio,
                random_state=random_state,
                stratify=stratify_temp
            )
            
            train_data = {"image_paths": X_train, "labels": y_train}
            val_data = {"image_paths": X_val, "labels": y_val}
            test_data = {"image_paths": X_test, "labels": y_test}
            
            logger.info(f"Dataset split - Train: {len(X_train)}, "
                       f"Val: {len(X_val)}, Test: {len(X_test)}")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            return {}, {}, {}


class ModelCheckpoint:
    """Utility for saving and loading model checkpoints."""
    
    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        epoch: int,
        metrics: Dict[str, float],
        model_name: str = "model"
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Training metrics
            model_name: Name for the checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        try:
            checkpoint_data = {
                "epoch": epoch,
                "metrics": metrics,
                "timestamp": json.dumps(
                    {"timestamp": "placeholder"}, default=str
                )
            }
            
            # Try to save model state if PyTorch is available
            try:
                import torch
                checkpoint_data.update({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                })
            except ImportError:
                logger.warning("PyTorch not available for checkpoint saving")
            
            checkpoint_path = self.checkpoint_dir / f"{model_name}_epoch_{epoch}.pth"
            
            # Save with pickle as fallback
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return ""
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Any = None,
        optimizer: Any = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            
        Returns:
            Checkpoint data dictionary
        """
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Try to load PyTorch states if available
            try:
                import torch
                if model and "model_state_dict" in checkpoint_data:
                    model.load_state_dict(checkpoint_data["model_state_dict"])
                if optimizer and "optimizer_state_dict" in checkpoint_data:
                    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            except ImportError:
                logger.warning("PyTorch not available for state loading")
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return {}
    
    def get_best_checkpoint(
        self,
        metric_name: str = "val_accuracy",
        maximize: bool = True
    ) -> Optional[str]:
        """
        Find the best checkpoint based on a metric.
        
        Args:
            metric_name: Name of metric to optimize
            maximize: Whether to maximize or minimize the metric
            
        Returns:
            Path to best checkpoint or None
        """
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("*.pth"))
            if not checkpoint_files:
                return None
            
            best_metric = float('-inf') if maximize else float('inf')
            best_checkpoint = None
            
            for checkpoint_file in checkpoint_files:
                try:
                    with open(checkpoint_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    if "metrics" in data and metric_name in data["metrics"]:
                        metric_value = data["metrics"][metric_name]
                        
                        if maximize and metric_value > best_metric:
                            best_metric = metric_value
                            best_checkpoint = str(checkpoint_file)
                        elif not maximize and metric_value < best_metric:
                            best_metric = metric_value
                            best_checkpoint = str(checkpoint_file)
                            
                except Exception as e:
                    logger.warning(f"Error reading checkpoint {checkpoint_file}: {e}")
                    continue
            
            return best_checkpoint
            
        except Exception as e:
            logger.error(f"Error finding best checkpoint: {e}")
            return None


class ExperimentTracker:
    """Utility for tracking ML experiments."""
    
    def __init__(self, experiment_dir: Path):
        """Initialize experiment tracker."""
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []
    
    def log_experiment(
        self,
        experiment_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        model_path: Optional[str] = None
    ) -> str:
        """
        Log an ML experiment.
        
        Args:
            experiment_name: Name of the experiment
            parameters: Hyperparameters used
            metrics: Resulting metrics
            model_path: Path to saved model
            
        Returns:
            Experiment ID
        """
        import time
        
        experiment_id = f"{experiment_name}_{int(time.time())}"
        
        experiment_data = {
            "id": experiment_id,
            "name": experiment_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": parameters,
            "metrics": metrics,
            "model_path": model_path
        }
        
        # Save experiment data
        experiment_file = self.experiment_dir / f"{experiment_id}.json"
        with open(experiment_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        self.experiments.append(experiment_data)
        logger.info(f"Experiment logged: {experiment_id}")
        
        return experiment_id
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        if not self.experiments:
            # Load from files
            experiment_files = list(self.experiment_dir.glob("*.json"))
            for file in experiment_files:
                try:
                    with open(file) as f:
                        experiment_data = json.load(f)
                    self.experiments.append(experiment_data)
                except Exception as e:
                    logger.warning(f"Error loading experiment {file}: {e}")
        
        return {
            "total_experiments": len(self.experiments),
            "experiments": self.experiments
        }
    
    def get_best_experiment(
        self,
        metric_name: str = "val_accuracy",
        maximize: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get the best experiment based on a metric."""
        if not self.experiments:
            self.get_experiment_summary()
        
        if not self.experiments:
            return None
        
        best_experiment = None
        best_metric = float('-inf') if maximize else float('inf')
        
        for experiment in self.experiments:
            if metric_name in experiment.get("metrics", {}):
                metric_value = experiment["metrics"][metric_name]
                
                if maximize and metric_value > best_metric:
                    best_metric = metric_value
                    best_experiment = experiment
                elif not maximize and metric_value < best_metric:
                    best_metric = metric_value
                    best_experiment = experiment
        
        return best_experiment


def setup_ml_environment(
    project_root: Path,
    create_dirs: bool = True
) -> Dict[str, Path]:
    """
    Setup ML environment directories and utilities.
    
    Args:
        project_root: Root directory of the project
        create_dirs: Whether to create directories
        
    Returns:
        Dictionary of important paths
    """
    paths = {
        "models": project_root / "models",
        "data": project_root / "data" / "processed",
        "checkpoints": project_root / "checkpoints",
        "experiments": project_root / "experiments",
        "logs": project_root / "logs"
    }
    
    if create_dirs:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    logger.info("ML environment setup complete")
    return paths


def validate_model_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate ML model configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    required_fields = ["model_type", "input_shape", "num_classes"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate model type
    valid_types = ["terrain", "hazard", "atmosphere"]
    if config.get("model_type") not in valid_types:
        errors.append(f"Invalid model_type. Must be one of: {valid_types}")
    
    # Validate numeric fields
    numeric_fields = ["num_classes", "batch_size", "learning_rate"]
    for field in numeric_fields:
        if field in config:
            try:
                float(config[field])
            except (ValueError, TypeError):
                errors.append(f"Field {field} must be numeric")
    
    return errors
