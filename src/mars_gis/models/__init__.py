"""
Mars GIS Foundation Models

This module provides comprehensive foundation models for Mars data analysis,
including Earth-Mars transfer learning, multi-modal processing, comparative
analysis, landing site optimization, self-supervised learning, and
planetary-scale embedding generation.
"""

from .comparative import (
    ComparativePlanetaryAnalyzer,
    EarthAnalog,
    EarthEmbeddingDatabase,
    MarsRegion,
    create_comparative_analyzer,
)
from .foundation import (
    EarthFoundationEncoder,
    MarsEarthTransferModel,
    MarsSpecificAdapter,
    create_mars_earth_transfer_model,
)
from .multimodal import (
    MarsDataSample,
    MultiModalMarsProcessor,
    OpticalProcessor,
    ThermalProcessor,
    create_multimodal_mars_processor,
)
from .optimization import MarsLandingSiteOptimizer, create_landing_site_optimizer
from .planetary_scale import (
    PlanetaryScaleEmbeddingGenerator,
    create_planetary_embedding_generator,
)
from .self_supervised import SelfSupervisedMarsLearning, create_self_supervised_learner

__all__ = [
    # Foundation Transfer Learning
    "MarsEarthTransferModel",
    "MarsSpecificAdapter",
    "EarthFoundationEncoder",
    "create_mars_earth_transfer_model",
    
    # Multi-Modal Processing
    "MultiModalMarsProcessor",
    "OpticalProcessor",
    "ThermalProcessor",
    "MarsDataSample",
    "create_multimodal_mars_processor",
    
    # Comparative Analysis
    "ComparativePlanetaryAnalyzer",
    "EarthAnalog",
    "MarsRegion",
    "EarthEmbeddingDatabase",
    "create_comparative_analyzer",
    
    # Landing Site Optimization
    "MarsLandingSiteOptimizer",
    "create_landing_site_optimizer",
    
    # Self-Supervised Learning
    "SelfSupervisedMarsLearning",
    "create_self_supervised_learner",
    
    # Planetary-Scale Embeddings
    "PlanetaryScaleEmbeddingGenerator",
    "create_planetary_embedding_generator"
]

__version__ = "1.0.0"
__author__ = "Mars GIS Development Team"
__description__ = "Foundation models for Mars geospatial analysis"
