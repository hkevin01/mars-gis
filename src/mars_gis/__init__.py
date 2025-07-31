"""MARS-GIS: Mars Geospatial Intelligence System."""

__version__ = "1.0.0"
__author__ = "Kevin Hildebrand"
__email__ = "kevin.hildebrand@gmail.com"

# Import core configuration
from mars_gis.core.config import settings

# Import foundation models
from mars_gis.models import (
    ComparativePlanetaryAnalyzer,
    MarsEarthTransferModel,
    MarsLandingSiteOptimizer,
    MultiModalMarsProcessor,
    PlanetaryScaleEmbeddingGenerator,
    SelfSupervisedMarsLearning,
)

# Import visualization components
from mars_gis.visualization import (
    InteractiveMarsMap,
    Mars3DGlobeGenerator,
    MarsAnalysisDashboard,
)

__all__ = [
    "settings",
    # Foundation Models
    "MarsEarthTransferModel",
    "MultiModalMarsProcessor",
    "ComparativePlanetaryAnalyzer",
    "MarsLandingSiteOptimizer",
    "SelfSupervisedMarsLearning",
    "PlanetaryScaleEmbeddingGenerator",
    # Visualization
    "Mars3DGlobeGenerator",
    "InteractiveMarsMap",
    "MarsAnalysisDashboard"
]
