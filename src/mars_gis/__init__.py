"""MARS-GIS: Mars Geospatial Intelligence System."""

__version__ = "1.0.0"
__author__ = "Kevin Hildebrand"
__email__ = "kevin.hildebrand@gmail.com"

"""MARS-GIS: Mars Geospatial Intelligence System."""

__version__ = "1.0.0"
__author__ = "Kevin Hildebrand"
__email__ = "kevin.hildebrand@gmail.com"

# Import core configuration (always available)
from mars_gis.core.config import settings

# Optional imports for models and visualization (require additional dependencies)
try:
    from mars_gis.models import (
        ComparativePlanetaryAnalyzer,
        MarsEarthTransferModel,
        MarsLandingSiteOptimizer,
        MultiModalMarsProcessor,
        PlanetaryScaleEmbeddingGenerator,
        SelfSupervisedMarsLearning,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False

try:
    from mars_gis.visualization import (
        InteractiveMarsMap,
        Mars3DGlobeGenerator,
        MarsAnalysisDashboard,
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

# Base exports (always available)
__all__ = [
    "settings",
]

# Add optional exports if available
if _MODELS_AVAILABLE:
    __all__.extend([
        "MarsEarthTransferModel",
        "MultiModalMarsProcessor",
        "ComparativePlanetaryAnalyzer",
        "MarsLandingSiteOptimizer",
        "SelfSupervisedMarsLearning",
        "PlanetaryScaleEmbeddingGenerator",
    ])

if _VISUALIZATION_AVAILABLE:
    __all__.extend([
        "Mars3DGlobeGenerator",
        "InteractiveMarsMap",
        "MarsAnalysisDashboard",
    ])
