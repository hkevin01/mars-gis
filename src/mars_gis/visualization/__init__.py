"""
Mars GIS Visualization Module

This module provides comprehensive visualization capabilities for Mars
data analysis, including 3D globe rendering, interactive mapping, and
real-time analysis dashboards.
"""

from .analysis_dashboard import (
    AnalysisJob,
    AnalysisJobManager,
    DashboardConfig,
    DashboardMetrics,
    MarsAnalysisDashboard,
    RealTimeMarsAnalytics,
    create_mars_analysis_dashboard,
)
from .interactive_mapping import (
    InteractiveFeature,
    InteractiveMarsMap,
    MapConfig,
    MapLayer,
    MarsMapTileGenerator,
    create_interactive_mars_map,
)
from .mars_3d_globe import (
    Globe3DData,
    Mars3DGlobeGenerator,
    MarsColorMapper,
    VisualizationConfig,
    create_mars_3d_globe,
)

__all__ = [
    # 3D Globe Visualization
    "Mars3DGlobeGenerator",
    "MarsColorMapper",
    "Globe3DData",
    "VisualizationConfig",
    "create_mars_3d_globe",
    
    # Interactive Mapping
    "InteractiveMarsMap",
    "MarsMapTileGenerator",
    "MapConfig",
    "MapLayer",
    "InteractiveFeature",
    "create_interactive_mars_map",
    
    # Analysis Dashboard
    "MarsAnalysisDashboard",
    "RealTimeMarsAnalytics",
    "AnalysisJobManager",
    "DashboardConfig",
    "AnalysisJob",
    "DashboardMetrics",
    "create_mars_analysis_dashboard"
]

__version__ = "1.0.0"
__author__ = "Mars GIS Development Team"
__description__ = "Comprehensive Mars data visualization and analysis tools"
