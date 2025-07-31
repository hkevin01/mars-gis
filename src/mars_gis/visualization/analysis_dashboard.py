"""
Real-time Mars Analysis Dashboard

This module provides a comprehensive real-time dashboard for Mars data analysis
using foundation models and interactive visualizations.
"""

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from mars_gis.core.types import BoundingBox, MarsCoordinate

from ..foundation_models.comparative_planetary import ComparativePlanetaryAnalyzer
from ..foundation_models.landing_site_optimization import (
    FoundationModelLandingSiteSelector,
)
from ..foundation_models.planetary_scale_embeddings import EmbeddingDatabase


@dataclass
class DashboardConfig:
    """Configuration for analysis dashboard."""
    update_interval: float = 5.0  # seconds
    max_concurrent_analyses: int = 5
    cache_enabled: bool = True
    real_time_updates: bool = True
    analysis_timeout: float = 30.0  # seconds


@dataclass
class AnalysisJob:
    """Analysis job definition."""
    job_id: str
    job_type: str
    parameters: Dict[str, Any]
    status: str  # pending, running, completed, failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class DashboardMetrics:
    """Dashboard performance metrics."""
    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    avg_analysis_time: float = 0.0
    active_connections: int = 0
    memory_usage_mb: float = 0.0
    embedding_queries_per_sec: float = 0.0


class AnalysisJobManager:
    """
    Manages analysis jobs and their execution.
    """
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.jobs: Dict[str, AnalysisJob] = {}
        self.running_jobs: List[str] = []
        self.job_queue: List[str] = []
        
    def submit_job(
        self,
        job_type: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Submit new analysis job."""
        job_id = f"{job_type}_{int(time.time() * 1000)}"
        
        job = AnalysisJob(
            job_id=job_id,
            job_type=job_type,
            parameters=parameters,
            status="pending",
            created_at=datetime.now()
        )
        
        self.jobs[job_id] = job
        self.job_queue.append(job_id)
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[AnalysisJob]:
        """Get status of analysis job."""
        return self.jobs.get(job_id)
    
    def process_jobs(
        self, analyzer_functions: Dict[str, Callable]
    ) -> List[str]:
        """Process pending jobs."""
        completed_jobs = []
        
        # Start new jobs if capacity available
        while (len(self.running_jobs) < self.config.max_concurrent_analyses and
               self.job_queue):
            job_id = self.job_queue.pop(0)
            job = self.jobs[job_id]
            
            if job.job_type in analyzer_functions:
                job.status = "running"
                job.started_at = datetime.now()
                self.running_jobs.append(job_id)
                
                try:
                    # Execute analysis
                    analyzer = analyzer_functions[job.job_type]
                    results = analyzer(**job.parameters)
                    
                    job.status = "completed"
                    job.completed_at = datetime.now()
                    job.results = results
                    
                except Exception as e:
                    job.status = "failed"
                    job.completed_at = datetime.now()
                    job.error_message = str(e)
                
                self.running_jobs.remove(job_id)
                completed_jobs.append(job_id)
            else:
                job.status = "failed"
                job.error_message = f"Unknown job type: {job.job_type}"
                completed_jobs.append(job_id)
        
        return completed_jobs


class RealTimeMarsAnalytics:
    """
    Real-time analytics engine for Mars data.
    """
    
    def __init__(
        self,
        embedding_db: EmbeddingDatabase,
        landing_site_selector: FoundationModelLandingSiteSelector,
        comparative_analyzer: ComparativePlanetaryAnalyzer
    ):
        self.embedding_db = embedding_db
        self.landing_site_selector = landing_site_selector
        self.comparative_analyzer = comparative_analyzer
        
        # Analytics state
        self.active_analyses: Dict[str, Any] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        self.performance_metrics = DashboardMetrics()
        
        # Real-time data streams
        self.data_streams: Dict[str, List[Dict[str, Any]]] = {
            "embedding_queries": [],
            "landing_site_assessments": [],
            "earth_analog_discoveries": [],
            "terrain_analyses": []
        }
    
    def analyze_region_comprehensive(
        self,
        bounds: BoundingBox,
        analysis_types: List[str]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of Mars region.
        
        Args:
            bounds: Geographic bounds to analyze
            analysis_types: Types of analysis to perform
            
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        results = {
            "region_bounds": asdict(bounds),
            "analysis_timestamp": datetime.now().isoformat(),
            "analyses": {}
        }
        
        # Landing site analysis
        if "landing_sites" in analysis_types:
            landing_results = self._analyze_landing_sites_comprehensive(bounds)
            results["analyses"]["landing_sites"] = landing_results
            
            # Update data stream
            self.data_streams["landing_site_assessments"].append({
                "timestamp": time.time(),
                "region": asdict(bounds),
                "site_count": len(landing_results.get("sites", []))
            })
        
        # Earth analog discovery
        if "earth_analogs" in analysis_types:
            analog_results = self._discover_earth_analogs(bounds)
            results["analyses"]["earth_analogs"] = analog_results
            
            # Update data stream
            self.data_streams["earth_analog_discoveries"].append({
                "timestamp": time.time(),
                "region": asdict(bounds),
                "analog_count": len(analog_results.get("analogs", []))
            })
        
        # Embedding analysis
        if "embeddings" in analysis_types:
            embedding_results = self._analyze_embeddings_detailed(bounds)
            results["analyses"]["embeddings"] = embedding_results
            
            # Update data stream
            self.data_streams["embedding_queries"].append({
                "timestamp": time.time(),
                "region": asdict(bounds),
                "embedding_count": embedding_results.get("tile_count", 0)
            })
        
        # Terrain analysis
        if "terrain" in analysis_types:
            terrain_results = self._analyze_terrain_detailed(bounds)
            results["analyses"]["terrain"] = terrain_results
            
            # Update data stream
            self.data_streams["terrain_analyses"].append({
                "timestamp": time.time(),
                "region": asdict(bounds),
                "features": terrain_results.get("feature_count", 0)
            })
        
        # Update performance metrics
        analysis_time = time.time() - start_time
        self.performance_metrics.total_analyses += 1
        self.performance_metrics.successful_analyses += 1
        
        # Update average analysis time
        current_avg = self.performance_metrics.avg_analysis_time
        total = self.performance_metrics.total_analyses
        self.performance_metrics.avg_analysis_time = (
            (current_avg * (total - 1) + analysis_time) / total
        )
        
        results["analysis_duration"] = analysis_time
        self.analysis_history.append(results)
        
        return results
    
    def _analyze_landing_sites_comprehensive(
        self, 
        bounds: BoundingBox
    ) -> Dict[str, Any]:
        """Comprehensive landing site analysis."""
        # Generate candidate sites
        candidate_sites = self._generate_candidate_sites(bounds, density=20)
        
        # Evaluate sites for different mission types
        mission_types = ["sample_return", "exploration", "human_precursor"]
        mission_results = {}
        
        for mission_type in mission_types:
            assessments = self.landing_site_selector.evaluate_landing_sites(
                candidate_sites, mission_type=mission_type, top_k=10
            )
            
            mission_results[mission_type] = [
                {
                    "rank": i + 1,
                    "coordinates": {
                        "latitude": assessment.location.latitude,
                        "longitude": assessment.location.longitude
                    },
                    "scores": {
                        "overall": assessment.overall_ranking,
                        "safety": assessment.safety_score,
                        "science": assessment.science_value,
                        "operations": assessment.operational_score
                    },
                    "confidence": assessment.confidence,
                    "recommendation": assessment.recommendation,
                    "risks": assessment.risk_factors,
                    "opportunities": assessment.opportunities
                }
                for i, assessment in enumerate(assessments)
            ]
        
        return {
            "candidate_count": len(candidate_sites),
            "mission_types": mission_results,
            "analysis_summary": {
                "best_overall_site": self._get_best_overall_site(mission_results),
                "safety_hotspots": self._identify_safety_hotspots(mission_results),
                "science_opportunities": self._identify_science_hotspots(mission_results)
            }
        }
    
    def _discover_earth_analogs(self, bounds: BoundingBox) -> Dict[str, Any]:
        """Discover Earth analogs for Mars region."""
        # Create mock Mars region for analysis
        mars_region_data = self._create_mars_region_data(bounds)
        
        # Find Earth analogs
        import torch

        from ..foundation_models.comparative_planetary import MarsRegion
        
        mars_region = MarsRegion(
            location=MarsCoordinate(
                latitude=(bounds.min_lat + bounds.max_lat) / 2,
                longitude=(bounds.min_lon + bounds.max_lon) / 2
            ),
            imagery=torch.randn(1, 12, 64, 64)  # Mock imagery
        )
        
        analogs = self.comparative_analyzer.find_earth_analogs(
            mars_region, top_k=10
        )
        
        analog_results = []
        for analog in analogs:
            analog_data = {
                "location": {
                    "latitude": analog.location.latitude,
                    "longitude": analog.location.longitude,
                    "name": analog.location_name
                },
                "similarity_score": analog.similarity_score,
                "confidence": analog.confidence,
                "geological_features": analog.geological_features,
                "suitability": analog.suitability_assessment
            }
            analog_results.append(analog_data)
        
        return {
            "analogs": analog_results,
            "mars_region": {
                "center_lat": mars_region.location.latitude,
                "center_lon": mars_region.location.longitude,
                "bounds": asdict(bounds)
            },
            "analysis_summary": {
                "top_analog": analog_results[0] if analog_results else None,
                "avg_similarity": np.mean([a["similarity_score"] for a in analog_results]) if analog_results else 0,
                "feature_diversity": len(set().union(*[a["geological_features"] for a in analog_results]))
            }
        }
    
    def _analyze_embeddings_detailed(self, bounds: BoundingBox) -> Dict[str, Any]:
        """Detailed embedding analysis."""
        embedding_tiles = self.embedding_db.query_embeddings(bounds)
        
        if not embedding_tiles:
            return {"error": "No embedding data available"}
        
        # Analyze embedding patterns
        embeddings = np.array([tile.embedding for tile in embedding_tiles])
        
        # Compute statistics
        stats = {
            "tile_count": len(embedding_tiles),
            "embedding_dimension": embeddings.shape[1],
            "statistics": {
                "mean": embeddings.mean(axis=0).tolist(),
                "std": embeddings.std(axis=0).tolist(),
                "min": embeddings.min(axis=0).tolist(),
                "max": embeddings.max(axis=0).tolist()
            },
            "diversity_metrics": {
                "variance": float(np.var(embeddings)),
                "entropy": self._compute_embedding_entropy(embeddings),
                "clusters": self._detect_embedding_clusters(embeddings)
            }
        }
        
        return stats
    
    def _analyze_terrain_detailed(self, bounds: BoundingBox) -> Dict[str, Any]:
        """Detailed terrain analysis."""
        # Mock terrain analysis
        return {
            "elevation_stats": {
                "min_elevation": -2.5,
                "max_elevation": 8.2,
                "mean_elevation": 2.1,
                "elevation_range": 10.7
            },
            "slope_analysis": {
                "avg_slope": 15.3,
                "max_slope": 47.8,
                "gentle_terrain_pct": 62.5,
                "steep_terrain_pct": 12.3
            },
            "surface_features": {
                "crater_count": 23,
                "channel_length_km": 45.2,
                "volcanic_features": 3,
                "erosion_patterns": ["wind", "thermal"]
            },
            "feature_count": 26
        }
    
    def _generate_candidate_sites(
        self, 
        bounds: BoundingBox, 
        density: int = 10
    ) -> List[MarsCoordinate]:
        """Generate candidate landing sites in region."""
        sites = []
        
        lat_step = (bounds.max_lat - bounds.min_lat) / density
        lon_step = (bounds.max_lon - bounds.min_lon) / density
        
        for i in range(density):
            for j in range(density):
                lat = bounds.min_lat + i * lat_step
                lon = bounds.min_lon + j * lon_step
                sites.append(MarsCoordinate(latitude=lat, longitude=lon))
        
        return sites
    
    def _create_mars_region_data(self, bounds: BoundingBox) -> Dict[str, Any]:
        """Create Mars region data for analysis."""
        return {
            "bounds": asdict(bounds),
            "area_km2": self._calculate_area(bounds),
            "center": {
                "lat": (bounds.min_lat + bounds.max_lat) / 2,
                "lon": (bounds.min_lon + bounds.max_lon) / 2
            }
        }
    
    def _calculate_area(self, bounds: BoundingBox) -> float:
        """Calculate area of bounding box in km²."""
        # Simplified area calculation for Mars
        lat_diff = bounds.max_lat - bounds.min_lat
        lon_diff = bounds.max_lon - bounds.min_lon
        
        # Mars radius = 3390 km
        mars_radius = 3390.0
        lat_km = lat_diff * (mars_radius * np.pi / 180)
        lon_km = lon_diff * (mars_radius * np.pi / 180) * np.cos(np.radians((bounds.min_lat + bounds.max_lat) / 2))
        
        return abs(lat_km * lon_km)
    
    def _get_best_overall_site(self, mission_results: Dict[str, List]) -> Dict[str, Any]:
        """Get best overall landing site across all mission types."""
        all_sites = []
        for mission_type, sites in mission_results.items():
            for site in sites:
                site_copy = site.copy()
                site_copy["mission_type"] = mission_type
                all_sites.append(site_copy)
        
        if all_sites:
            best_site = max(all_sites, key=lambda x: x["scores"]["overall"])
            return best_site
        
        return {}
    
    def _identify_safety_hotspots(self, mission_results: Dict[str, List]) -> List[Dict[str, Any]]:
        """Identify areas with safety concerns."""
        hotspots = []
        
        for mission_type, sites in mission_results.items():
            for site in sites:
                if site["scores"]["safety"] < 0.5:  # Low safety threshold
                    hotspot = {
                        "location": site["coordinates"],
                        "safety_score": site["scores"]["safety"],
                        "mission_type": mission_type,
                        "risks": site["risks"]
                    }
                    hotspots.append(hotspot)
        
        return hotspots
    
    def _identify_science_hotspots(self, mission_results: Dict[str, List]) -> List[Dict[str, Any]]:
        """Identify areas with high science value."""
        hotspots = []
        
        for mission_type, sites in mission_results.items():
            for site in sites:
                if site["scores"]["science"] > 0.7:  # High science threshold
                    hotspot = {
                        "location": site["coordinates"],
                        "science_score": site["scores"]["science"],
                        "mission_type": mission_type,
                        "opportunities": site["opportunities"]
                    }
                    hotspots.append(hotspot)
        
        return hotspots
    
    def _compute_embedding_entropy(self, embeddings: np.ndarray) -> float:
        """Compute entropy of embedding distribution."""
        # Discretize embeddings for entropy calculation
        hist, _ = np.histogram(embeddings.flatten(), bins=50)
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        
        return float(-np.sum(hist * np.log2(hist)))
    
    def _detect_embedding_clusters(self, embeddings: np.ndarray) -> int:
        """Detect number of clusters in embeddings."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            # Try different numbers of clusters
            best_k = 1
            best_score = -1
            
            for k in range(2, min(10, len(embeddings))):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            return best_k
        except ImportError:
            return 1  # Default if sklearn not available
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time dashboard metrics."""
        return {
            "performance": asdict(self.performance_metrics),
            "data_streams": {
                stream_name: len(stream_data)
                for stream_name, stream_data in self.data_streams.items()
            },
            "recent_analyses": len([
                analysis for analysis in self.analysis_history
                if (datetime.now() - datetime.fromisoformat(analysis["analysis_timestamp"])).seconds < 300
            ]),
            "system_status": "operational"
        }


class MarsAnalysisDashboard:
    """
    Main dashboard for Mars analysis and visualization.
    """
    
    def __init__(
        self,
        embedding_db: EmbeddingDatabase,
        landing_site_selector: FoundationModelLandingSiteSelector,
        comparative_analyzer: ComparativePlanetaryAnalyzer,
        config: Optional[DashboardConfig] = None
    ):
        self.config = config or DashboardConfig()
        
        # Core components
        self.analytics = RealTimeMarsAnalytics(
            embedding_db, landing_site_selector, comparative_analyzer
        )
        self.job_manager = AnalysisJobManager(self.config)
        
        # Dashboard state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.dashboard_metrics = DashboardMetrics()
        
        # Analysis functions
        self.analyzer_functions = {
            "comprehensive_region": self._analyze_region_wrapper,
            "landing_site": self._landing_site_wrapper,
            "earth_analogs": self._earth_analogs_wrapper
        }
    
    def start_analysis(
        self,
        analysis_type: str,
        parameters: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> str:
        """Start new analysis job."""
        job_id = self.job_manager.submit_job(analysis_type, parameters)
        
        if session_id:
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {"jobs": [], "created_at": datetime.now()}
            self.active_sessions[session_id]["jobs"].append(job_id)
        
        return job_id
    
    def get_analysis_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis job status."""
        job = self.job_manager.get_job_status(job_id)
        if job:
            return asdict(job)
        return None
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get overall dashboard status."""
        # Process pending jobs
        completed_jobs = self.job_manager.process_jobs(self.analyzer_functions)
        
        # Get real-time metrics
        analytics_metrics = self.analytics.get_real_time_metrics()
        
        return {
            "dashboard_metrics": asdict(self.dashboard_metrics),
            "analytics_metrics": analytics_metrics,
            "job_queue_size": len(self.job_manager.job_queue),
            "running_jobs": len(self.job_manager.running_jobs),
            "active_sessions": len(self.active_sessions),
            "recently_completed": len(completed_jobs),
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_region_wrapper(self, **kwargs) -> Dict[str, Any]:
        """Wrapper for comprehensive region analysis."""
        bounds = BoundingBox(**kwargs["bounds"])
        analysis_types = kwargs.get("analysis_types", ["landing_sites", "terrain"])
        
        return self.analytics.analyze_region_comprehensive(bounds, analysis_types)
    
    def _landing_site_wrapper(self, **kwargs) -> Dict[str, Any]:
        """Wrapper for landing site analysis."""
        bounds = BoundingBox(**kwargs["bounds"])
        return self.analytics._analyze_landing_sites_comprehensive(bounds)
    
    def _earth_analogs_wrapper(self, **kwargs) -> Dict[str, Any]:
        """Wrapper for Earth analog discovery."""
        bounds = BoundingBox(**kwargs["bounds"])
        return self.analytics._discover_earth_analogs(bounds)


def create_mars_analysis_dashboard(
    embedding_db: EmbeddingDatabase,
    landing_site_selector: FoundationModelLandingSiteSelector,
    comparative_analyzer: ComparativePlanetaryAnalyzer,
    update_interval: float = 5.0,
    max_concurrent_analyses: int = 5
) -> MarsAnalysisDashboard:
    """
    Factory function to create Mars analysis dashboard.
    
    Args:
        embedding_db: Embedding database
        landing_site_selector: Landing site selector
        comparative_analyzer: Comparative analyzer
        update_interval: Dashboard update interval
        max_concurrent_analyses: Max concurrent analyses
        
    Returns:
        Initialized MarsAnalysisDashboard
    """
    config = DashboardConfig(
        update_interval=update_interval,
        max_concurrent_analyses=max_concurrent_analyses
    )
    
    return MarsAnalysisDashboard(
        embedding_db,
        landing_site_selector, 
        comparative_analyzer,
        config
    )


# Example usage and testing
if __name__ == "__main__":
    from ..foundation_models.comparative_planetary import ComparativePlanetaryAnalyzer
    from ..foundation_models.landing_site_optimization import (
        FoundationModelLandingSiteSelector,
    )
    from ..foundation_models.planetary_scale_embeddings import EmbeddingDatabase

    # Create mock components
    db_path = Path("/tmp/mars_embeddings")
    embedding_db = EmbeddingDatabase(db_path)
    
    # Create dashboard
    dashboard = create_mars_analysis_dashboard(
        embedding_db,
        None,  # Mock selector
        None,  # Mock analyzer
        update_interval=2.0,
        max_concurrent_analyses=3
    )
    
    print("Mars Analysis Dashboard Initialized")
    print("=" * 50)
    
    # Test analysis submission
    test_bounds = {
        "min_lat": 14.0, "max_lat": 15.0,
        "min_lon": 175.0, "max_lon": 176.0
    }
    
    job_id = dashboard.start_analysis(
        "comprehensive_region",
        {
            "bounds": test_bounds,
            "analysis_types": ["terrain", "embeddings"]
        },
        session_id="test_session_1"
    )
    
    print(f"Analysis job submitted: {job_id}")
    
    # Get dashboard status
    status = dashboard.get_dashboard_status()
    print(f"Job queue size: {status['job_queue_size']}")
    print(f"Running jobs: {status['running_jobs']}")
    print(f"Active sessions: {status['active_sessions']}")
    
    print("\nReal-time Mars analysis dashboard ready for deployment!")
