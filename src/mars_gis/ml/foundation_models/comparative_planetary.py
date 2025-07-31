"""
Comparative Planetary Analysis Module

This module implements Earth-Mars comparative analysis using foundation models,
enabling discovery of Earth analogs for Mars regions and cross-planetary 
geological feature comparison.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mars_gis.core.types import MarsCoordinate

from .earth_mars_transfer import EarthMarsFoundationModel


@dataclass
class EarthCoordinate:
    """Earth coordinate representation."""
    latitude: float
    longitude: float
    elevation: Optional[float] = None


@dataclass
class EarthAnalog:
    """Earth analog location with similarity metrics."""
    location: EarthCoordinate
    similarity_score: float
    geological_features: List[str]
    confidence_score: float
    earth_imagery: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MarsRegion:
    """Mars region representation with imagery and metadata."""
    location: MarsCoordinate
    imagery: torch.Tensor
    geological_context: Optional[List[str]] = None
    temporal_info: Optional[str] = None
    data_quality: Optional[float] = None


class EarthEmbeddingDatabase:
    """
    Database for storing and querying Earth location embeddings.
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.embeddings: Optional[torch.Tensor] = None
        self.metadata: List[Dict[str, Any]] = []
        self.locations: List[EarthCoordinate] = []
        
    def add_location(
        self,
        location: EarthCoordinate,
        embedding: torch.Tensor,
        metadata: Dict[str, Any]
    ):
        """Add a location to the database."""
        if self.embeddings is None:
            self.embeddings = embedding.unsqueeze(0)
        else:
            self.embeddings = torch.cat([self.embeddings, embedding.unsqueeze(0)])
            
        self.locations.append(location)
        self.metadata.append(metadata)
    
    def query_similar(
        self,
        query_embedding: torch.Tensor,
        similarity_threshold: float = 0.85,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query for similar Earth locations.
        
        Args:
            query_embedding: Mars region embedding to match
            similarity_threshold: Minimum similarity score
            top_k: Maximum number of results
            
        Returns:
            List of similar Earth location candidates
        """
        if self.embeddings is None:
            return []
            
        # Calculate cosine similarities
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.embeddings,
            dim=1
        )
        
        # Filter by threshold
        valid_indices = torch.where(similarities >= similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            return []
            
        # Sort by similarity
        sorted_indices = valid_indices[
            torch.argsort(similarities[valid_indices], descending=True)
        ][:top_k]
        
        # Return candidate information
        candidates = []
        for idx in sorted_indices:
            candidates.append({
                'location': self.locations[idx.item()],
                'embedding': self.embeddings[idx.item()],
                'similarity': similarities[idx.item()].item(),
                'metadata': self.metadata[idx.item()]
            })
            
        return candidates
    
    def save_database(self, filepath: str):
        """Save database to disk."""
        database_data = {
            'embeddings': self.embeddings,
            'locations': self.locations,
            'metadata': self.metadata,
            'embedding_dim': self.embedding_dim
        }
        torch.save(database_data, filepath)
    
    def load_database(self, filepath: str):
        """Load database from disk."""
        database_data = torch.load(filepath)
        self.embeddings = database_data['embeddings']
        self.locations = database_data['locations']
        self.metadata = database_data['metadata']
        self.embedding_dim = database_data['embedding_dim']


class GeologicalFeatureExtractor(nn.Module):
    """
    Neural network for extracting geological features from embeddings.
    """
    
    def __init__(self, embedding_dim: int = 64, num_features: int = 20):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_features),
            nn.Sigmoid()  # Feature presence probabilities
        )
        
        # Geological feature names (example)
        self.feature_names = [
            'volcanic_terrain', 'sedimentary_layers', 'impact_craters',
            'channel_networks', 'alluvial_fans', 'aeolian_features',
            'tectonic_structures', 'hydrothermal_deposits', 'carbonate_deposits',
            'clay_minerals', 'sulfate_deposits', 'iron_oxides',
            'polar_layered_deposits', 'chaos_terrain', 'outflow_channels',
            'valley_networks', 'delta_deposits', 'landing_hazards',
            'slope_instability', 'dust_cover'
        ]
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Extract geological features from embeddings."""
        return self.feature_extractor(embeddings)
    
    def get_feature_names(self) -> List[str]:
        """Return list of geological feature names."""
        return self.feature_names


class ComparativePlanetaryAnalyzer:
    """
    Main class for comparing Earth and Mars geological features 
    using foundation models.
    """
    
    def __init__(
        self,
        foundation_model: EarthMarsFoundationModel,
        earth_database: Optional[EarthEmbeddingDatabase] = None
    ):
        self.foundation_model = foundation_model
        self.earth_database = earth_database or EarthEmbeddingDatabase()
        
        # Geological feature extractor
        self.feature_extractor = GeologicalFeatureExtractor()
        
        # Confidence estimation network
        self.confidence_estimator = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def find_earth_analogs(
        self, 
        mars_region: MarsRegion,
        similarity_threshold: float = 0.75,
        top_k: int = 5
    ) -> List[EarthAnalog]:
        """
        Find Earth locations similar to Mars regions.
        
        Args:
            mars_region: Mars region to analyze
            similarity_threshold: Minimum similarity for matches
            top_k: Maximum number of analogs to return
            
        Returns:
            List of Earth analog locations with similarity scores
        """
        # Get Mars embedding using foundation model
        with torch.no_grad():
            mars_embedding = self.foundation_model.get_mars_embedding(
                mars_region.imagery
            )
        
        # Query Earth embedding database
        earth_candidates = self.earth_database.query_similar(
            mars_embedding,
            similarity_threshold=similarity_threshold,
            top_k=top_k * 2  # Get more candidates for filtering
        )
        
        analogs = []
        for candidate in earth_candidates[:top_k]:
            # Extract geological features for comparison
            mars_features = self.feature_extractor(mars_embedding)
            earth_features = self.feature_extractor(
                candidate['embedding'].unsqueeze(0)
            )
            
            # Calculate feature similarity
            feature_similarity = F.cosine_similarity(
                mars_features, earth_features, dim=1
            ).item()
            
            # Estimate confidence
            confidence = self.confidence_estimator(mars_embedding).item()
            
            # Create analog object
            analog = EarthAnalog(
                location=candidate['location'],
                similarity_score=candidate['similarity'],
                geological_features=self._extract_feature_list(earth_features),
                confidence_score=confidence * feature_similarity,
                metadata=candidate['metadata']
            )
            
            analogs.append(analog)
        
        # Sort by combined score (similarity * confidence)
        analogs.sort(
            key=lambda x: x.similarity_score * x.confidence_score,
            reverse=True
        )
        
        return analogs
    
    def _extract_feature_list(self, feature_tensor: torch.Tensor) -> List[str]:
        """Extract list of present geological features."""
        feature_probs = feature_tensor.squeeze()
        feature_names = self.feature_extractor.get_feature_names()
        
        # Use threshold of 0.5 for feature presence
        present_features = []
        for i, prob in enumerate(feature_probs):
            if prob > 0.5 and i < len(feature_names):
                present_features.append(feature_names[i])
                
        return present_features
    
    def compare_geological_contexts(
        self,
        mars_region: MarsRegion,
        earth_analog: EarthAnalog
    ) -> Dict[str, Any]:
        """
        Compare geological contexts between Mars region and Earth analog.
        
        Args:
            mars_region: Mars region for comparison
            earth_analog: Earth analog location
            
        Returns:
            Dictionary with detailed comparison results
        """
        # Get embeddings
        with torch.no_grad():
            mars_embedding = self.foundation_model.get_mars_embedding(
                mars_region.imagery
            )
            
        # Extract features
        mars_features = self.feature_extractor(mars_embedding)
        
        # Create comparison report
        comparison = {
            'overall_similarity': earth_analog.similarity_score,
            'confidence': earth_analog.confidence_score,
            'geological_features': {
                'mars_features': mars_region.geological_context or [],
                'earth_features': earth_analog.geological_features,
                'common_features': list(set(
                    mars_region.geological_context or []
                ).intersection(set(earth_analog.geological_features)))
            },
            'feature_analysis': self._analyze_feature_differences(
                mars_features, earth_analog
            ),
            'suitability_assessment': self._assess_analog_suitability(
                mars_region, earth_analog
            )
        }
        
        return comparison
    
    def _analyze_feature_differences(
        self,
        mars_features: torch.Tensor,
        earth_analog: EarthAnalog
    ) -> Dict[str, float]:
        """Analyze differences in geological features."""
        feature_names = self.feature_extractor.get_feature_names()
        mars_probs = mars_features.squeeze()
        
        analysis = {}
        for i, feature_name in enumerate(feature_names):
            if i < len(mars_probs):
                mars_prob = mars_probs[i].item()
                earth_present = feature_name in earth_analog.geological_features
                earth_prob = 1.0 if earth_present else 0.0
                
                analysis[feature_name] = {
                    'mars_probability': mars_prob,
                    'earth_probability': earth_prob,
                    'difference': abs(mars_prob - earth_prob)
                }
                
        return analysis
    
    def _assess_analog_suitability(
        self,
        mars_region: MarsRegion,
        earth_analog: EarthAnalog
    ) -> Dict[str, Any]:
        """Assess how suitable Earth analog is for Mars mission training."""
        suitability_factors = {
            'terrain_similarity': earth_analog.similarity_score,
            'accessibility': 1.0,  # Placeholder - would assess actual accessibility
            'safety': 1.0,  # Placeholder - safety assessment
            'logistical_feasibility': 0.8,  # Placeholder
            'scientific_value': earth_analog.confidence_score
        }
        
        # Calculate overall suitability score
        overall_score = np.mean(list(suitability_factors.values()))
        
        # Determine recommendation
        if overall_score > 0.8:
            recommendation = "Highly Recommended"
        elif overall_score > 0.6:
            recommendation = "Recommended"
        elif overall_score > 0.4:
            recommendation = "Conditionally Recommended"
        else:
            recommendation = "Not Recommended"
            
        return {
            'factors': suitability_factors,
            'overall_score': overall_score,
            'recommendation': recommendation
        }
    
    def generate_analog_report(
        self,
        mars_region: MarsRegion,
        analogs: List[EarthAnalog]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive report for Earth analogs.
        
        Args:
            mars_region: Target Mars region
            analogs: List of Earth analogs
            
        Returns:
            Comprehensive analysis report
        """
        report = {
            'mars_region': {
                'location': mars_region.location,
                'geological_context': mars_region.geological_context,
                'data_quality': mars_region.data_quality
            },
            'analogs': [],
            'summary': {
                'num_analogs_found': len(analogs),
                'best_analog_similarity': max(
                    [a.similarity_score for a in analogs]
                ) if analogs else 0.0,
                'average_confidence': np.mean(
                    [a.confidence_score for a in analogs]
                ) if analogs else 0.0
            }
        }
        
        # Add detailed analog information
        for i, analog in enumerate(analogs):
            comparison = self.compare_geological_contexts(mars_region, analog)
            
            analog_info = {
                'rank': i + 1,
                'location': analog.location,
                'similarity_score': analog.similarity_score,
                'confidence_score': analog.confidence_score,
                'geological_features': analog.geological_features,
                'detailed_comparison': comparison,
                'coordinates': {
                    'latitude': analog.location.latitude,
                    'longitude': analog.location.longitude,
                    'elevation': analog.location.elevation
                }
            }
            
            report['analogs'].append(analog_info)
            
        return report
    
    def build_earth_database_from_samples(
        self,
        earth_samples: List[Tuple[torch.Tensor, EarthCoordinate, Dict]]
    ):
        """
        Build Earth embedding database from sample data.
        
        Args:
            earth_samples: List of (imagery, location, metadata) tuples
        """
        for imagery, location, metadata in earth_samples:
            # Generate Earth embedding using foundation model
            with torch.no_grad():
                earth_features = self.foundation_model.earth_encoder(
                    imagery.unsqueeze(0)
                )
                
            # Add to database
            self.earth_database.add_location(
                location, earth_features.squeeze(0), metadata
            )


def create_comparative_analyzer(
    foundation_model: EarthMarsFoundationModel,
    earth_database_path: Optional[str] = None
) -> ComparativePlanetaryAnalyzer:
    """
    Factory function to create comparative planetary analyzer.
    
    Args:
        foundation_model: Trained Earth-Mars foundation model
        earth_database_path: Optional path to Earth embedding database
        
    Returns:
        Initialized ComparativePlanetaryAnalyzer
    """
    earth_db = EarthEmbeddingDatabase()
    
    if earth_database_path and Path(earth_database_path).exists():
        earth_db.load_database(earth_database_path)
        
    analyzer = ComparativePlanetaryAnalyzer(foundation_model, earth_db)
    
    return analyzer


# Example usage and testing
if __name__ == "__main__":
    # This would typically be run with a trained foundation model
    from .earth_mars_transfer import create_earth_mars_foundation_model

    # Create foundation model
    foundation_model = create_earth_mars_foundation_model()
    
    # Create analyzer
    analyzer = create_comparative_analyzer(foundation_model)
    
    # Create sample Mars region
    mars_imagery = torch.randn(1, 12, 64, 64)
    mars_location = MarsCoordinate(latitude=14.5, longitude=175.9)
    mars_region = MarsRegion(
        location=mars_location,
        imagery=mars_imagery,
        geological_context=['craters', 'sedimentary_layers', 'channel_networks']
    )
    
    # Add sample Earth data to database
    earth_samples = []
    for i in range(10):
        earth_imagery = torch.randn(10, 64, 64)
        earth_location = EarthCoordinate(
            latitude=40.0 + i, 
            longitude=-110.0 + i
        )
        metadata = {
            'name': f'Earth_Site_{i}',
            'geological_features': ['sedimentary_layers', 'channel_networks']
        }
        earth_samples.append((earth_imagery, earth_location, metadata))
    
    analyzer.build_earth_database_from_samples(earth_samples)
    
    # Find Earth analogs
    analogs = analyzer.find_earth_analogs(mars_region, top_k=3)
    
    print(f"Found {len(analogs)} Earth analogs:")
    for i, analog in enumerate(analogs):
        print(f"{i+1}. Similarity: {analog.similarity_score:.3f}, "
              f"Confidence: {analog.confidence_score:.3f}")
        print(f"   Features: {analog.geological_features}")
    
    # Generate comprehensive report
    if analogs:
        report = analyzer.generate_analog_report(mars_region, analogs)
        print(f"\nReport Summary:")
        print(f"Best similarity: {report['summary']['best_analog_similarity']:.3f}")
        print(f"Average confidence: {report['summary']['average_confidence']:.3f}")
