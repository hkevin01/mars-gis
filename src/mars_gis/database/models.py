"""Database models for MARS-GIS."""

from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from geoalchemy2 import Geometry
    from sqlalchemy import (
        Boolean,
        Column,
        DateTime,
        Float,
        ForeignKey,
        Integer,
        MetaData,
        String,
        Table,
        Text,
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    # Fallback base class
    class declarative_base:
        def __init__(self):
            pass

    Column = Integer = String = Float = DateTime = Boolean = Text = None
    ForeignKey = Table = MetaData = relationship = Geometry = None


Base = declarative_base() if SQLALCHEMY_AVAILABLE else None


class MarsFeature(Base if SQLALCHEMY_AVAILABLE else object):
    """Mars geological features model."""
    
    if SQLALCHEMY_AVAILABLE:
        __tablename__ = "mars_features"
        
        id = Column(Integer, primary_key=True, index=True)
        name = Column(String(255), nullable=False)
        feature_type = Column(String(100), nullable=False)  # crater, mountain, valley
        geometry = Column(Geometry("POLYGON", srid=4326))
        elevation_min = Column(Float)
        elevation_max = Column(Float)
        diameter_km = Column(Float)
        age_estimate = Column(String(100))  # geological age
        description = Column(Text)
        discovered_by = Column(String(255))
        discovery_date = Column(DateTime)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # Relationships
        images = relationship("SatelliteImage", back_populates="feature")


class SatelliteImage(Base if SQLALCHEMY_AVAILABLE else object):
    """Satellite imagery metadata model."""
    
    if SQLALCHEMY_AVAILABLE:
        __tablename__ = "satellite_images"
        
        id = Column(Integer, primary_key=True, index=True)
        filename = Column(String(255), nullable=False, unique=True)
        mission_name = Column(String(100), nullable=False)  # MRO, MGS, etc.
        instrument = Column(String(100))  # HiRISE, CTX, MOLA
        acquisition_date = Column(DateTime, nullable=False)
        geometry = Column(Geometry("POLYGON", srid=4326))  # coverage area
        center_lat = Column(Float)
        center_lon = Column(Float)
        resolution_meters = Column(Float)
        file_size_mb = Column(Float)
        file_path = Column(String(500))
        processing_level = Column(String(50))  # raw, calibrated, derived
        quality_score = Column(Float)  # 0-1 quality assessment
        metadata_json = Column(Text)  # additional metadata as JSON
        created_at = Column(DateTime, default=datetime.utcnow)
        
        # Foreign keys
        feature_id = Column(Integer, ForeignKey("mars_features.id"))
        
        # Relationships
        feature = relationship("MarsFeature", back_populates="images")


class MissionPlan(Base if SQLALCHEMY_AVAILABLE else object):
    """Mission planning data model."""
    
    if SQLALCHEMY_AVAILABLE:
        __tablename__ = "mission_plans"
        
        id = Column(Integer, primary_key=True, index=True)
        name = Column(String(255), nullable=False)
        mission_type = Column(String(100))  # landing, rover, orbital
        target_geometry = Column(Geometry("POINT", srid=4326))
        target_lat = Column(Float)
        target_lon = Column(Float)
        landing_ellipse = Column(Geometry("POLYGON", srid=4326))
        safety_score = Column(Float)  # 0-1 safety assessment
        scientific_priority = Column(Float)  # 0-1 priority score
        terrain_difficulty = Column(Float)  # 0-1 difficulty score
        planned_date = Column(DateTime)
        status = Column(String(50), default="planned")  # planned, active, completed
        notes = Column(Text)
        created_by = Column(String(255))
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TerrainAnalysis(Base if SQLALCHEMY_AVAILABLE else object):
    """AI/ML terrain analysis results model."""
    
    if SQLALCHEMY_AVAILABLE:
        __tablename__ = "terrain_analysis"
        
        id = Column(Integer, primary_key=True, index=True)
        image_id = Column(Integer, ForeignKey("satellite_images.id"))
        analysis_type = Column(String(100))  # classification, hazard_detection
        model_name = Column(String(255))
        model_version = Column(String(50))
        confidence_score = Column(Float)  # 0-1 confidence
        results_json = Column(Text)  # analysis results as JSON
        processing_time_ms = Column(Integer)
        geometry = Column(Geometry("POLYGON", srid=4326))  # analyzed area
        created_at = Column(DateTime, default=datetime.utcnow)
        
        # Relationships
        image = relationship("SatelliteImage")


class User(Base if SQLALCHEMY_AVAILABLE else object):
    """User authentication model."""
    
    if SQLALCHEMY_AVAILABLE:
        __tablename__ = "users"
        
        id = Column(Integer, primary_key=True, index=True)
        username = Column(String(100), unique=True, nullable=False)
        email = Column(String(255), unique=True, nullable=False)
        hashed_password = Column(String(255), nullable=False)
        full_name = Column(String(255))
        organization = Column(String(255))
        role = Column(String(50), default="user")  # user, admin, researcher
        is_active = Column(Boolean, default=True)
        last_login = Column(DateTime)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Create all tables
def create_tables(engine):
    """Create all database tables."""
    if SQLALCHEMY_AVAILABLE and Base:
        Base.metadata.create_all(bind=engine)
    else:
        raise ImportError("SQLAlchemy not available. Please install requirements.")


# Database utility functions
def get_mars_features_by_type(session, feature_type: str):
    """Get Mars features by type."""
    if not SQLALCHEMY_AVAILABLE:
        return []
    return session.query(MarsFeature).filter(
        MarsFeature.feature_type == feature_type
    ).all()


def get_images_by_mission(session, mission_name: str):
    """Get satellite images by mission."""
    if not SQLALCHEMY_AVAILABLE:
        return []
    return session.query(SatelliteImage).filter(
        SatelliteImage.mission_name == mission_name
    ).all()


def get_landing_sites_by_safety_score(session, min_score: float):
    """Get mission plans by minimum safety score."""
    if not SQLALCHEMY_AVAILABLE:
        return []
    return session.query(MissionPlan).filter(
        MissionPlan.safety_score >= min_score
    ).all()
