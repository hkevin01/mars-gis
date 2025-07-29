"""Database connection and session management."""

from typing import Generator

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session, sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    create_engine = sessionmaker = Session = None

from mars_gis.core.config import settings


class DatabaseManager:
    """Database connection manager."""
    
    def __init__(self):
        """Initialize database manager."""
        self.engine = None
        self.SessionLocal = None
        
        if SQLALCHEMY_AVAILABLE:
            self._setup_database()
    
    def _setup_database(self):
        """Set up database connection."""
        if not SQLALCHEMY_AVAILABLE:
            return
            
        self.engine = create_engine(
            settings.DATABASE_URL,
            echo=settings.DATABASE_ECHO,
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def get_db(self) -> Generator[Session, None, None]:
        """Get database session."""
        if not SQLALCHEMY_AVAILABLE or not self.SessionLocal:
            yield None
            return
            
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def create_tables(self):
        """Create all database tables."""
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy not available")
            
        from mars_gis.database.models import Base
        if Base and self.engine:
            Base.metadata.create_all(bind=self.engine)


# Global database manager instance
db_manager = DatabaseManager()

# Convenience function for FastAPI dependency injection
def get_database() -> Generator[Session, None, None]:
    """FastAPI database dependency."""
    yield from db_manager.get_db()
