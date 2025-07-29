-- Mars GIS Database Initialization Script
-- This script sets up the initial database schema and extensions for Mars GIS Platform
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS postgis_sfcgal;
CREATE EXTENSION IF NOT EXISTS uuid - ossp;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
-- Create Mars-specific spatial reference systems
-- Mars 2000 coordinate system
INSERT INTO spatial_ref_sys (srid, auth_name, auth_srid, proj4text, srtext)
VALUES (
        949900,
        'MARS',
        2000,
        '+proj=longlat +a=3396190 +b=3376200 +no_defs',
        'GEOGCS["Mars 2000",DATUM["Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190.0,169.894447223612]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    ) ON CONFLICT (srid) DO NOTHING;
-- Create database schema
CREATE SCHEMA IF NOT EXISTS mars_gis;
CREATE SCHEMA IF NOT EXISTS data_sources;
CREATE SCHEMA IF NOT EXISTS analysis;
CREATE SCHEMA IF NOT EXISTS missions;
-- Set search path
ALTER DATABASE mars_gis
SET search_path TO mars_gis,
    data_sources,
    analysis,
    missions,
    public;
-- Create custom types
CREATE TYPE mars_gis.terrain_type AS ENUM (
    'plains',
    'hills',
    'mountains',
    'craters',
    'valleys',
    'polar_ice',
    'sand_dunes',
    'rocky_terrain',
    'lava_flows',
    'impact_ejecta'
);
CREATE TYPE mars_gis.mission_status AS ENUM (
    'planned',
    'active',
    'completed',
    'failed',
    'cancelled'
);
CREATE TYPE mars_gis.data_source AS ENUM (
    'MRO_HiRISE',
    'MRO_CTX',
    'MRO_MARCI',
    'MGS_MOLA',
    'MARS_EXPRESS',
    'ODYSSEY_THEMIS',
    'VIKING',
    'PATHFINDER',
    'MER_SPIRIT',
    'MER_OPPORTUNITY',
    'MSL_CURIOSITY',
    'MARS2020_PERSEVERANCE',
    'INSIGHT'
);
-- Create core tables
CREATE TABLE IF NOT EXISTS mars_gis.missions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    status mars_gis.mission_status DEFAULT 'planned',
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    landing_site GEOMETRY(POINT, 949900),
    mission_boundary GEOMETRY(POLYGON, 949900),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS mars_gis.assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mission_id UUID REFERENCES mars_gis.missions(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    asset_type VARCHAR(100) NOT NULL,
    location GEOMETRY(POINT, 949900),
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS data_sources.mars_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source mars_gis.data_source NOT NULL,
    data_type VARCHAR(100) NOT NULL,
    acquisition_date TIMESTAMP WITH TIME ZONE,
    location GEOMETRY(POINT, 949900),
    bounds GEOMETRY(POLYGON, 949900),
    resolution_meters DECIMAL(10, 6),
    file_path TEXT,
    file_size_bytes BIGINT,
    checksum VARCHAR(64),
    metadata JSONB,
    quality_score DECIMAL(3, 2) CHECK (
        quality_score >= 0
        AND quality_score <= 1
    ),
    processing_level VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS analysis.terrain_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    data_id UUID REFERENCES data_sources.mars_data(id) ON DELETE CASCADE,
    analysis_type VARCHAR(100) NOT NULL,
    location GEOMETRY(POINT, 949900) NOT NULL,
    terrain_classification mars_gis.terrain_type,
    elevation_meters DECIMAL(10, 3),
    slope_degrees DECIMAL(5, 2),
    roughness_index DECIMAL(8, 6),
    confidence_score DECIMAL(3, 2) CHECK (
        confidence_score >= 0
        AND confidence_score <= 1
    ),
    analysis_parameters JSONB,
    results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS analysis.hazard_assessment (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    location GEOMETRY(POINT, 949900) NOT NULL,
    hazard_type VARCHAR(100) NOT NULL,
    severity_level INTEGER CHECK (
        severity_level >= 1
        AND severity_level <= 5
    ),
    probability DECIMAL(3, 2) CHECK (
        probability >= 0
        AND probability <= 1
    ),
    impact_radius_meters DECIMAL(10, 2),
    mitigation_strategies TEXT [],
    assessment_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);
CREATE TABLE IF NOT EXISTS missions.path_planning (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mission_id UUID REFERENCES mars_gis.missions(id) ON DELETE CASCADE,
    route_name VARCHAR(255) NOT NULL,
    start_point GEOMETRY(POINT, 949900) NOT NULL,
    end_point GEOMETRY(POINT, 949900) NOT NULL,
    waypoints GEOMETRY(MULTIPOINT, 949900),
    path_geometry GEOMETRY(LINESTRING, 949900),
    total_distance_meters DECIMAL(12, 2),
    estimated_travel_time_hours DECIMAL(8, 2),
    difficulty_score DECIMAL(3, 2) CHECK (
        difficulty_score >= 0
        AND difficulty_score <= 1
    ),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_missions_location ON mars_gis.missions USING GIST (landing_site);
CREATE INDEX IF NOT EXISTS idx_missions_boundary ON mars_gis.missions USING GIST (mission_boundary);
CREATE INDEX IF NOT EXISTS idx_missions_status ON mars_gis.missions (status);
CREATE INDEX IF NOT EXISTS idx_missions_dates ON mars_gis.missions (start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_assets_location ON mars_gis.assets USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_assets_mission ON mars_gis.assets (mission_id);
CREATE INDEX IF NOT EXISTS idx_assets_type ON mars_gis.assets (asset_type);
CREATE INDEX IF NOT EXISTS idx_mars_data_location ON data_sources.mars_data USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_mars_data_bounds ON data_sources.mars_data USING GIST (bounds);
CREATE INDEX IF NOT EXISTS idx_mars_data_source ON data_sources.mars_data (source);
CREATE INDEX IF NOT EXISTS idx_mars_data_type ON data_sources.mars_data (data_type);
CREATE INDEX IF NOT EXISTS idx_mars_data_date ON data_sources.mars_data (acquisition_date);
CREATE INDEX IF NOT EXISTS idx_terrain_location ON analysis.terrain_analysis USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_terrain_type ON analysis.terrain_analysis (terrain_classification);
CREATE INDEX IF NOT EXISTS idx_terrain_elevation ON analysis.terrain_analysis (elevation_meters);
CREATE INDEX IF NOT EXISTS idx_hazard_location ON analysis.hazard_assessment USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_hazard_type ON analysis.hazard_assessment (hazard_type);
CREATE INDEX IF NOT EXISTS idx_hazard_severity ON analysis.hazard_assessment (severity_level);
CREATE INDEX IF NOT EXISTS idx_path_mission ON missions.path_planning (mission_id);
CREATE INDEX IF NOT EXISTS idx_path_start ON missions.path_planning USING GIST (start_point);
CREATE INDEX IF NOT EXISTS idx_path_end ON missions.path_planning USING GIST (end_point);
CREATE INDEX IF NOT EXISTS idx_path_geometry ON missions.path_planning USING GIST (path_geometry);
-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_modified_time() RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = NOW();
RETURN NEW;
END;
$$ LANGUAGE plpgsql;
-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_missions_modified_time BEFORE
UPDATE ON mars_gis.missions FOR EACH ROW EXECUTE FUNCTION update_modified_time();
CREATE TRIGGER update_assets_modified_time BEFORE
UPDATE ON mars_gis.assets FOR EACH ROW EXECUTE FUNCTION update_modified_time();
CREATE TRIGGER update_mars_data_modified_time BEFORE
UPDATE ON data_sources.mars_data FOR EACH ROW EXECUTE FUNCTION update_modified_time();
CREATE TRIGGER update_path_planning_modified_time BEFORE
UPDATE ON missions.path_planning FOR EACH ROW EXECUTE FUNCTION update_modified_time();
-- Create function for Mars distance calculations
CREATE OR REPLACE FUNCTION mars_distance(point1 GEOMETRY, point2 GEOMETRY) RETURNS DECIMAL AS $$
DECLARE mars_radius DECIMAL := 3396200.0;
-- Mars radius in meters
lat1_rad DECIMAL;
lat2_rad DECIMAL;
delta_lat_rad DECIMAL;
delta_lon_rad DECIMAL;
a DECIMAL;
c DECIMAL;
BEGIN -- Convert to radians
lat1_rad := radians(ST_Y(point1));
lat2_rad := radians(ST_Y(point2));
delta_lat_rad := radians(ST_Y(point2) - ST_Y(point1));
delta_lon_rad := radians(ST_X(point2) - ST_X(point1));
-- Haversine formula adapted for Mars
a := sin(delta_lat_rad / 2) * sin(delta_lat_rad / 2) + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon_rad / 2) * sin(delta_lon_rad / 2);
c := 2 * atan2(sqrt(a), sqrt(1 - a));
RETURN mars_radius * c;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
-- Insert sample data for testing
INSERT INTO mars_gis.missions (name, description, status, landing_site)
VALUES (
        'Olympia Undae Exploration',
        'Exploration of the Olympia Undae dune field in the northern polar region',
        'planned',
        ST_SetSRID(ST_MakePoint(175.4729, -14.5684), 949900)
    ),
    (
        'Gale Crater Analysis',
        'Detailed geological analysis of Gale Crater',
        'active',
        ST_SetSRID(ST_MakePoint(137.8414, -5.4453), 949900)
    ),
    (
        'Valles Marineris Survey',
        'Comprehensive survey of the Valles Marineris canyon system',
        'completed',
        ST_SetSRID(ST_MakePoint(-49.97, 22.5), 949900)
    ) ON CONFLICT (name) DO NOTHING;
-- Create test user (if not exists)
DO $$ BEGIN IF NOT EXISTS (
    SELECT
    FROM pg_catalog.pg_roles
    WHERE rolname = 'mars_test'
) THEN CREATE ROLE mars_test WITH LOGIN PASSWORD 'test_password';
GRANT USAGE ON SCHEMA mars_gis,
    data_sources,
    analysis,
    missions TO mars_test;
GRANT SELECT,
    INSERT,
    UPDATE,
    DELETE ON ALL TABLES IN SCHEMA mars_gis,
    data_sources,
    analysis,
    missions TO mars_test;
GRANT USAGE,
    SELECT ON ALL SEQUENCES IN SCHEMA mars_gis,
    data_sources,
    analysis,
    missions TO mars_test;
END IF;
END $$;
-- Create development functions
CREATE OR REPLACE FUNCTION reset_test_data() RETURNS VOID AS $$ BEGIN TRUNCATE TABLE missions.path_planning CASCADE;
TRUNCATE TABLE analysis.hazard_assessment CASCADE;
TRUNCATE TABLE analysis.terrain_analysis CASCADE;
TRUNCATE TABLE data_sources.mars_data CASCADE;
TRUNCATE TABLE mars_gis.assets CASCADE;
TRUNCATE TABLE mars_gis.missions CASCADE;
-- Reinsert sample missions
INSERT INTO mars_gis.missions (name, description, status, landing_site)
VALUES (
        'Olympia Undae Exploration',
        'Exploration of the Olympia Undae dune field in the northern polar region',
        'planned',
        ST_SetSRID(ST_MakePoint(175.4729, -14.5684), 949900)
    ),
    (
        'Gale Crater Analysis',
        'Detailed geological analysis of Gale Crater',
        'active',
        ST_SetSRID(ST_MakePoint(137.8414, -5.4453), 949900)
    ),
    (
        'Valles Marineris Survey',
        'Comprehensive survey of the Valles Marineris canyon system',
        'completed',
        ST_SetSRID(ST_MakePoint(-49.97, 22.5), 949900)
    );
END;
$$ LANGUAGE plpgsql;
-- Log successful initialization
SELECT 'Mars GIS Database initialized successfully!' AS status;