// Enhanced 3D Visualization Integration for Mars-GIS v3.0
// Phase 3.0 - Option B: Advanced 3D Visualization & VR Integration

export { default as VRMarsExplorer } from '../vr-interface/components/VRMarsExplorer';
export { default as Enhanced3DInterface } from './components/Enhanced3DInterface';
export { default as Mars3DGlobe } from './components/Mars3DGlobe';
export { default as Mars3DTerrain } from './components/Mars3DTerrain';

// Type exports will be added after interface stabilization

// 3D Visualization utilities
export const MARS_CONSTANTS = {
  RADIUS: 3389.5, // km
  ATMOSPHERE_HEIGHT: 150, // km
  GRAVITY: 3.711, // m/s²
  DAY_LENGTH: 24.6229 // hours
};

export const MARS_REGIONS = [
  {
    name: 'Valles Marineris',
    bounds: [-15, -5, -10, 5] as [number, number, number, number],
    description: 'The largest canyon system in the solar system',
    type: 'canyon'
  },
  {
    name: 'Olympus Mons',
    bounds: [15, -140, 25, -130] as [number, number, number, number],
    description: 'The largest volcano on Mars and in the solar system',
    type: 'volcano'
  },
  {
    name: 'Gale Crater',
    bounds: [-6, 137, -4, 139] as [number, number, number, number],
    description: 'Curiosity rover landing site with ancient lake evidence',
    type: 'crater'
  },
  {
    name: 'Meridiani Planum',
    bounds: [-5, 5, 5, 15] as [number, number, number, number],
    description: 'Opportunity rover landing site with hematite deposits',
    type: 'plain'
  },
  {
    name: 'Chryse Planitia',
    bounds: [18, -55, 28, -45] as [number, number, number, number],
    description: 'Viking 1 landing site in the northern lowlands',
    type: 'plain'
  },
  {
    name: 'Acidalia Planitia',
    bounds: [40, -30, 60, -10] as [number, number, number, number],
    description: 'Northern plains region with ancient flood evidence',
    type: 'plain'
  }
];

// Color schemes for Mars visualization
export const MARS_COLOR_SCHEMES = {
  elevation: {
    low: '#4a1a0d',     // Dark brown for low elevations
    mid: '#cd7f32',     // Bronze for mid elevations
    high: '#f4a460'     // Sandy brown for high elevations
  },
  thermal: {
    cold: '#1e3a8a',    // Deep blue for cold
    warm: '#dc2626',    // Red for warm
    hot: '#fbbf24'      // Amber for hot
  },
  composition: {
    basalt: '#2d1b69',  // Dark purple
    olivine: '#166534', // Dark green
    pyroxene: '#7c2d12', // Dark brown
    feldspar: '#78716c'  // Gray
  }
};

// NASA data source configurations
export const NASA_DATA_SOURCES = {
  mola: {
    name: 'Mars Orbiter Laser Altimeter',
    resolution: '463m/pixel',
    coverage: 'Global',
    baseUrl: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_ClrShade_merge_global_463m'
  },
  themis: {
    name: 'Thermal Emission Imaging System',
    resolution: '100m/pixel',
    coverage: 'Near-global',
    baseUrl: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_2001_Odyssey_THEMIS_DayIR_Mosaic_global_100m'
  },
  hirise: {
    name: 'High Resolution Imaging Science Experiment',
    resolution: '0.25-0.5m/pixel',
    coverage: 'Selected regions',
    baseUrl: 'https://trek.nasa.gov/tiles/Mars/Equirectangular/HiRISE'
  },
  ctx: {
    name: 'Context Camera',
    resolution: '6m/pixel',
    coverage: 'Extensive',
    baseUrl: 'https://trek.nasa.gov/tiles/Mars/Equirectangular/CTX'
  }
};

// 3D Visualization configuration
export const VISUALIZATION_CONFIG = {
  performance: {
    targetFPS: 60,
    maxVertices: 500000,
    textureResolution: '4K',
    enableLOD: true
  },
  rendering: {
    antialias: true,
    shadows: true,
    atmosphere: true,
    postProcessing: false
  },
  interaction: {
    enableZoom: true,
    enableRotation: true,
    enablePan: true,
    clickSelection: true
  }
};

// VR Configuration
export const VR_CONFIG = {
  devices: ['oculus', 'vive', 'pico', 'quest'],
  features: {
    handTracking: true,
    roomScale: true,
    teleportation: true,
    voiceCommands: false
  },
  performance: {
    targetFPS: 90,
    adaptiveQuality: true,
    maxDrawCalls: 1000
  }
};

// Utility functions for 3D visualization
export const convertLatLonToCartesian = (lat: number, lon: number, radius: number = MARS_CONSTANTS.RADIUS) => {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);

  return {
    x: radius * Math.sin(phi) * Math.cos(theta),
    y: radius * Math.cos(phi),
    z: radius * Math.sin(phi) * Math.sin(theta)
  };
};

export const convertCartesianToLatLon = (x: number, y: number, z: number, radius: number = MARS_CONSTANTS.RADIUS) => {
  const r = Math.sqrt(x * x + y * y + z * z);
  const lat = 90 - (Math.acos(y / r) * 180 / Math.PI);
  const lon = (Math.atan2(z, x) * 180 / Math.PI) - 180;

  return { lat, lon };
};

export const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number) => {
  const R = MARS_CONSTANTS.RADIUS;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a =
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
};

// Export default configuration object
const Mars3DConfig = {
  MARS_CONSTANTS,
  MARS_REGIONS,
  MARS_COLOR_SCHEMES,
  NASA_DATA_SOURCES,
  VISUALIZATION_CONFIG,
  VR_CONFIG
};

export default Mars3DConfig;
