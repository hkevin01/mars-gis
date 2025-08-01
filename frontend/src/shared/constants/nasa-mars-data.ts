// NASA Mars Web Map Service (WMS) integration
export const NASA_MARS_WMS_SERVICES = {
  // NASA Mars Trek WMS endpoints
  trek: {
    baseUrl: 'https://trek.nasa.gov/mars/TrekWS/rest/cat/Mars',
    layers: {
      mola_elevation: 'Mars_MGS_MOLA_DEM_mosaic_global_463m',
      mola_colorshade: 'Mars_MGS_MOLA_ClrShade_merge_global_463m',
      themis_ir: 'Mars_ODY_THEMIS_IR_Day_ClrShade_global_100m',
      themis_thermal: 'Mars_MGS_TES_Thermal_Inertia_mosaic_global_7410m',
      ctx_mosaic: 'Mars_MRO_CTX_mosaic_global_24m',
    }
  },

  // USGS Astrogeology WMS
  usgs: {
    baseUrl: 'https://astrowebmaps.wr.usgs.gov/webmapatlas',
    layers: {
      viking_mdim: 'Mars_Viking_MDIM21_ClrMosaic_global_232m',
      themis_day: 'Mars_ODY_THEMIS_IR_Day_global_100m',
      themis_night: 'Mars_ODY_THEMIS_IR_Night_global_100m',
    }
  }
};

// Mars coordinate reference systems
export const MARS_CRS = {
  geographic: 'EPSG:4326',
  mars2000: 'IAU:49900', // Mars 2000 coordinate system
  equirectangular: 'EPSG:32767'
};

// NASA Mars data endpoints for real-time data
export const NASA_MARS_DATA_APIS = {
  // Mars Weather from InSight
  insight_weather: 'https://api.nasa.gov/insight_weather/?api_key=',

  // Mars Rover Photos
  curiosity_photos: 'https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos',
  perseverance_photos: 'https://api.nasa.gov/mars-photos/api/v1/rovers/perseverance/photos',

  // Mars atmospheric data
  atmospheric: 'https://api.mars.nasa.gov/weather',

  // Planetary Data System (PDS)
  pds_imaging: 'https://pds-imaging.jpl.nasa.gov/api/mars'
};

// Mars location database with scientific accuracy
export const ENHANCED_MARS_LOCATIONS = [
  {
    id: 1,
    name: 'Olympus Mons',
    lat: 18.65,
    lon: -133.8,
    type: 'volcano',
    description: 'Largest volcano in the solar system - 21.9 km high, 600 km diameter',
    elevation: 21287,
    scientificImportance: 'Geological formation studies',
    missions: ['Mars Express', 'Mars Global Surveyor'],
    dataAvailable: ['elevation', 'thermal', 'imagery']
  },
  {
    id: 2,
    name: 'Valles Marineris',
    lat: -14,
    lon: -59,
    type: 'canyon',
    description: 'Massive canyon system - 4000 km long, 7 km deep',
    elevation: -7000,
    scientificImportance: 'Geological history and water activity',
    missions: ['Viking', 'Mars Global Surveyor', 'Mars Express'],
    dataAvailable: ['elevation', 'imagery', 'spectral']
  },
  {
    id: 3,
    name: 'Gale Crater',
    lat: -5.4,
    lon: 137.8,
    type: 'crater',
    description: 'Curiosity rover landing site with ancient lake evidence',
    elevation: -4500,
    scientificImportance: 'Astrobiology and past habitability',
    missions: ['Curiosity Rover', 'MRO'],
    dataAvailable: ['elevation', 'imagery', 'spectral', 'chemistry']
  },
  {
    id: 4,
    name: 'Jezero Crater',
    lat: 18.44,
    lon: 77.45,
    type: 'crater',
    description: 'Perseverance rover landing site with ancient river delta',
    elevation: -2500,
    scientificImportance: 'Sample collection for Mars Sample Return',
    missions: ['Perseverance Rover', 'Ingenuity Helicopter'],
    dataAvailable: ['elevation', 'imagery', 'spectral', 'samples']
  },
  {
    id: 5,
    name: 'Acidalia Planitia',
    lat: 46.7,
    lon: -29.8,
    type: 'plain',
    description: 'Northern lowlands region with potential subsurface ice',
    elevation: -4000,
    scientificImportance: 'Climate studies and ice deposits',
    missions: ['Mars Global Surveyor', 'Mars Express'],
    dataAvailable: ['elevation', 'thermal', 'radar']
  },
  {
    id: 6,
    name: 'Hellas Planitia',
    lat: -42.4,
    lon: 70.5,
    type: 'basin',
    description: 'Largest impact crater on Mars - 2300 km diameter',
    elevation: -8200,
    scientificImportance: 'Impact processes and atmospheric dynamics',
    missions: ['Viking', 'Mars Global Surveyor'],
    dataAvailable: ['elevation', 'atmospheric', 'thermal']
  },
  {
    id: 7,
    name: 'North Polar Ice Cap',
    lat: 85,
    lon: 0,
    type: 'ice',
    description: 'Permanent water ice cap with seasonal CO2 ice',
    elevation: -5000,
    scientificImportance: 'Climate history and water cycle',
    missions: ['Mars Global Surveyor', 'Mars Express', 'MRO'],
    dataAvailable: ['elevation', 'thermal', 'spectral', 'radar']
  },
  {
    id: 8,
    name: 'South Polar Ice Cap',
    lat: -85,
    lon: 0,
    type: 'ice',
    description: 'Permanent water ice with CO2 ice deposits',
    elevation: -6000,
    scientificImportance: 'Climate variability and ice dynamics',
    missions: ['Mars Global Surveyor', 'Mars Express'],
    dataAvailable: ['elevation', 'thermal', 'spectral']
  },
  {
    id: 9,
    name: 'Tharsis Volcanic Province',
    lat: 0,
    lon: -100,
    type: 'volcano',
    description: 'Major volcanic region with four large shield volcanoes',
    elevation: 10000,
    scientificImportance: 'Volcanic processes and crustal evolution',
    missions: ['Mars Global Surveyor', 'Mars Express'],
    dataAvailable: ['elevation', 'thermal', 'imagery', 'gravity']
  },
  {
    id: 10,
    name: 'Chryse Planitia',
    lat: 20,
    lon: -50,
    type: 'plain',
    description: 'Viking 1 landing site with evidence of ancient flooding',
    elevation: -3000,
    scientificImportance: 'Early Mars exploration and flood geology',
    missions: ['Viking 1', 'Mars Global Surveyor'],
    dataAvailable: ['elevation', 'imagery', 'chemistry']
  },
  {
    id: 11,
    name: 'Utopia Planitia',
    lat: 48.97,
    lon: 117.63,
    type: 'plain',
    description: 'Viking 2 landing site with subsurface ice detection',
    elevation: -4000,
    scientificImportance: 'Subsurface ice and past climate',
    missions: ['Viking 2', 'Mars Express'],
    dataAvailable: ['elevation', 'thermal', 'radar']
  },
  {
    id: 12,
    name: 'Mariner Valley',
    lat: -13.9,
    lon: -59.2,
    type: 'canyon',
    description: 'Part of Valles Marineris system with layered deposits',
    elevation: -6000,
    scientificImportance: 'Sedimentary processes and water history',
    missions: ['Mariner 9', 'Viking', 'Mars Global Surveyor'],
    dataAvailable: ['elevation', 'imagery', 'spectral']
  }
];

// Mars atmospheric conditions (real-time simulation)
export const MARS_ATMOSPHERIC_DATA = {
  temperature: {
    min: -143,  // Celsius, polar winter
    max: 35,    // Celsius, equatorial summer
    average: -80
  },
  pressure: {
    min: 0.4,   // kPa, high elevation
    max: 0.87,  // kPa, low elevation
    average: 0.6
  },
  atmosphere: {
    co2: 95.32,      // %
    nitrogen: 2.7,   // %
    argon: 1.6,      // %
    oxygen: 0.13,    // %
    other: 0.25      // %
  },
  wind: {
    average: 10,     // m/s
    max: 60,         // m/s (dust storms)
    direction: 'variable'
  },
  dust: {
    opacity: 0.2,    // normal conditions
    stormOpacity: 5.0 // global dust storm
  }
};

// Scientific instruments and data types
export const MARS_INSTRUMENTS = {
  mola: {
    name: 'Mars Orbiter Laser Altimeter',
    mission: 'Mars Global Surveyor',
    dataTypes: ['elevation', 'surface_roughness', 'slope'],
    resolution: '463m'
  },
  themis: {
    name: 'Thermal Emission Imaging System',
    mission: 'Mars Odyssey',
    dataTypes: ['thermal_infrared', 'visible'],
    resolution: '100m'
  },
  ctx: {
    name: 'Context Camera',
    mission: 'Mars Reconnaissance Orbiter',
    dataTypes: ['high_resolution_imagery'],
    resolution: '6m'
  },
  hirise: {
    name: 'High Resolution Imaging Science Experiment',
    mission: 'Mars Reconnaissance Orbiter',
    dataTypes: ['ultra_high_resolution_imagery'],
    resolution: '25cm'
  }
};
