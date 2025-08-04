import type { MarsLocation } from '../types/mars-types';

// Mars-specific constants and data
export const MARS_RADIUS = 3396190; // meters

// Mars layer configurations
export const MARS_LAYERS = {
  elevation: {
    name: 'MOLA Elevation',
    description: 'Mars Orbiter Laser Altimeter elevation data',
    type: 'elevation',
    color: '#8B4513',
    visible: true,
    opacity: 0.8
  },
  imagery: {
    name: 'CTX Global Mosaic',
    description: 'Mars Reconnaissance Orbiter Context Camera global mosaic',
    type: 'imagery',
    color: '#CD853F',
    visible: true,
    opacity: 1.0
  },
  thermal: {
    name: 'THEMIS Thermal IR',
    description: 'Thermal Emission Imaging System infrared data',
    type: 'thermal',
    color: '#FF4500',
    visible: false,
    opacity: 0.6
  },
  geology: {
    name: 'Geological Map',
    description: 'USGS geological mapping of Mars',
    type: 'geology',
    color: '#DAA520',
    visible: false,
    opacity: 0.7
  },
  atmosphere: {
    name: 'Atmospheric Data',
    description: 'Mars atmospheric pressure and density',
    type: 'atmosphere',
    color: '#4169E1',
    visible: false,
    opacity: 0.5
  }
} as const;

// Famous Mars locations
export const MARS_LOCATIONS: MarsLocation[] = [
  { id: '1', name: 'Olympus Mons', lat: 18.65, lon: -133.8, type: 'volcano', description: 'Largest volcano in the solar system', elevation: 21287 },
  { id: '2', name: 'Valles Marineris', lat: -14, lon: -59, type: 'canyon', description: 'Massive canyon system', elevation: -7000 },
  { id: '3', name: 'Gale Crater', lat: -5.4, lon: 137.8, type: 'crater', description: 'Curiosity rover landing site', elevation: -4500 },
  { id: '4', name: 'Jezero Crater', lat: 18.44, lon: 77.45, type: 'crater', description: 'Perseverance rover landing site', elevation: -2500 },
  { id: '5', name: 'Acidalia Planitia', lat: 46.7, lon: -29.8, type: 'plain', description: 'Northern lowlands region', elevation: -4000 },
  { id: '6', name: 'Hellas Planitia', lat: -42.4, lon: 70.5, type: 'basin', description: 'Largest impact crater on Mars', elevation: -8200 },
  { id: '7', name: 'Polar Ice Cap (North)', lat: 85, lon: 0, type: 'ice', description: 'North polar ice cap', elevation: -5000 },
  { id: '8', name: 'Polar Ice Cap (South)', lat: -85, lon: 0, type: 'ice', description: 'South polar ice cap', elevation: -6000 },
  { id: '9', name: 'Tharsis Volcanic Province', lat: 0, lon: -100, type: 'volcano', description: 'Major volcanic region', elevation: 10000 },
  { id: '10', name: 'Chryse Planitia', lat: 20, lon: -50, type: 'plain', description: 'Viking 1 landing site', elevation: -3000 }
];
