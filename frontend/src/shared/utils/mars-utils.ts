import type { MarsLocation } from '../types/mars-types';

export const getLocationColor = (type: string): string => {
  switch (type) {
    case 'volcano': return '#FF4500';
    case 'canyon': return '#8B4513';
    case 'crater': return '#DAA520';
    case 'plain': return '#CD853F';
    case 'basin': return '#4169E1';
    case 'ice': return '#87CEEB';
    default: return '#FFA500';
  }
};

export const getLocationIcon = (type: string) => {
  // This returns the icon name for dynamic import
  switch (type) {
    case 'volcano': return 'Mountain';
    case 'canyon': return 'Activity';
    case 'crater': return 'Target';
    case 'plain': return 'MapIcon';
    case 'basin': return 'Target';
    case 'ice': return 'Thermometer';
    default: return 'Navigation';
  }
};

export const getLayerIcon = (type: string) => {
  switch (type) {
    case 'elevation': return 'Mountain';
    case 'imagery': return 'Camera';
    case 'thermal': return 'Thermometer';
    case 'geology': return 'Activity';
    case 'atmosphere': return 'Globe';
    default: return 'MapIcon';
  }
};

export const searchMarsLocations = (query: string, locations: MarsLocation[]): MarsLocation[] => {
  if (!query.trim()) {
    return [];
  }

  return locations.filter(location =>
    location.name.toLowerCase().includes(query.toLowerCase()) ||
    location.description.toLowerCase().includes(query.toLowerCase()) ||
    location.type.toLowerCase().includes(query.toLowerCase())
  );
};

export const formatCoordinates = (lat: number, lon: number): string => {
  return `${lat.toFixed(4)}°, ${lon.toFixed(4)}°`;
};

export const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
  // Simple distance calculation for Mars surface
  return Math.sqrt(Math.pow(lat2 - lat1, 2) + Math.pow(lon2 - lon1, 2));
};
