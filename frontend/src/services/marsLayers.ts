import { Feature } from 'ol';
import { Point } from 'ol/geom';
import TileLayer from 'ol/layer/Tile';
import VectorLayer from 'ol/layer/Vector';
import { fromLonLat } from 'ol/proj';
import VectorSource from 'ol/source/Vector';
import XYZ from 'ol/source/XYZ';

// Properly exported interface
export interface MarsLayer extends TileLayer<XYZ> {
  type: string;
}

// Exported constant
export const MARS_LANDING_SITES = [
  { name: 'Perseverance', lon: 77.45, lat: 18.45 },
  { name: 'Curiosity', lon: 137.44, lat: -4.59 },
  { name: 'Opportunity', lon: -5.53, lat: -1.95 },
  { name: 'Spirit', lon: 175.47, lat: -14.57 },
];

// Exported layer creation functions
export function createMarsBaseLayer(): MarsLayer {
  const layer = new TileLayer({
    source: new XYZ({
      url: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg',
      maxZoom: 8,
      minZoom: 0,
      crossOrigin: 'anonymous'
    }),
  }) as MarsLayer;

  layer.type = 'base';
  return layer;
}

export function createMarsElevationLayer(): MarsLayer {
  const layer = new TileLayer({
    source: new XYZ({
      url: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_DEM_mosaic_global_463m/1.0.0/default/default028mm/{z}/{y}/{x}.png',
      maxZoom: 8,
      minZoom: 0,
      crossOrigin: 'anonymous'
    }),
    opacity: 0.7,
    visible: false,
  }) as MarsLayer;

  layer.type = 'elevation';
  return layer;
}

export function createMarsInfraredLayer(): MarsLayer {
  const layer = new TileLayer({
    source: new XYZ({
      url: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_TES_Thermal_Inertia_mosaic_global_3cpd/1.0.0/default/default028mm/{z}/{y}/{x}.png',
      maxZoom: 8,
      minZoom: 0,
      crossOrigin: 'anonymous'
    }),
    opacity: 0.7,
    visible: false,
  }) as MarsLayer;

  layer.type = 'infrared';
  return layer;
}

export function createLandingSitesLayer(): VectorLayer<VectorSource> {
  const features = MARS_LANDING_SITES.map(site => {
    return new Feature({
      geometry: new Point(fromLonLat([site.lon, site.lat])),
      name: site.name,
    });
  });

  return new VectorLayer({
    source: new VectorSource({
      features,
    }),
    visible: false,
  });
}
