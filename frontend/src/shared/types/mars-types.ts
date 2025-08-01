export interface LayerState {
  id: string;
  visible: boolean;
  opacity: number;
}

export interface ViewState {
  centerLat: number;
  centerLon: number;
  zoom: number;
  rotation: number;
}

export interface BookmarkType {
  id: string;
  name: string;
  lat: number;
  lon: number;
  zoom: number;
  description?: string;
  created: Date;
}

export interface MarsLocation {
  id: number;
  name: string;
  lat: number;
  lon: number;
  type: string;
  description: string;
  elevation: number;
}

export type MarsLocationArray = MarsLocation[];
