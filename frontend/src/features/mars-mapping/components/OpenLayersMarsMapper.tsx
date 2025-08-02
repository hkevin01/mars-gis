// Professional OpenLayers Mars Mapper with NASA Data Integration
import {
    Activity,
    Bookmark,
    Download,
    Info,
    Layers,
    MapPin,
    RotateCcw,
    Search,
    Settings,
    ZoomIn,
    ZoomOut,
} from 'lucide-react';
import { Feature, Map, View } from 'ol';
import { Point } from 'ol/geom';
import { defaults as defaultInteractions, DragRotateAndZoom } from 'ol/interaction';
import { Tile as TileLayer, Vector as VectorLayer } from 'ol/layer';
import 'ol/ol.css';
import { Vector as VectorSource, XYZ } from 'ol/source';
import { Circle, Fill, Stroke, Style, Text } from 'ol/style';
import React, { useCallback, useEffect, useRef, useState } from 'react';

// Import shared modules
import { MARS_LOCATIONS } from '../../../shared/constants/mars-data';
import type { BookmarkType, MarsLocation, ViewState } from '../../../shared/types/mars-types';
import { formatCoordinates, getLocationColor, searchMarsLocations } from '../../../shared/utils/mars-utils';

// NASA Mars data endpoints
const NASA_MARS_APIS = {
  // USGS Astrogeology Mars Global Color Mosaic
  imagery: 'https://astrowebmaps.wr.usgs.gov/webmapatlas/Layers/Mars/Mars_Viking_ClrMosaic_global_925m/{z}/{x}/{y}.png',
  // USGS Mars MOLA elevation service
  usgs: 'https://astrowebmaps.wr.usgs.gov/webmapatlas/Layers/Mars/Mars_MGS_MOLA_ClrShade_merge_global_463m/{z}/{x}/{y}.png',
  // Mars Global Surveyor MOLA elevation data
  elevation: 'https://astrowebmaps.wr.usgs.gov/webmapatlas/Layers/Mars/Mars_MGS_MOLA_Shade_global_463m/{z}/{x}/{y}.png',
  // Alternative OpenStreetMap as fallback
  fallback: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
};

// Mars projection configuration
const MARS_PROJECTION = 'EPSG:4326'; // Geographic coordinate system for Mars
const MARS_EXTENT = [-180, -90, 180, 90]; // Mars coordinate bounds

interface LayerConfig {
  id: string;
  name: string;
  description: string;
  url: string;
  visible: boolean;
  opacity: number;
}

const OpenLayersMarsMapper: React.FC = () => {
  const mapRef = useRef<HTMLDivElement>(null);
  const olMapRef = useRef<Map | null>(null);
  const markersLayerRef = useRef<VectorLayer<VectorSource> | null>(null);

  const [viewState, setViewState] = useState<ViewState>({
    centerLat: 0,
    centerLon: 0,
    zoom: 2,
    rotation: 0,
  });

  const [layers, setLayers] = useState<LayerConfig[]>([
    {
      id: 'imagery',
      name: 'Mars Color Mosaic',
      description: 'USGS Mars Viking Color Mosaic - Global View',
      url: NASA_MARS_APIS.imagery,
      visible: true,
      opacity: 1.0
    },
    {
      id: 'usgs',
      name: 'MOLA Shaded Relief',
      description: 'Mars Global Surveyor MOLA Shaded Relief',
      url: NASA_MARS_APIS.usgs,
      visible: false,
      opacity: 0.8
    },
    {
      id: 'elevation',
      name: 'MOLA Elevation',
      description: 'Mars Global Surveyor MOLA elevation data',
      url: NASA_MARS_APIS.elevation,
      visible: false,
      opacity: 0.7
    },
    {
      id: 'fallback',
      name: 'Earth Reference',
      description: 'OpenStreetMap for reference and fallback',
      url: NASA_MARS_APIS.fallback,
      visible: false,
      opacity: 0.6
    }
  ]);

  const [bookmarks, setBookmarks] = useState<BookmarkType[]>([]);
  const [selectedLocation, setSelectedLocation] = useState<MarsLocation | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const [showBookmarks, setShowBookmarks] = useState(false);
  const [showLayerPanel, setShowLayerPanel] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [mapLoading, setMapLoading] = useState(true);

  // Initialize OpenLayers map
  const initializeMap = useCallback(() => {
    if (!mapRef.current) return;

    // Create base layers with error handling
    const baseLayers = layers.map(layerConfig => {
      const source = new XYZ({
        url: layerConfig.url,
        projection: MARS_PROJECTION,
        crossOrigin: 'anonymous',
        maxZoom: 10
      });

      // Add error handling for tile loading
      source.on('tileloaderror', () => {
        // Tile failed to load - this is expected for some Mars tile services
        setMapLoading(false);
      });

      source.on('tileloadend', () => {
        // Tile loaded successfully
        setMapLoading(false);
      });

      return new TileLayer({
        source,
        visible: layerConfig.visible,
        opacity: layerConfig.opacity,
        properties: { id: layerConfig.id }
      });
    });

    // Create markers layer for Mars locations
    const markersSource = new VectorSource();
    const markersLayer = new VectorLayer({
      source: markersSource,
      style: (feature) => {
        const location = feature.get('location') as MarsLocation;
        return new Style({
          image: new Circle({
            radius: 8,
            fill: new Fill({
              color: getLocationColor(location.type)
            }),
            stroke: new Stroke({
              color: '#ffffff',
              width: 2
            })
          }),
          text: new Text({
            text: location.name,
            offsetY: -20,
            font: '12px Arial, sans-serif',
            fill: new Fill({ color: '#ffffff' }),
            stroke: new Stroke({ color: '#000000', width: 3 })
          })
        });
      }
    });

    markersLayerRef.current = markersLayer;

    // Add Mars location features
    MARS_LOCATIONS.forEach(location => {
      const feature = new Feature({
        geometry: new Point([location.lon, location.lat]),
        location: location
      });
      markersSource.addFeature(feature);
    });

    // Create map
    const map = new Map({
      target: mapRef.current,
      layers: [...baseLayers, markersLayer],
      view: new View({
        projection: MARS_PROJECTION,
        center: [viewState.centerLon, viewState.centerLat],
        zoom: viewState.zoom,
        extent: MARS_EXTENT,
        minZoom: 1,
        maxZoom: 12
      }),
      interactions: defaultInteractions().extend([
        new DragRotateAndZoom()
      ])
    });

    // Handle map clicks
    map.on('click', (event) => {
      const coordinate = event.coordinate;
      const [lon, lat] = coordinate;

      // Check if clicked on a location marker
      const features = map.getFeaturesAtPixel(event.pixel);
      if (features.length > 0) {
        const feature = features[0];
        const location = feature.get('location') as MarsLocation;
        if (location) {
          setSelectedLocation(location);
          return;
        }
      }

      // Update view state with clicked coordinates
      setViewState(prev => ({
        ...prev,
        centerLat: lat,
        centerLon: lon
      }));
    });

    // Handle view changes
    map.getView().on('change', () => {
      const view = map.getView();
      const center = view.getCenter();
      if (center) {
        setViewState(prev => ({
          ...prev,
          centerLat: center[1],
          centerLon: center[0],
          zoom: view.getZoom() || prev.zoom,
          rotation: view.getRotation()
        }));
      }
    });

    olMapRef.current = map;
    setMapLoading(false);

    return () => {
      map.setTarget();
    };
  }, [layers, viewState.centerLat, viewState.centerLon, viewState.zoom]);

  // Update layer visibility and opacity
  const updateLayer = useCallback((layerId: string, updates: Partial<LayerConfig>) => {
    setLayers(prev => prev.map(layer =>
      layer.id === layerId ? { ...layer, ...updates } : layer
    ));

    // Update OpenLayers layer
    if (olMapRef.current) {
      const olLayer = olMapRef.current.getLayers().getArray().find(
        layer => layer.get('id') === layerId
      );
      if (olLayer) {
        if (updates.visible !== undefined) {
          olLayer.setVisible(updates.visible);
        }
        if (updates.opacity !== undefined) {
          olLayer.setOpacity(updates.opacity);
        }
      }
    }
  }, []);

  // Map control functions
  const zoomIn = useCallback(() => {
    if (olMapRef.current) {
      const view = olMapRef.current.getView();
      view.animate({ zoom: view.getZoom()! + 1, duration: 300 });
    }
  }, []);

  const zoomOut = useCallback(() => {
    if (olMapRef.current) {
      const view = olMapRef.current.getView();
      view.animate({ zoom: view.getZoom()! - 1, duration: 300 });
    }
  }, []);

  const resetView = useCallback(() => {
    if (olMapRef.current) {
      const view = olMapRef.current.getView();
      view.animate({
        center: [0, 0],
        zoom: 2,
        rotation: 0,
        duration: 500
      });
    }
  }, []);

  const flyToLocation = useCallback((location: MarsLocation) => {
    if (olMapRef.current) {
      const view = olMapRef.current.getView();
      view.animate({
        center: [location.lon, location.lat],
        zoom: 6,
        duration: 1000
      });
      setSelectedLocation(location);
    }
  }, []);

  // Bookmark management
  const addBookmark = useCallback(() => {
    if (selectedLocation) {
      const newBookmark: BookmarkType = {
        id: Date.now().toString(),
        name: selectedLocation.name,
        lat: selectedLocation.lat,
        lon: selectedLocation.lon,
        zoom: viewState.zoom,
        description: `Bookmark for ${selectedLocation.name}`,
        created: new Date(),
      };
      setBookmarks(prev => [...prev, newBookmark]);
    }
  }, [selectedLocation, viewState.zoom]);

  const goToBookmark = useCallback((bookmark: BookmarkType) => {
    if (olMapRef.current) {
      const view = olMapRef.current.getView();
      view.animate({
        center: [bookmark.lon, bookmark.lat],
        zoom: bookmark.zoom,
        duration: 1000
      });
    }
  }, []);

  // Data export
  const exportData = useCallback(() => {
    const data = {
      viewState,
      layers,
      bookmarks,
      selectedLocation,
      exportedAt: new Date().toISOString(),
      projection: MARS_PROJECTION
    };
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `mars-mapping-data-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [viewState, layers, bookmarks, selectedLocation]);

  // Initialize map on mount
  useEffect(() => {
    const cleanup = initializeMap();
    return cleanup;
  }, [initializeMap]);

  // Search functionality
  const searchResults = searchMarsLocations(searchQuery, MARS_LOCATIONS);

  return (
    <div className="w-full h-screen bg-black relative overflow-hidden">
      {/* OpenLayers Map Container */}
      <div
        ref={mapRef}
        className="w-full h-full mars-background"
        style={{
          background: 'linear-gradient(135deg, #8B4513 0%, #CD853F 25%, #A0522D 50%, #654321 75%, #2F1B14 100%)',
        }}
      />

      {/* Loading Overlay */}
      {mapLoading && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-50">
          <div className="text-white text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <div className="text-lg font-semibold">Loading Mars Data...</div>
            <div className="text-sm text-gray-400">Connecting to NASA servers</div>
          </div>
        </div>
      )}

      {/* Map Controls */}
      <div className="absolute top-4 left-4 z-30 space-y-2">
        <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 border border-gray-700 shadow-lg">
          <h3 className="text-white text-sm font-semibold mb-2 flex items-center">
            <Activity className="w-4 h-4 mr-2 text-blue-400" />
            Map Controls
          </h3>
          <div className="space-y-2">
            <button
              onClick={zoomIn}
              className="w-full flex items-center justify-center p-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
            >
              <ZoomIn className="w-4 h-4 mr-2" />
              Zoom In
            </button>
            <button
              onClick={zoomOut}
              className="w-full flex items-center justify-center p-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
            >
              <ZoomOut className="w-4 h-4 mr-2" />
              Zoom Out
            </button>
            <button
              onClick={resetView}
              className="w-full flex items-center justify-center p-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset View
            </button>
          </div>
        </div>
      </div>

      {/* Layer Panel */}
      {showLayerPanel && (
        <div className="absolute left-4 top-1/2 transform -translate-y-1/2 z-30 w-64">
          <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700 shadow-lg">
            <h3 className="text-white text-sm font-semibold mb-3 flex items-center">
              <Layers className="w-4 h-4 mr-2 text-green-400" />
              NASA Mars Data Layers
            </h3>
            {layers.map((layer) => (
              <div key={layer.id} className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-white text-xs flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={layer.visible}
                      onChange={(e) => updateLayer(layer.id, { visible: e.target.checked })}
                      className="mr-2 rounded"
                    />
                    {layer.name}
                  </label>
                </div>
                <div className="text-xs text-gray-400 mb-2">{layer.description}</div>
                {layer.visible && (
                  <div className="ml-4">
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={layer.opacity}
                      onChange={(e) => updateLayer(layer.id, { opacity: parseFloat(e.target.value) })}
                      className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="text-xs text-gray-400 mt-1">
                      Opacity: {Math.round(layer.opacity * 100)}%
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="absolute top-4 right-4 z-30 space-y-2">
        <button
          onClick={() => setShowSearch(!showSearch)}
          className="p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors shadow-lg"
        >
          <Search className="w-5 h-5" />
        </button>
        <button
          onClick={() => setShowBookmarks(!showBookmarks)}
          className="p-3 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors shadow-lg"
        >
          <Bookmark className="w-5 h-5" />
        </button>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors shadow-lg"
        >
          <Settings className="w-5 h-5" />
        </button>
        <button
          onClick={exportData}
          className="p-3 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors shadow-lg"
        >
          <Download className="w-5 h-5" />
        </button>
      </div>

      {/* Search Panel */}
      {showSearch && (
        <div className="absolute top-16 right-4 z-30 w-80">
          <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700 shadow-lg">
            <h3 className="text-white text-sm font-semibold mb-3 flex items-center">
              <Search className="w-4 h-4 mr-2 text-blue-400" />
              Search Mars Locations
            </h3>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search famous Mars locations..."
              className="w-full p-2 bg-gray-800 text-white border border-gray-600 rounded focus:border-blue-500 focus:outline-none"
            />
            {searchResults.length > 0 && (
              <div className="mt-3 max-h-60 overflow-y-auto">
                {searchResults.map((location) => (
                  <button
                    key={location.id}
                    onClick={() => flyToLocation(location)}
                    className="w-full text-left p-3 text-white hover:bg-gray-700 cursor-pointer rounded transition-colors"
                  >
                    <div className="font-medium">{location.name}</div>
                    <div className="text-xs text-gray-400">
                      {formatCoordinates(location.lat, location.lon)} • {location.type}
                    </div>
                    <div className="text-xs text-blue-400 mt-1">
                      Click to fly to location
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Bookmarks Panel */}
      {showBookmarks && (
        <div className="absolute top-16 right-4 z-30 w-80">
          <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700 shadow-lg">
            <h3 className="text-white text-sm font-semibold mb-3 flex items-center">
              <Bookmark className="w-4 h-4 mr-2 text-green-400" />
              Saved Locations
            </h3>
            {selectedLocation && (
              <button
                onClick={addBookmark}
                className="w-full p-2 bg-green-600 hover:bg-green-700 text-white rounded mb-3 transition-colors"
              >
                Bookmark Current Location
              </button>
            )}
            <div className="max-h-60 overflow-y-auto">
              {bookmarks.map((bookmark) => (
                <button
                  key={bookmark.id}
                  onClick={() => goToBookmark(bookmark)}
                  className="w-full text-left p-3 text-white hover:bg-gray-700 cursor-pointer rounded transition-colors mb-2"
                >
                  <div className="font-medium">{bookmark.name}</div>
                  <div className="text-xs text-gray-400">
                    {formatCoordinates(bookmark.lat, bookmark.lon)}
                  </div>
                  <div className="text-xs text-blue-400">
                    Zoom: {bookmark.zoom.toFixed(1)}x
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Settings Panel */}
      {showSettings && (
        <div className="absolute top-16 right-4 z-30 w-80">
          <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700 shadow-lg">
            <h3 className="text-white text-sm font-semibold mb-3 flex items-center">
              <Settings className="w-4 h-4 mr-2 text-purple-400" />
              Map Settings
            </h3>
            <div className="space-y-3">
              <div>
                <label className="text-white text-xs flex items-center">
                  <input
                    type="checkbox"
                    checked={showLayerPanel}
                    onChange={(e) => setShowLayerPanel(e.target.checked)}
                    className="mr-2 rounded"
                  />
                  {' '}Show Layer Panel
                </label>
              </div>
              <div className="text-xs text-gray-400">
                <div className="mb-1">Data Sources:</div>
                <div>• NASA Mars Trek</div>
                <div>• USGS Astrogeology</div>
                <div>• Mars Global Surveyor MOLA</div>
                <div>• Thermal Emission Imaging System</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Location Info Panel */}
      {selectedLocation && (
        <div className="absolute bottom-4 left-4 z-30 w-80">
          <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700 shadow-lg">
            <h3 className="text-white text-lg font-semibold mb-2 flex items-center">
              <MapPin className="w-5 h-5 mr-2 text-red-400" />
              {selectedLocation.name}
            </h3>
            <div className="text-gray-300 space-y-1">
              <div className="flex items-center">
                <Info className="w-4 h-4 mr-2 text-blue-400" />
                Type: {selectedLocation.type}
              </div>
              <div>Coordinates: {formatCoordinates(selectedLocation.lat, selectedLocation.lon)}</div>
              <div>Elevation: {selectedLocation.elevation}m</div>
              {selectedLocation.description && (
                <div className="mt-2 text-sm bg-gray-800 p-2 rounded">
                  {selectedLocation.description}
                </div>
              )}
            </div>
            <button
              onClick={() => flyToLocation(selectedLocation)}
              className="mt-3 w-full p-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
            >
              Center on Location
            </button>
          </div>
        </div>
      )}

      {/* Status Bar */}
      <div className="absolute bottom-4 right-4 z-30">
        <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 border border-gray-700 shadow-lg">
          <div className="text-white text-sm flex items-center space-x-4">
            <div className="flex items-center">
              <Activity className="w-4 h-4 mr-1 text-green-400" />
              Zoom: {viewState.zoom.toFixed(1)}x
            </div>
            <div>
              Center: {formatCoordinates(viewState.centerLat, viewState.centerLon)}
            </div>
            <div className="text-green-400 text-xs">
              NASA Live
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OpenLayersMarsMapper;
