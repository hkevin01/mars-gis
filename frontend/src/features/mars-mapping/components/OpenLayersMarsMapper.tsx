// Professional OpenLayers Mars Mapper with NASA Data Integration
import {
    Activity,
    Bookmark,
    ChevronDown,
    ChevronUp,
    Compass,
    Download,
    Eye,
    EyeOff,
    Home,
    Info,
    Layers,
    MapPin,
    Maximize,
    Minimize,
    MousePointer,
    RotateCcw,
    Ruler,
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
import { Projection } from 'ol/proj';
import { register } from 'ol/proj/proj4';
import { Vector as VectorSource, XYZ } from 'ol/source';
import { Circle, Fill, Stroke, Style, Text } from 'ol/style';
import proj4 from 'proj4';
import React, { useCallback, useEffect, useRef, useState } from 'react';

// Import shared modules
import { MARS_LOCATIONS } from '../../../shared/constants/mars-data';
import type { BookmarkType, MarsLocation, ViewState } from '../../../shared/types/mars-types';
import { formatCoordinates, getLocationColor, searchMarsLocations } from '../../../shared/utils/mars-utils';

// Register Mars coordinate system
proj4.defs('IAU2000:49900', '+proj=longlat +a=3396190 +b=3376200 +no_defs');
register(proj4);

// High-Definition NASA Mars data endpoints using correct Trek API format
const NASA_MARS_HD_APIS = {
  // Viking Color Mosaic - Primary base layer (working endpoint)
  vikingColor: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg',
  // MOLA Color Hillshade - Global Mars topography with colors (working endpoint)
  molaColorShade: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_ClrShade_merge_global_463m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg',
  // THEMIS Day IR - Corrected endpoint
  themisDay: 'https://trek.nasa.gov/tiles/Mars/EQ/THEMIS_DayIR_ControlledMosaics_100m_v2_oct2018/1.0.0/default/default028mm/{z}/{y}/{x}.png',
  // THEMIS Night IR - Corrected endpoint
  themisNight: 'https://trek.nasa.gov/tiles/Mars/EQ/THEMIS_NightIR_ControlledMosaics_100m_v2_oct2018/1.0.0/default/default028mm/{z}/{y}/{x}.png',
  // CTX Global Uncontrolled - High resolution context camera
  ctxGlobal: 'https://trek.nasa.gov/tiles/Mars/EQ/CTX_beta01_uncontrolled_5m_Caltech/1.0.0/default/default028mm/{z}/{y}/{x}.png',
  // HiRISE Global - Ultra high resolution where available
  hirise: 'https://trek.nasa.gov/tiles/Mars/EQ/HiRISE_Global/1.0.0/default/default028mm/{z}/{y}/{x}.png',
  // Fallback - USGS Astrogeology (different coordinate pattern)
  fallback: 'https://astrowebmaps.wr.usgs.gov/webmapatlas/Layers/Mars/Mars_Viking_ClrMosaic_global_925m/{z}/{x}/{y}.png'
};

// Mars projection configuration with proper Mars coordinate system
const MARS_PROJECTION = new Projection({
  code: 'IAU2000:49900',
  extent: [-180, -90, 180, 90],
  worldExtent: [-180, -90, 180, 90],
  units: 'degrees'
});

const MARS_EXTENT = [-180, -90, 180, 90]; // Mars coordinate bounds

interface LayerConfig {
  id: string;
  name: string;
  description: string;
  url: string;
  visible: boolean;
  opacity: number;
  minZoom?: number;
  maxZoom?: number;
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
      id: 'vikingColor',
      name: 'Viking Color Mosaic',
      description: 'NASA Mars Viking Color Global Mosaic - Primary base layer',
      url: NASA_MARS_HD_APIS.vikingColor,
      visible: true,
      opacity: 1.0,
      maxZoom: 12
    },
    {
      id: 'molaColorShade',
      name: 'MOLA Color Hillshade',
      description: 'Mars Global Surveyor MOLA Color Hillshade',
      url: NASA_MARS_HD_APIS.molaColorShade,
      visible: false,
      opacity: 0.9,
      maxZoom: 10
    },
    {
      id: 'themisDay',
      name: 'THEMIS Day IR',
      description: 'THEMIS Daytime Infrared Global Mosaic',
      url: NASA_MARS_HD_APIS.themisDay,
      visible: false,
      opacity: 0.8,
      maxZoom: 14
    },
    {
      id: 'themisNight',
      name: 'THEMIS Night IR',
      description: 'THEMIS Nighttime Infrared Global Mosaic',
      url: NASA_MARS_HD_APIS.themisNight,
      visible: false,
      opacity: 0.7,
      maxZoom: 14
    },
    {
      id: 'ctxGlobal',
      name: 'CTX Global HD',
      description: 'MRO Context Camera Global Uncontrolled Mosaic',
      url: NASA_MARS_HD_APIS.ctxGlobal,
      visible: false,
      opacity: 0.8,
      minZoom: 8,
      maxZoom: 18
    },
    {
      id: 'hirise',
      name: 'HiRISE Ultra HD',
      description: 'HiRISE Ultra High Resolution Global Coverage',
      url: NASA_MARS_HD_APIS.hirise,
      visible: false,
      opacity: 0.9,
      minZoom: 15,
      maxZoom: 20
    },
    {
      id: 'fallback',
      name: 'Fallback Layer',
      description: 'USGS Astrogeology Fallback',
      url: NASA_MARS_HD_APIS.fallback,
      visible: false,
      opacity: 0.8
    }
  ]);

  const [bookmarks, setBookmarks] = useState<BookmarkType[]>([]);
  const [selectedLocation, setSelectedLocation] = useState<MarsLocation | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const [showBookmarks, setShowBookmarks] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [mapLoading, setMapLoading] = useState(true);
  
  // New state for enhanced OpenLayers features
  const [isLayerPanelExpanded, setIsLayerPanelExpanded] = useState(false);
  const [showMeasurementTool, setShowMeasurementTool] = useState(false);
  const [showCoordinateInfo, setShowCoordinateInfo] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [mouseCoordinates, setMouseCoordinates] = useState<{lat: number, lon: number} | null>(null);

  // Initialize OpenLayers map with HD Mars terrain
  const initializeMap = useCallback(() => {
    if (!mapRef.current) return;

    // Create enhanced base layers with proper zoom levels and Mars projection
    const baseLayers = layers.map(layerConfig => {
      const source = new XYZ({
        url: layerConfig.url,
        projection: MARS_PROJECTION,
        crossOrigin: 'anonymous',
        maxZoom: layerConfig.maxZoom || 18,
        minZoom: layerConfig.minZoom || 0
      });

      // Enhanced error handling for tile loading
      source.on('tileloadstart', () => {
        setMapLoading(true);
      });

      source.on('tileloaderror', () => {
        setMapLoading(false);
      });

      source.on('tileloadend', () => {
        setMapLoading(false);
      });

      return new TileLayer({
        source,
        visible: layerConfig.visible,
        opacity: layerConfig.opacity,
        minZoom: layerConfig.minZoom,
        maxZoom: layerConfig.maxZoom,
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

    // Create enhanced Mars map with proper projection and HD zoom levels
    const map = new Map({
      target: mapRef.current,
      layers: [...baseLayers, markersLayer],
      view: new View({
        projection: MARS_PROJECTION,
        center: [viewState.centerLon, viewState.centerLat],
        zoom: viewState.zoom,
        extent: MARS_EXTENT,
        minZoom: 1,
        maxZoom: 20,  // Increased for HD viewing
        constrainResolution: true,
        smoothResolutionConstraint: true,
        enableRotation: true
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

    // Handle mouse movement for coordinate tracking
    map.on('pointermove', (event) => {
      const coordinate = event.coordinate;
      const [lon, lat] = coordinate;
      setMouseCoordinates({ lat, lon });
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

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  const goHome = useCallback(() => {
    resetView();
  }, [resetView]);

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

      {/* Compact Map Controls */}
      <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-30">
        <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-2 border border-gray-700 shadow-lg">
          <div className="flex items-center space-x-1">
            <button
              onClick={zoomIn}
              className="flex items-center justify-center p-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="w-3 h-3" />
            </button>
            <button
              onClick={zoomOut}
              className="flex items-center justify-center p-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="w-3 h-3" />
            </button>
            <button
              onClick={goHome}
              className="flex items-center justify-center p-1.5 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
              title="Home View"
            >
              <Home className="w-3 h-3" />
            </button>
            <button
              onClick={resetView}
              className="flex items-center justify-center p-1.5 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
              title="Reset View"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
            <button
              onClick={toggleFullscreen}
              className="flex items-center justify-center p-1.5 bg-purple-600 hover:bg-purple-700 text-white rounded transition-colors"
              title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
            >
              {isFullscreen ? <Minimize className="w-3 h-3" /> : <Maximize className="w-3 h-3" />}
            </button>
            <button
              onClick={() => setShowMeasurementTool(!showMeasurementTool)}
              className={`flex items-center justify-center p-1.5 ${showMeasurementTool ? 'bg-orange-600' : 'bg-gray-600'} hover:bg-orange-700 text-white rounded transition-colors`}
              title="Measurement Tool"
            >
              <Ruler className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Expandable Layer Panel */}
      <div className="absolute right-4 top-20 z-30">
        <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg border border-gray-700 shadow-lg">
          {/* Layer Panel Header */}
          <button 
            className="flex items-center justify-between p-3 w-full text-left hover:bg-gray-800/50 rounded-t-lg transition-colors"
            onClick={() => setIsLayerPanelExpanded(!isLayerPanelExpanded)}
          >
            <div className="flex items-center">
              <Layers className="w-4 h-4 mr-2 text-green-400" />
              <span className="text-white text-sm font-semibold">NASA Mars Data Layers</span>
            </div>
            {isLayerPanelExpanded ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </button>
          
          {/* Expandable Layer Content */}
          {isLayerPanelExpanded && (
            <div className="p-4 pt-0 max-h-96 overflow-y-auto w-80">
              {layers.map((layer) => (
                <div key={layer.id} className="mb-4 border-b border-gray-700 pb-3 last:border-b-0">
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-white text-xs flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={layer.visible}
                        onChange={(e) => updateLayer(layer.id, { visible: e.target.checked })}
                        className="mr-2 rounded"
                      />
                      <span className="font-medium">{layer.name}</span>
                    </label>
                    <button
                      onClick={() => updateLayer(layer.id, { visible: !layer.visible })}
                      className="p-1 hover:bg-gray-700 rounded transition-colors"
                      title={layer.visible ? "Hide Layer" : "Show Layer"}
                    >
                      {layer.visible ? (
                        <Eye className="w-3 h-3 text-green-400" />
                      ) : (
                        <EyeOff className="w-3 h-3 text-gray-400" />
                      )}
                    </button>
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
          )}
        </div>
      </div>

      {/* Enhanced Action Buttons */}
      <div className="absolute top-4 right-4 z-30 space-y-2">
        <button
          onClick={() => setShowSearch(!showSearch)}
          className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors shadow-lg"
          title="Search Mars Locations"
        >
          <Search className="w-4 h-4" />
        </button>
        <button
          onClick={() => setShowBookmarks(!showBookmarks)}
          className="p-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors shadow-lg"
          title="Bookmarks"
        >
          <Bookmark className="w-4 h-4" />
        </button>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors shadow-lg"
          title="Settings"
        >
          <Settings className="w-4 h-4" />
        </button>
        <button
          onClick={() => setShowCoordinateInfo(!showCoordinateInfo)}
          className={`p-2 ${showCoordinateInfo ? 'bg-orange-600' : 'bg-gray-600'} hover:bg-orange-700 text-white rounded-lg transition-colors shadow-lg`}
          title="Coordinate Info"
        >
          <MousePointer className="w-4 h-4" />
        </button>
        <button
          onClick={exportData}
          className="p-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg transition-colors shadow-lg"
          title="Export Data"
        >
          <Download className="w-4 h-4" />
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

      {/* Enhanced Settings Panel */}
      {showSettings && (
        <div className="absolute top-16 right-4 z-30 w-80">
          <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700 shadow-lg">
            <h3 className="text-white text-sm font-semibold mb-3 flex items-center">
              <Settings className="w-4 h-4 mr-2 text-purple-400" />
              Map Settings & Controls
            </h3>
            <div className="space-y-3">
              <div>
                <label className="text-white text-xs flex items-center">
                  <input
                    type="checkbox"
                    checked={isLayerPanelExpanded}
                    onChange={(e) => setIsLayerPanelExpanded(e.target.checked)}
                    className="mr-2 rounded"
                  />
                  {' '}Expand Layer Panel
                </label>
              </div>
              <div>
                <label className="text-white text-xs flex items-center">
                  <input
                    type="checkbox"
                    checked={showCoordinateInfo}
                    onChange={(e) => setShowCoordinateInfo(e.target.checked)}
                    className="mr-2 rounded"
                  />
                  {' '}Show Mouse Coordinates
                </label>
              </div>
              <div>
                <label className="text-white text-xs flex items-center">
                  <input
                    type="checkbox"
                    checked={showMeasurementTool}
                    onChange={(e) => setShowMeasurementTool(e.target.checked)}
                    className="mr-2 rounded"
                  />
                  {' '}Measurement Tool
                </label>
              </div>
              <div className="border-t border-gray-600 pt-3">
                <div className="text-xs text-gray-400">
                  <div className="mb-2 font-medium text-white">Controls:</div>
                  <div>• Left Click: Navigate</div>
                  <div>• Scroll: Zoom in/out</div>
                  <div>• Shift+Drag: Rotate map</div>
                  <div>• Right Click: Context menu</div>
                </div>
              </div>
              <div className="border-t border-gray-600 pt-3">
                <div className="text-xs text-gray-400">
                  <div className="mb-1 font-medium text-white">Data Sources:</div>
                  <div>• NASA Mars Trek</div>
                  <div>• USGS Astrogeology</div>
                  <div>• Mars Global Surveyor MOLA</div>
                  <div>• Thermal Emission Imaging System</div>
                  <div>• Mars Reconnaissance Orbiter</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Location Info Panel */}
      {selectedLocation && (
        <div className="absolute bottom-4 right-4 z-30 w-80">
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

      {/* Status Bar with Enhanced Info */}
      <div className="absolute bottom-4 left-4 z-30">
        <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 border border-gray-700 shadow-lg">
          <div className="text-white text-sm space-y-2">
            <div className="flex items-center space-x-4">
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
            {showCoordinateInfo && mouseCoordinates && (
              <div className="text-xs text-blue-400 border-t border-gray-600 pt-2">
                <div className="flex items-center">
                  <MousePointer className="w-3 h-3 mr-1" />
                  Mouse: {formatCoordinates(mouseCoordinates.lat, mouseCoordinates.lon)}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Measurement Tool Panel */}
      {showMeasurementTool && (
        <div className="absolute bottom-4 right-4 z-30 w-64">
          <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700 shadow-lg">
            <h3 className="text-white text-sm font-semibold mb-3 flex items-center">
              <Ruler className="w-4 h-4 mr-2 text-orange-400" />
              Measurement Tool
            </h3>
            <div className="text-gray-300 text-sm space-y-2">
              <div>Click on the map to start measuring distances</div>
              <div className="text-xs text-gray-400">
                • Single click: Start/continue measuring
                • Double click: Finish measurement
                • ESC: Cancel measurement
              </div>
              <button
                onClick={() => setShowMeasurementTool(false)}
                className="w-full mt-3 p-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
              >
                Close Tool
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Navigation Compass */}
      <div className="absolute top-20 left-4 z-30">
        <div className="bg-gray-900/90 backdrop-blur-sm rounded-full p-3 border border-gray-700 shadow-lg">
          <div className="relative">
            <Compass 
              className="w-8 h-8 text-blue-400" 
              style={{ transform: `rotate(${-viewState.rotation * 180 / Math.PI}deg)` }}
            />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-1 h-1 bg-red-500 rounded-full"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OpenLayersMarsMapper;
