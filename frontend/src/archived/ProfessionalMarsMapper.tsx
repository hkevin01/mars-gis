import { AnimatePresence, motion } from 'framer-motion';
import {
    Activity,
    Bookmark,
    Camera,
    Eye,
    EyeOff,
    FileText,
    Globe,
    Home,
    Info,
    Layers,
    Map as MapIcon,
    Maximize2,
    Minimize2,
    Mountain,
    MousePointer,
    Navigation,
    Plus,
    Ruler,
    Search,
    Target,
    Thermometer,
    X,
    ZoomIn,
    ZoomOut
} from 'lucide-react';
import { Feature, Map, View } from 'ol';
import { Point } from 'ol/geom';
import TileLayer from 'ol/layer/Tile';
import VectorLayer from 'ol/layer/Vector';
import { transform } from 'ol/proj';
import VectorSource from 'ol/source/Vector';
import XYZ from 'ol/source/XYZ';
import { Circle, Fill, Stroke, Style } from 'ol/style';
import React, { useCallback, useEffect, useRef, useState } from 'react';

// Mars coordinate systems and constants
const MARS_RADIUS = 3396190; // meters
const MARS_PROJ = 'EPSG:4326'; // Using WGS84 for Mars (common practice)

// Mars tile layer configurations
const MARS_LAYERS = {
  mola_elevation: {
    name: 'MOLA Elevation',
    description: 'Mars Orbiter Laser Altimeter elevation data',
    url: 'https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/mars/mars_simp_cyl.map&LAYERS=MOLA_elevation&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&STYLES=&FORMAT=image/png&SRS=EPSG:4326&BBOX={bbox-epsg-4326}&WIDTH=256&HEIGHT=256',
    type: 'elevation',
    maxZoom: 10,
    attribution: '© USGS/NASA MOLA'
  },
  ctx_mosaic: {
    name: 'CTX Global Mosaic',
    description: 'Mars Reconnaissance Orbiter Context Camera global mosaic',
    url: 'https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/mars/mars_simp_cyl.map&LAYERS=CTX_mosaic_global&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&STYLES=&FORMAT=image/png&SRS=EPSG:4326&BBOX={bbox-epsg-4326}&WIDTH=256&HEIGHT=256',
    type: 'imagery',
    maxZoom: 12,
    attribution: '© NASA/JPL/MSSS'
  },
  themis_ir: {
    name: 'THEMIS Thermal IR',
    description: 'Thermal Emission Imaging System infrared data',
    url: 'https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/mars/mars_simp_cyl.map&LAYERS=THEMIS_IR_day&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&STYLES=&FORMAT=image/png&SRS=EPSG:4326&BBOX={bbox-epsg-4326}&WIDTH=256&HEIGHT=256',
    type: 'thermal',
    maxZoom: 8,
    attribution: '© NASA/JPL/ASU'
  },
  geological_map: {
    name: 'Geological Map',
    description: 'USGS geological mapping of Mars',
    url: 'https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/mars/mars_simp_cyl.map&LAYERS=geology&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&STYLES=&FORMAT=image/png&SRS=EPSG:4326&BBOX={bbox-epsg-4326}&WIDTH=256&HEIGHT=256',
    type: 'geology',
    maxZoom: 6,
    attribution: '© USGS Astrogeology'
  },
  viking_mdim: {
    name: 'Viking MDIM 2.1',
    description: 'Viking Mars Digital Image Mosaic',
    url: 'https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/mars/mars_simp_cyl.map&LAYERS=MDIM21&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&STYLES=&FORMAT=image/png&SRS=EPSG:4326&BBOX={bbox-epsg-4326}&WIDTH=256&HEIGHT=256',
    type: 'historical',
    maxZoom: 7,
    attribution: '© NASA/JPL/USGS'
  }
};

// Famous Mars locations
const MARS_LOCATIONS = [
  { name: 'Olympus Mons', lat: 18.65, lon: -133.8, type: 'volcano', description: 'Largest volcano in the solar system' },
  { name: 'Valles Marineris', lat: -14, lon: -59, type: 'canyon', description: 'Massive canyon system' },
  { name: 'Gale Crater', lat: -5.4, lon: 137.8, type: 'crater', description: 'Curiosity rover landing site' },
  { name: 'Jezero Crater', lat: 18.44, lon: 77.45, type: 'crater', description: 'Perseverance rover landing site' },
  { name: 'Acidalia Planitia', lat: 46.7, lon: -29.8, type: 'plain', description: 'Northern lowlands region' },
  { name: 'Hellas Planitia', lat: -42.4, lon: 70.5, type: 'basin', description: 'Largest impact crater on Mars' },
  { name: 'Polar Ice Cap (North)', lat: 85, lon: 0, type: 'ice', description: 'North polar ice cap' },
  { name: 'Polar Ice Cap (South)', lat: -85, lon: 0, type: 'ice', description: 'South polar ice cap' }
];

interface LayerState {
  id: string;
  visible: boolean;
  opacity: number;
  blend?: string;
}

interface Bookmark {
  id: string;
  name: string;
  lat: number;
  lon: number;
  zoom: number;
  description?: string;
  created: Date;
}

interface MeasurementState {
  active: boolean;
  type: 'distance' | 'area' | 'elevation';
  coordinates: number[][];
  result?: string;
}

const ProfessionalMarsMapper: React.FC = () => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<Map | null>(null);
  const [map, setMap] = useState<Map | null>(null);
  const [currentView, setCurrentView] = useState({ lat: 0, lon: 0, zoom: 2 });
  const [activeLayers, setActiveLayers] = useState<LayerState[]>([
    { id: 'ctx_mosaic', visible: true, opacity: 1.0 }
  ]);
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [measurement, setMeasurement] = useState<MeasurementState>({ active: false, type: 'distance', coordinates: [] });
  const [coordinateDisplay, setCoordinateDisplay] = useState({ lat: 0, lon: 0 });
  const [selectedTool, setSelectedTool] = useState<string>('pan');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [layerPanelOpen, setLayerPanelOpen] = useState(false);
  const [bookmarkPanelOpen, setBookmarkPanelOpen] = useState(false);
  const [searchPanelOpen, setSearchPanelOpen] = useState(false);
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d');
  const [fullscreen, setFullscreen] = useState(false);

  // Initialize map
  useEffect(() => {
    if (mapRef.current && !mapInstanceRef.current) {
      const initialMap = new Map({
        target: mapRef.current,
        layers: [],
        view: new View({
          center: [0, 0],
          zoom: 2,
          projection: MARS_PROJ,
          maxZoom: 15,
          minZoom: 1
        }),
        controls: []
      });

      // Add initial layer
      const ctxLayer = new TileLayer({
        source: new XYZ({
          url: MARS_LAYERS.ctx_mosaic.url,
          projection: MARS_PROJ,
          maxZoom: MARS_LAYERS.ctx_mosaic.maxZoom
        }),
        opacity: 1.0
      });
      initialMap.addLayer(ctxLayer);

      // Add vector layer for markers
      const vectorSource = new VectorSource();
      const vectorLayer = new VectorLayer({
        source: vectorSource,
        style: new Style({
          image: new Circle({
            radius: 8,
            fill: new Fill({ color: '#ff6b35' }),
            stroke: new Stroke({ color: '#fff', width: 2 })
          })
        })
      });
      initialMap.addLayer(vectorLayer);

      // Add location markers
      MARS_LOCATIONS.forEach(location => {
        const feature = new Feature({
          geometry: new Point(transform([location.lon, location.lat], MARS_PROJ, MARS_PROJ)),
          name: location.name,
          type: location.type,
          description: location.description
        });
        vectorSource.addFeature(feature);
      });

      // Mouse move handler for coordinates
      initialMap.on('pointermove', (evt) => {
        const coordinate = transform(evt.coordinate, MARS_PROJ, MARS_PROJ);
        setCoordinateDisplay({ lat: coordinate[1], lon: coordinate[0] });
      });

      // View change handler
      initialMap.getView().on('change', () => {
        const view = initialMap.getView();
        const center = transform(view.getCenter()!, MARS_PROJ, MARS_PROJ);
        setCurrentView({
          lat: center[1],
          lon: center[0],
          zoom: view.getZoom() || 2
        });
      });

      mapInstanceRef.current = initialMap;
      setMap(initialMap);
    }
  }, []);

  // Layer management
  const toggleLayer = useCallback((layerId: string) => {
    if (!map) return;

    setActiveLayers(prev => {
      const existing = prev.find(l => l.id === layerId);
      if (existing) {
        // Toggle visibility
        const updated = prev.map(l =>
          l.id === layerId ? { ...l, visible: !l.visible } : l
        );

        // Update map layer
        const layer = map.getLayers().getArray().find(l => l.get('id') === layerId);
        if (layer) {
          layer.setVisible(!existing.visible);
        }

        return updated;
      } else {
        // Add new layer
        const layerConfig = MARS_LAYERS[layerId as keyof typeof MARS_LAYERS];
        if (layerConfig) {
          const newLayer = new TileLayer({
            source: new XYZ({
              url: layerConfig.url,
              projection: MARS_PROJ,
              maxZoom: layerConfig.maxZoom
            }),
            opacity: 0.8
          });
          newLayer.set('id', layerId);
          map.addLayer(newLayer);

          return [...prev, { id: layerId, visible: true, opacity: 0.8 }];
        }
      }
      return prev;
    });
  }, [map]);

  const setLayerOpacity = useCallback((layerId: string, opacity: number) => {
    if (!map) return;

    setActiveLayers(prev => prev.map(l =>
      l.id === layerId ? { ...l, opacity } : l
    ));

    const layer = map.getLayers().getArray().find(l => l.get('id') === layerId);
    if (layer) {
      layer.setOpacity(opacity);
    }
  }, [map]);

  // Navigation functions
  const flyToLocation = useCallback((lat: number, lon: number, zoom: number = 8) => {
    if (!map) return;

    const view = map.getView();
    const center = transform([lon, lat], MARS_PROJ, MARS_PROJ);

    view.animate({
      center,
      zoom,
      duration: 1000
    });
  }, [map]);

  const zoomIn = useCallback(() => {
    if (!map) return;
    const view = map.getView();
    view.animate({ zoom: view.getZoom()! + 1, duration: 250 });
  }, [map]);

  const zoomOut = useCallback(() => {
    if (!map) return;
    const view = map.getView();
    view.animate({ zoom: view.getZoom()! - 1, duration: 250 });
  }, [map]);

  const resetView = useCallback(() => {
    flyToLocation(0, 0, 2);
  }, [flyToLocation]);

  // Search functionality
  const searchLocations = useCallback((query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    const filtered = MARS_LOCATIONS.filter(location =>
      location.name.toLowerCase().includes(query.toLowerCase()) ||
      location.description.toLowerCase().includes(query.toLowerCase()) ||
      location.type.toLowerCase().includes(query.toLowerCase())
    );

    setSearchResults(filtered);
  }, []);

  useEffect(() => {
    const debounceTimer = setTimeout(() => {
      searchLocations(searchQuery);
    }, 300);

    return () => clearTimeout(debounceTimer);
  }, [searchQuery, searchLocations]);

  // Bookmark management
  const addBookmark = useCallback((name: string, description?: string) => {
    const newBookmark: Bookmark = {
      id: Date.now().toString(),
      name,
      lat: currentView.lat,
      lon: currentView.lon,
      zoom: currentView.zoom,
      description,
      created: new Date()
    };

    setBookmarks(prev => [...prev, newBookmark]);
  }, [currentView]);

  const removeBookmark = useCallback((id: string) => {
    setBookmarks(prev => prev.filter(b => b.id !== id));
  }, []);

  // Tool selection
  const selectTool = useCallback((tool: string) => {
    setSelectedTool(tool);

    if (tool === 'measure') {
      setMeasurement({ active: true, type: 'distance', coordinates: [] });
    } else {
      setMeasurement(prev => ({ ...prev, active: false }));
    }
  }, []);

  // Get layer icon
  const getLayerIcon = (type: string) => {
    switch (type) {
      case 'elevation': return Mountain;
      case 'imagery': return Camera;
      case 'thermal': return Thermometer;
      case 'geology': return Activity;
      case 'historical': return FileText;
      default: return MapIcon;
    }
  };

  // Get location type icon
  const getLocationIcon = (type: string) => {
    switch (type) {
      case 'volcano': return Mountain;
      case 'canyon': return Activity;
      case 'crater': return Target;
      case 'plain': return MapIcon;
      case 'basin': return Target;
      case 'ice': return Thermometer;
      default: return Navigation;
    }
  };

  return (
    <div className={`h-screen w-full bg-gray-900 text-white relative overflow-hidden ${fullscreen ? 'fixed inset-0 z-50' : ''}`}>
      {/* Main toolbar */}
      <motion.div
        className="absolute top-4 left-4 z-20 flex items-center space-x-2"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="bg-black/20 backdrop-blur-md rounded-lg border border-white/10 p-2 flex items-center space-x-2">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-md bg-white/10 hover:bg-white/20 transition-colors"
          >
            <MapIcon className="w-5 h-5" />
          </button>

          <div className="h-6 w-px bg-white/20" />

          <button
            onClick={() => selectTool('pan')}
            className={`p-2 rounded-md transition-colors ${
              selectedTool === 'pan' ? 'bg-blue-500/30 text-blue-300' : 'hover:bg-white/10'
            }`}
          >
            <MousePointer className="w-5 h-5" />
          </button>

          <button
            onClick={() => selectTool('measure')}
            className={`p-2 rounded-md transition-colors ${
              selectedTool === 'measure' ? 'bg-blue-500/30 text-blue-300' : 'hover:bg-white/10'
            }`}
          >
            <Ruler className="w-5 h-5" />
          </button>

          <button
            onClick={() => selectTool('info')}
            className={`p-2 rounded-md transition-colors ${
              selectedTool === 'info' ? 'bg-blue-500/30 text-blue-300' : 'hover:bg-white/10'
            }`}
          >
            <Info className="w-5 h-5" />
          </button>

          <div className="h-6 w-px bg-white/20" />

          <button
            onClick={zoomIn}
            className="p-2 rounded-md hover:bg-white/10 transition-colors"
          >
            <ZoomIn className="w-5 h-5" />
          </button>

          <button
            onClick={zoomOut}
            className="p-2 rounded-md hover:bg-white/10 transition-colors"
          >
            <ZoomOut className="w-5 h-5" />
          </button>

          <button
            onClick={resetView}
            className="p-2 rounded-md hover:bg-white/10 transition-colors"
          >
            <Home className="w-5 h-5" />
          </button>

          <div className="h-6 w-px bg-white/20" />

          <button
            onClick={() => setViewMode(viewMode === '2d' ? '3d' : '2d')}
            className={`p-2 rounded-md transition-colors ${
              viewMode === '3d' ? 'bg-blue-500/30 text-blue-300' : 'hover:bg-white/10'
            }`}
          >
            {viewMode === '2d' ? <Globe className="w-5 h-5" /> : <MapIcon className="w-5 h-5" />}
          </button>

          <button
            onClick={() => setFullscreen(!fullscreen)}
            className="p-2 rounded-md hover:bg-white/10 transition-colors"
          >
            {fullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
          </button>
        </div>
      </motion.div>

      {/* Map container */}
      <div ref={mapRef} className="w-full h-full" />

      {/* Coordinate display */}
      <motion.div
        className="absolute bottom-4 left-4 z-20"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1 }}
      >
        <div className="bg-black/20 backdrop-blur-md rounded-lg border border-white/10 px-3 py-2">
          <div className="text-sm font-mono">
            <span className="text-gray-300">Lat:</span> {coordinateDisplay.lat.toFixed(4)}°
            <span className="mx-2 text-gray-500">|</span>
            <span className="text-gray-300">Lon:</span> {coordinateDisplay.lon.toFixed(4)}°
            <span className="mx-2 text-gray-500">|</span>
            <span className="text-gray-300">Zoom:</span> {currentView.zoom.toFixed(1)}
          </div>
        </div>
      </motion.div>

      {/* Control panels */}
      <motion.div
        className="absolute top-4 right-4 z-20 flex flex-col space-y-2"
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3, delay: 0.2 }}
      >
        <button
          onClick={() => setLayerPanelOpen(!layerPanelOpen)}
          className="bg-black/20 backdrop-blur-md rounded-lg border border-white/10 p-3 hover:bg-white/10 transition-colors"
        >
          <Layers className="w-5 h-5" />
        </button>

        <button
          onClick={() => setSearchPanelOpen(!searchPanelOpen)}
          className="bg-black/20 backdrop-blur-md rounded-lg border border-white/10 p-3 hover:bg-white/10 transition-colors"
        >
          <Search className="w-5 h-5" />
        </button>

        <button
          onClick={() => setBookmarkPanelOpen(!bookmarkPanelOpen)}
          className="bg-black/20 backdrop-blur-md rounded-lg border border-white/10 p-3 hover:bg-white/10 transition-colors"
        >
          <Bookmark className="w-5 h-5" />
        </button>
      </motion.div>

      {/* Layer Panel */}
      <AnimatePresence>
        {layerPanelOpen && (
          <motion.div
            className="absolute top-16 right-4 z-30 w-80"
            initial={{ opacity: 0, x: 20, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 20, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <div className="bg-black/30 backdrop-blur-xl rounded-xl border border-white/10 p-4 max-h-96 overflow-y-auto">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center">
                  <Layers className="w-5 h-5 mr-2" />
                  Mars Data Layers
                </h3>
                <button
                  onClick={() => setLayerPanelOpen(false)}
                  className="p-1 rounded-md hover:bg-white/10 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-3">
                {Object.entries(MARS_LAYERS).map(([id, layer]) => {
                  const isActive = activeLayers.find(l => l.id === id);
                  const Icon = getLayerIcon(layer.type);

                  return (
                    <div key={id} className="border border-white/10 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center">
                          <Icon className="w-4 h-4 mr-2 text-gray-400" />
                          <span className="font-medium">{layer.name}</span>
                        </div>
                        <button
                          onClick={() => toggleLayer(id)}
                          className="p-1 rounded-md hover:bg-white/10 transition-colors"
                        >
                          {isActive?.visible ?
                            <Eye className="w-4 h-4 text-blue-400" /> :
                            <EyeOff className="w-4 h-4 text-gray-500" />
                          }
                        </button>
                      </div>

                      <p className="text-sm text-gray-400 mb-2">{layer.description}</p>

                      {isActive && (
                        <div>
                          <label className="text-xs text-gray-400 block mb-1">Opacity</label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={isActive.opacity}
                            onChange={(e) => setLayerOpacity(id, parseFloat(e.target.value))}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Search Panel */}
      <AnimatePresence>
        {searchPanelOpen && (
          <motion.div
            className="absolute top-16 right-4 z-30 w-80"
            initial={{ opacity: 0, x: 20, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 20, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <div className="bg-black/30 backdrop-blur-xl rounded-xl border border-white/10 p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center">
                  <Search className="w-5 h-5 mr-2" />
                  Search Mars
                </h3>
                <button
                  onClick={() => setSearchPanelOpen(false)}
                  className="p-1 rounded-md hover:bg-white/10 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <input
                type="text"
                placeholder="Search locations, features..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-400/50 mb-3"
              />

              <div className="max-h-64 overflow-y-auto space-y-2">
                {searchResults.map((location, index) => {
                  const Icon = getLocationIcon(location.type);

                  return (
                    <div
                      key={index}
                      onClick={() => {
                        flyToLocation(location.lat, location.lon, 8);
                        setSearchPanelOpen(false);
                      }}
                      className="p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors cursor-pointer"
                    >
                      <div className="flex items-center mb-1">
                        <Icon className="w-4 h-4 mr-2 text-orange-400" />
                        <span className="font-medium">{location.name}</span>
                      </div>
                      <p className="text-sm text-gray-400">{location.description}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        {location.lat.toFixed(2)}°, {location.lon.toFixed(2)}°
                      </p>
                    </div>
                  );
                })}

                {searchQuery && searchResults.length === 0 && (
                  <div className="text-center text-gray-500 py-4">
                    No locations found for "{searchQuery}"
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Bookmark Panel */}
      <AnimatePresence>
        {bookmarkPanelOpen && (
          <motion.div
            className="absolute top-16 right-4 z-30 w-80"
            initial={{ opacity: 0, x: 20, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 20, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <div className="bg-black/30 backdrop-blur-xl rounded-xl border border-white/10 p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center">
                  <Bookmark className="w-5 h-5 mr-2" />
                  Bookmarks
                </h3>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => addBookmark(`Location ${bookmarks.length + 1}`, 'Custom bookmark')}
                    className="p-1 rounded-md hover:bg-white/10 transition-colors"
                  >
                    <Plus className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setBookmarkPanelOpen(false)}
                    className="p-1 rounded-md hover:bg-white/10 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="max-h-64 overflow-y-auto space-y-2">
                {bookmarks.map((bookmark) => (
                  <div
                    key={bookmark.id}
                    className="p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors group"
                  >
                    <div className="flex items-center justify-between">
                      <div
                        onClick={() => {
                          flyToLocation(bookmark.lat, bookmark.lon, bookmark.zoom);
                          setBookmarkPanelOpen(false);
                        }}
                        className="flex-1 cursor-pointer"
                      >
                        <div className="font-medium mb-1">{bookmark.name}</div>
                        {bookmark.description && (
                          <p className="text-sm text-gray-400 mb-1">{bookmark.description}</p>
                        )}
                        <p className="text-xs text-gray-500">
                          {bookmark.lat.toFixed(4)}°, {bookmark.lon.toFixed(4)}° • Zoom {bookmark.zoom.toFixed(1)}
                        </p>
                      </div>
                      <button
                        onClick={() => removeBookmark(bookmark.id)}
                        className="p-1 rounded-md hover:bg-red-500/20 transition-colors opacity-0 group-hover:opacity-100"
                      >
                        <X className="w-4 h-4 text-red-400" />
                      </button>
                    </div>
                  </div>
                ))}

                {bookmarks.length === 0 && (
                  <div className="text-center text-gray-500 py-8">
                    <Bookmark className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>No bookmarks yet</p>
                    <p className="text-xs">Click + to add current view</p>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Attribution */}
      <div className="absolute bottom-4 right-4 z-10">
        <div className="bg-black/20 backdrop-blur-md rounded-lg border border-white/10 px-3 py-2">
          <div className="text-xs text-gray-400">
            Mars data © NASA/JPL/USGS • Interface © Mars-GIS
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfessionalMarsMapper;
