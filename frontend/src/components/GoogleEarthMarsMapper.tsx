// Animation handling with C) from 'lucide-react';

// Mars layer configurationsransitions
import {
    Activity,
    Bookmark,
    Camera,
    Compass,
    Eye,
    EyeOff,
    Globe,
    Home,
    Info,
    Layers,
    Map as MapIcon,
    Maximize2,
    Minimize2,
    Mountain,
    MousePointer,
    Move3d,
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
import React, { useCallback, useEffect, useRef, useState } from 'react';

// Mars layer configurations
const MARS_LAYERS = {
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
};

// Famous Mars locations
const MARS_LOCATIONS = [
  { id: 1, name: 'Olympus Mons', lat: 18.65, lon: -133.8, type: 'volcano', description: 'Largest volcano in the solar system', elevation: 21287 },
  { id: 2, name: 'Valles Marineris', lat: -14, lon: -59, type: 'canyon', description: 'Massive canyon system', elevation: -7000 },
  { id: 3, name: 'Gale Crater', lat: -5.4, lon: 137.8, type: 'crater', description: 'Curiosity rover landing site', elevation: -4500 },
  { id: 4, name: 'Jezero Crater', lat: 18.44, lon: 77.45, type: 'crater', description: 'Perseverance rover landing site', elevation: -2500 },
  { id: 5, name: 'Acidalia Planitia', lat: 46.7, lon: -29.8, type: 'plain', description: 'Northern lowlands region', elevation: -4000 },
  { id: 6, name: 'Hellas Planitia', lat: -42.4, lon: 70.5, type: 'basin', description: 'Largest impact crater on Mars', elevation: -8200 },
  { id: 7, name: 'Polar Ice Cap (North)', lat: 85, lon: 0, type: 'ice', description: 'North polar ice cap', elevation: -5000 },
  { id: 8, name: 'Polar Ice Cap (South)', lat: -85, lon: 0, type: 'ice', description: 'South polar ice cap', elevation: -6000 },
  { id: 9, name: 'Tharsis Volcanic Province', lat: 0, lon: -100, type: 'volcano', description: 'Major volcanic region', elevation: 10000 },
  { id: 10, name: 'Chryse Planitia', lat: 20, lon: -50, type: 'plain', description: 'Viking 1 landing site', elevation: -3000 }
];

interface LayerState {
  id: string;
  visible: boolean;
  opacity: number;
}

interface ViewState {
  centerLat: number;
  centerLon: number;
  zoom: number;
  rotation: number;
}

interface BookmarkType {
  id: string;
  name: string;
  lat: number;
  lon: number;
  zoom: number;
  description?: string;
  created: Date;
}

const GoogleEarthMarsMapper: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState<ViewState>({
    centerLat: 0,
    centerLon: 0,
    zoom: 2,
    rotation: 0
  });

  const [layers, setLayers] = useState<LayerState[]>(
    Object.keys(MARS_LAYERS).map(id => ({
      id,
      visible: MARS_LAYERS[id as keyof typeof MARS_LAYERS].visible,
      opacity: MARS_LAYERS[id as keyof typeof MARS_LAYERS].opacity
    }))
  );

  const [bookmarks, setBookmarks] = useState<BookmarkType[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<typeof MARS_LOCATIONS>([]);
  const [selectedLocation, setSelectedLocation] = useState<typeof MARS_LOCATIONS[0] | null>(null);
  const [currentCoords, setCurrentCoords] = useState({ lat: 0, lon: 0 });
  const [selectedTool, setSelectedTool] = useState<string>('pan');
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

  // Panel states
  const [layerPanelOpen, setLayerPanelOpen] = useState(false);
  const [searchPanelOpen, setSearchPanelOpen] = useState(false);
  const [bookmarkPanelOpen, setBookmarkPanelOpen] = useState(false);
  const [locationPanelOpen, setLocationPanelOpen] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d');

  // Canvas rendering
  const renderMarsMap = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);

    // Set background - space black
    ctx.fillStyle = '#000012';
    ctx.fillRect(0, 0, width, height);

    // Calculate scale and projection
    const scale = Math.pow(2, view.zoom);
    const pixelsPerDegree = (width / 360) * scale;

    // Draw Mars surface
    const visibleLayers = layers.filter(layer => {
      const config = MARS_LAYERS[layer.id as keyof typeof MARS_LAYERS];
      return layer.visible && config;
    });

    visibleLayers.forEach(layer => {
      const config = MARS_LAYERS[layer.id as keyof typeof MARS_LAYERS];
      ctx.globalAlpha = layer.opacity;

      // Create gradient for Mars surface based on layer type
      if (config.type === 'elevation') {
        const gradient = ctx.createRadialGradient(width/2, height/2, 0, width/2, height/2, Math.min(width, height)/2);
        gradient.addColorStop(0, '#CD853F'); // Sandy brown
        gradient.addColorStop(0.3, '#A0522D'); // Sienna
        gradient.addColorStop(0.6, '#8B4513'); // Saddle brown
        gradient.addColorStop(1, '#654321'); // Dark brown
        ctx.fillStyle = gradient;
      } else if (config.type === 'thermal') {
        const gradient = ctx.createRadialGradient(width/2, height/2, 0, width/2, height/2, Math.min(width, height)/2);
        gradient.addColorStop(0, '#FF4500'); // Orange red
        gradient.addColorStop(0.5, '#FF6347'); // Tomato
        gradient.addColorStop(1, '#B22222'); // Fire brick
        ctx.fillStyle = gradient;
      } else {
        ctx.fillStyle = config.color;
      }

      // Draw Mars sphere projection
      ctx.beginPath();
      ctx.arc(width/2, height/2, Math.min(width, height)/3, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.globalAlpha = 1.0;

    // Draw coordinate grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;

    // Latitude lines
    for (let lat = -90; lat <= 90; lat += 30) {
      const y = height/2 + (lat - view.centerLat) * pixelsPerDegree;
      if (y >= 0 && y <= height) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }

    // Longitude lines
    for (let lon = -180; lon <= 180; lon += 30) {
      const x = width/2 + (lon - view.centerLon) * pixelsPerDegree;
      if (x >= 0 && x <= width) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
    }

    // Draw locations
    MARS_LOCATIONS.forEach(location => {
      const x = width/2 + (location.lon - view.centerLon) * pixelsPerDegree;
      const y = height/2 - (location.lat - view.centerLat) * pixelsPerDegree;

      if (x >= 0 && x <= width && y >= 0 && y <= height) {
        // Location marker
        ctx.fillStyle = getLocationColor(location.type);
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, 2 * Math.PI);
        ctx.fill();

        // White border
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Location label (if zoomed in enough)
        if (view.zoom > 3) {
          ctx.fillStyle = '#ffffff';
          ctx.font = '12px Inter, sans-serif';
          ctx.fillText(location.name, x + 10, y + 4);
        }
      }
    });

    // Highlight selected location
    if (selectedLocation) {
      const x = width/2 + (selectedLocation.lon - view.centerLon) * pixelsPerDegree;
      const y = height/2 - (selectedLocation.lat - view.centerLat) * pixelsPerDegree;

      if (x >= 0 && x <= width && y >= 0 && y <= height) {
        ctx.strokeStyle = '#00FF88';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }

  }, [view, layers, selectedLocation]);

  // Helper functions
  const getLocationColor = (type: string): string => {
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

  const getLayerIcon = (type: string) => {
    switch (type) {
      case 'elevation': return Mountain;
      case 'imagery': return Camera;
      case 'thermal': return Thermometer;
      case 'geology': return Activity;
      case 'atmosphere': return Globe;
      default: return MapIcon;
    }
  };

  // Coordinate conversion
  const screenToLatLon = useCallback((x: number, y: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { lat: 0, lon: 0 };

    const rect = canvas.getBoundingClientRect();
    const canvasX = x - rect.left;
    const canvasY = y - rect.top;

    const scale = Math.pow(2, view.zoom);
    const pixelsPerDegree = (canvas.width / 360) * scale;

    const lat = view.centerLat - (canvasY - canvas.height/2) / pixelsPerDegree;
    const lon = view.centerLon + (canvasX - canvas.width/2) / pixelsPerDegree;

    return {
      lat: Math.max(-90, Math.min(90, lat)),
      lon: ((lon + 180) % 360) - 180
    };
  }, [view]);

  // Navigation functions
  const flyToLocation = useCallback((lat: number, lon: number, zoom?: number) => {
    setView(prev => ({
      ...prev,
      centerLat: lat,
      centerLon: lon,
      zoom: zoom || Math.max(prev.zoom, 6)
    }));
  }, []);

  const zoomIn = useCallback(() => {
    setView(prev => ({ ...prev, zoom: Math.min(prev.zoom + 1, 15) }));
  }, []);

  const zoomOut = useCallback(() => {
    setView(prev => ({ ...prev, zoom: Math.max(prev.zoom - 1, 1) }));
  }, []);

  const resetView = useCallback(() => {
    setView({ centerLat: 0, centerLon: 0, zoom: 2, rotation: 0 });
  }, []);

  // Layer management
  const toggleLayer = useCallback((layerId: string) => {
    setLayers(prev => prev.map(layer =>
      layer.id === layerId ? { ...layer, visible: !layer.visible } : layer
    ));
  }, []);

  const setLayerOpacity = useCallback((layerId: string, opacity: number) => {
    setLayers(prev => prev.map(layer =>
      layer.id === layerId ? { ...layer, opacity } : layer
    ));
  }, []);

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

  // Mouse events
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (selectedTool === 'pan') {
      setIsDragging(true);
      setLastMousePos({ x: e.clientX, y: e.clientY });
    }
  }, [selectedTool]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const coords = screenToLatLon(e.clientX, e.clientY);
    setCurrentCoords(coords);

    if (isDragging && selectedTool === 'pan') {
      const deltaX = e.clientX - lastMousePos.x;
      const deltaY = e.clientY - lastMousePos.y;

      const scale = Math.pow(2, view.zoom);
      const pixelsPerDegree = (canvasRef.current?.width || 800) / 360 * scale;

      setView(prev => ({
        ...prev,
        centerLon: prev.centerLon - deltaX / pixelsPerDegree,
        centerLat: prev.centerLat + deltaY / pixelsPerDegree
      }));

      setLastMousePos({ x: e.clientX, y: e.clientY });
    }
  }, [isDragging, selectedTool, lastMousePos, view.zoom, screenToLatLon]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleClick = useCallback((e: React.MouseEvent) => {
    if (selectedTool === 'info') {
      const coords = screenToLatLon(e.clientX, e.clientY);

      // Find nearest location
      const nearest = MARS_LOCATIONS.reduce((closest, location) => {
        const distance = Math.sqrt(
          Math.pow(location.lat - coords.lat, 2) +
          Math.pow(location.lon - coords.lon, 2)
        );
        return distance < closest.distance ? { location, distance } : closest;
      }, { location: MARS_LOCATIONS[0], distance: Infinity });

      if (nearest.distance < 5) { // Within 5 degrees
        setSelectedLocation(nearest.location);
        setLocationPanelOpen(true);
      }
    }
  }, [selectedTool, screenToLatLon]);

  // Bookmark management
  const addBookmark = useCallback((name: string, description?: string) => {
    const newBookmark: BookmarkType = {
      id: Date.now().toString(),
      name,
      lat: view.centerLat,
      lon: view.centerLon,
      zoom: view.zoom,
      description,
      created: new Date()
    };

    setBookmarks(prev => [...prev, newBookmark]);
  }, [view]);

  const removeBookmark = useCallback((id: string) => {
    setBookmarks(prev => prev.filter(b => b.id !== id));
  }, []);

  // Render canvas
  useEffect(() => {
    renderMarsMap();
  }, [renderMarsMap]);

  // Resize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';

      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      }

      renderMarsMap();
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    return () => window.removeEventListener('resize', resizeCanvas);
  }, [renderMarsMap]);

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
          <div className="flex items-center space-x-1">
            <Globe className="w-5 h-5 text-orange-400" />
            <span className="font-semibold text-sm">Mars Explorer</span>
          </div>

          <div className="h-6 w-px bg-white/20" />

          <button
            onClick={() => setSelectedTool('pan')}
            className={`p-2 rounded-md transition-colors ${
              selectedTool === 'pan' ? 'bg-blue-500/30 text-blue-300' : 'hover:bg-white/10'
            }`}
            title="Pan"
          >
            <MousePointer className="w-4 h-4" />
          </button>

          <button
            onClick={() => setSelectedTool('info')}
            className={`p-2 rounded-md transition-colors ${
              selectedTool === 'info' ? 'bg-blue-500/30 text-blue-300' : 'hover:bg-white/10'
            }`}
            title="Info"
          >
            <Info className="w-4 h-4" />
          </button>

          <button
            onClick={() => setSelectedTool('measure')}
            className={`p-2 rounded-md transition-colors ${
              selectedTool === 'measure' ? 'bg-blue-500/30 text-blue-300' : 'hover:bg-white/10'
            }`}
            title="Measure"
          >
            <Ruler className="w-4 h-4" />
          </button>

          <div className="h-6 w-px bg-white/20" />

          <button
            onClick={zoomIn}
            className="p-2 rounded-md hover:bg-white/10 transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          <button
            onClick={zoomOut}
            className="p-2 rounded-md hover:bg-white/10 transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>

          <button
            onClick={resetView}
            className="p-2 rounded-md hover:bg-white/10 transition-colors"
            title="Reset View"
          >
            <Home className="w-4 h-4" />
          </button>

          <div className="h-6 w-px bg-white/20" />

          <button
            onClick={() => setViewMode(viewMode === '2d' ? '3d' : '2d')}
            className={`p-2 rounded-md transition-colors ${
              viewMode === '3d' ? 'bg-blue-500/30 text-blue-300' : 'hover:bg-white/10'
            }`}
            title={viewMode === '2d' ? 'Switch to 3D' : 'Switch to 2D'}
          >
            {viewMode === '2d' ? <Move3d className="w-4 h-4" /> : <MapIcon className="w-4 h-4" />}
          </button>

          <button
            onClick={() => setFullscreen(!fullscreen)}
            className="p-2 rounded-md hover:bg-white/10 transition-colors"
            title={fullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          >
            {fullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </motion.div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-crosshair"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
      />

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
          title="Layers"
        >
          <Layers className="w-5 h-5" />
        </button>

        <button
          onClick={() => setSearchPanelOpen(!searchPanelOpen)}
          className="bg-black/20 backdrop-blur-md rounded-lg border border-white/10 p-3 hover:bg-white/10 transition-colors"
          title="Search"
        >
          <Search className="w-5 h-5" />
        </button>

        <button
          onClick={() => setBookmarkPanelOpen(!bookmarkPanelOpen)}
          className="bg-black/20 backdrop-blur-md rounded-lg border border-white/10 p-3 hover:bg-white/10 transition-colors"
          title="Bookmarks"
        >
          <Bookmark className="w-5 h-5" />
        </button>
      </motion.div>

      {/* Coordinate display */}
      <motion.div
        className="absolute bottom-4 left-4 z-20"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1 }}
      >
        <div className="bg-black/20 backdrop-blur-md rounded-lg border border-white/10 px-3 py-2">
          <div className="text-sm font-mono flex items-center space-x-4">
            <div>
              <span className="text-gray-300">Lat:</span> {currentCoords.lat.toFixed(4)}°
            </div>
            <div>
              <span className="text-gray-300">Lon:</span> {currentCoords.lon.toFixed(4)}°
            </div>
            <div>
              <span className="text-gray-300">Zoom:</span> {view.zoom.toFixed(1)}
            </div>
            <div className="flex items-center">
              <Compass className="w-4 h-4 mr-1 text-orange-400" />
              <span className="text-gray-300">Mars</span>
            </div>
          </div>
        </div>
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
                {Object.entries(MARS_LAYERS).map(([id, layerConfig]) => {
                  const layerState = layers.find(l => l.id === id);
                  const Icon = getLayerIcon(layerConfig.type);

                  return (
                    <div key={id} className="border border-white/10 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center">
                          <Icon className="w-4 h-4 mr-2 text-gray-400" />
                          <span className="font-medium">{layerConfig.name}</span>
                        </div>
                        <button
                          onClick={() => toggleLayer(id)}
                          className="p-1 rounded-md hover:bg-white/10 transition-colors"
                        >
                          {layerState?.visible ?
                            <Eye className="w-4 h-4 text-blue-400" /> :
                            <EyeOff className="w-4 h-4 text-gray-500" />
                          }
                        </button>
                      </div>

                      <p className="text-sm text-gray-400 mb-2">{layerConfig.description}</p>

                      {layerState && layerState.visible && (
                        <div>
                          <label className="text-xs text-gray-400 block mb-1">
                            Opacity: {Math.round(layerState.opacity * 100)}%
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={layerState.opacity}
                            onChange={(e) => setLayerOpacity(id, parseFloat(e.target.value))}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
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
                {searchResults.map((location) => {
                  const Icon = getLocationIcon(location.type);

                  return (
                    <button
                      key={location.id}
                      onClick={() => {
                        flyToLocation(location.lat, location.lon, 8);
                        setSelectedLocation(location);
                        setSearchPanelOpen(false);
                      }}
                      className="w-full text-left p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors"
                    >
                      <div className="flex items-center mb-1">
                        <Icon className="w-4 h-4 mr-2 text-orange-400" />
                        <span className="font-medium">{location.name}</span>
                      </div>
                      <p className="text-sm text-gray-400">{location.description}</p>
                      <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
                        <span>{location.lat.toFixed(2)}°, {location.lon.toFixed(2)}°</span>
                        <span>Elev: {location.elevation.toLocaleString()}m</span>
                      </div>
                    </button>
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
                    onClick={() => addBookmark(`Mars Location ${bookmarks.length + 1}`, 'Custom bookmark')}
                    className="p-1 rounded-md hover:bg-white/10 transition-colors"
                    title="Add Bookmark"
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
                      <button
                        onClick={() => {
                          flyToLocation(bookmark.lat, bookmark.lon, bookmark.zoom);
                          setBookmarkPanelOpen(false);
                        }}
                        className="flex-1 text-left"
                      >
                        <div className="font-medium mb-1">{bookmark.name}</div>
                        {bookmark.description && (
                          <p className="text-sm text-gray-400 mb-1">{bookmark.description}</p>
                        )}
                        <p className="text-xs text-gray-500">
                          {bookmark.lat.toFixed(4)}°, {bookmark.lon.toFixed(4)}° • Zoom {bookmark.zoom.toFixed(1)}
                        </p>
                      </button>
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

      {/* Location Info Panel */}
      <AnimatePresence>
        {locationPanelOpen && selectedLocation && (
          <motion.div
            className="absolute bottom-20 left-4 z-30 w-80"
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <div className="bg-black/30 backdrop-blur-xl rounded-xl border border-white/10 p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-semibold flex items-center">
                  {React.createElement(getLocationIcon(selectedLocation.type), { className: "w-5 h-5 mr-2 text-orange-400" })}
                  {selectedLocation.name}
                </h3>
                <button
                  onClick={() => setLocationPanelOpen(false)}
                  className="p-1 rounded-md hover:bg-white/10 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-3">
                <p className="text-gray-300">{selectedLocation.description}</p>

                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-gray-400">Latitude:</span>
                    <div className="font-mono">{selectedLocation.lat.toFixed(4)}°</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Longitude:</span>
                    <div className="font-mono">{selectedLocation.lon.toFixed(4)}°</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Type:</span>
                    <div className="capitalize">{selectedLocation.type}</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Elevation:</span>
                    <div>{selectedLocation.elevation.toLocaleString()}m</div>
                  </div>
                </div>

                <div className="flex space-x-2 pt-2">
                  <button
                    onClick={() => flyToLocation(selectedLocation.lat, selectedLocation.lon, 10)}
                    className="flex-1 px-3 py-2 bg-blue-600/20 text-blue-300 rounded-lg hover:bg-blue-600/30 transition-colors text-sm"
                  >
                    Zoom To
                  </button>
                  <button
                    onClick={() => addBookmark(selectedLocation.name, selectedLocation.description)}
                    className="flex-1 px-3 py-2 bg-orange-600/20 text-orange-300 rounded-lg hover:bg-orange-600/30 transition-colors text-sm"
                  >
                    Bookmark
                  </button>
                </div>
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

export default GoogleEarthMarsMapper;
