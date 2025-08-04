// Animation handling with CSS transitions
import {
    Bookmark,
    Globe,
    Info,
    Layers,
    Maximize2,
    Minimize2,
    MousePointer,
    Search,
    Target,
    X
} from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';

import { MARS_LAYERS, MARS_LOCATIONS } from '../../../shared/constants/mars-data';
import type { BookmarkType, LayerState, MarsLocation, ViewState } from '../../../shared/types/mars-types';

// CSS animations for smooth transitions
const animationStyles = `
  .fade-in-up {
    animation: fadeInUp 0.3s ease-out forwards;
  }

  .fade-in-right {
    animation: fadeInRight 0.3s ease-out forwards;
  }

  .scale-in {
    animation: scaleIn 0.2s ease-out forwards;
  }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(-20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes fadeInRight {
    from {
      opacity: 0;
      transform: translateX(20px) scale(0.95);
    }
    to {
      opacity: 1;
      transform: translateX(0) scale(1);
    }
  }

  @keyframes scaleIn {
    from {
      opacity: 0;
      transform: scale(0.9);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }
`;

const GoogleEarthMarsMapper: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState<ViewState>({
    centerLat: 0,
    centerLon: 0,
    zoom: 2,
    rotation: 0
  });

  const [layers] = useState<LayerState[]>(
    Object.keys(MARS_LAYERS).map(id => ({
      id,
      visible: MARS_LAYERS[id as keyof typeof MARS_LAYERS].visible,
      opacity: MARS_LAYERS[id as keyof typeof MARS_LAYERS].opacity
    }))
  );

  const [selectedTool, setSelectedTool] = useState<string>('pan');
  const [currentCoords, setCurrentCoords] = useState({ lat: 0, lon: 0 });
  const [layerPanelOpen, setLayerPanelOpen] = useState(false);
  const [searchPanelOpen, setSearchPanelOpen] = useState(false);
  const [bookmarkPanelOpen, setBookmarkPanelOpen] = useState(false);
  const [locationPanelOpen, setLocationPanelOpen] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState<MarsLocation | null>(null);
  const [bookmarks, setBookmarks] = useState<BookmarkType[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [fullscreen, setFullscreen] = useState(false);

  // Canvas rendering and interaction functions would go here
  const renderMarsMap = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#0f1419';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Simple Mars globe representation
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) / 3;

    // Draw Mars surface
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.fillStyle = '#cd5c5c';
    ctx.fill();

    // Add some surface details
    ctx.fillStyle = '#a0522d';
    for (let i = 0; i < 20; i++) {
      const x = centerX + (Math.random() - 0.5) * radius * 1.5;
      const y = centerY + (Math.random() - 0.5) * radius * 1.5;
      const size = Math.random() * 5 + 2;
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, []);

  const screenToLatLon = useCallback((screenX: number, screenY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { lat: 0, lon: 0 };

    const rect = canvas.getBoundingClientRect();
    const x = screenX - rect.left;
    const y = screenY - rect.top;

    // Simple screen to lat/lon conversion
    const lat = 90 - (y / canvas.height) * 180;
    const lon = (x / canvas.width) * 360 - 180;

    return { lat, lon };
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const coords = screenToLatLon(e.clientX, e.clientY);
    setCurrentCoords(coords);
  }, [screenToLatLon]);

  const handleClick = useCallback((e: React.MouseEvent) => {
    if (selectedTool === 'marker') {
      const coords = screenToLatLon(e.clientX, e.clientY);

      // Find nearest location
      const nearest = MARS_LOCATIONS.reduce((closest, location) => {
        const distance = Math.sqrt(
          Math.pow(location.lat - coords.lat, 2) +
          Math.pow(location.lon - coords.lon, 2)
        );
        return distance < closest.distance ? { location, distance } : closest;
      }, { location: MARS_LOCATIONS[0], distance: Infinity });

      if (nearest.distance < 5) {
        setSelectedLocation(nearest.location);
        setLocationPanelOpen(true);
      }
    }
  }, [selectedTool, screenToLatLon]);

  const addBookmark = useCallback((name: string, description?: string) => {
    const newBookmark: BookmarkType = {
      id: Date.now().toString(),
      name,
      description: description || '',
      lat: currentCoords.lat,
      lon: currentCoords.lon,
      zoom: view.zoom,
      created: new Date()
    };
    setBookmarks(prev => [...prev, newBookmark]);
  }, [currentCoords, view.zoom]);

  const flyToLocation = useCallback((lat: number, lon: number, zoom: number) => {
    setView({ centerLat: lat, centerLon: lon, zoom, rotation: 0 });
  }, []);

  useEffect(() => {
    renderMarsMap();
  }, [renderMarsMap, view, layers]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      renderMarsMap();
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    return () => window.removeEventListener('resize', resizeCanvas);
  }, [renderMarsMap]);

  return (
    <div className={`h-screen w-full bg-gray-900 text-white relative overflow-hidden ${fullscreen ? 'fixed inset-0 z-50' : ''}`}>
      {/* Inject CSS animations */}
      <style>{animationStyles}</style>

      {/* Main toolbar */}
      <div className="absolute top-4 left-4 z-20 flex items-center space-x-2 fade-in-up">
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
            onClick={() => setSelectedTool('marker')}
            className={`p-2 rounded-md transition-colors ${
              selectedTool === 'marker' ? 'bg-blue-500/30 text-blue-300' : 'hover:bg-white/10'
            }`}
            title="Select Location"
          >
            <Target className="w-4 h-4" />
          </button>

          <div className="h-6 w-px bg-white/20" />

          <button
            onClick={() => setFullscreen(!fullscreen)}
            className="p-2 rounded-md hover:bg-white/10 transition-colors"
            title="Toggle Fullscreen"
          >
            {fullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-crosshair"
        onMouseMove={handleMouseMove}
        onClick={handleClick}
      />

      {/* Control panels */}
      <div className="absolute top-4 right-[316px] z-20 flex flex-col space-y-2 fade-in-right">
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
      </div>

      {/* Coordinate display */}
      <div className="absolute bottom-4 left-4 z-20 scale-in">
        <div className="bg-black/20 backdrop-blur-md rounded-lg border border-white/10 px-3 py-2">
          <div className="text-sm font-mono flex items-center space-x-4">
            <div>
              <span className="text-gray-300">Lat:</span> {currentCoords.lat.toFixed(4)}°
            </div>
            <div>
              <span className="text-gray-300">Lon:</span> {currentCoords.lon.toFixed(4)}°
            </div>
          </div>
        </div>
      </div>

      {/* Layer Panel */}
      {layerPanelOpen && (
        <div className="absolute top-16 right-4 z-30 w-80 fade-in-right">
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
            <div className="text-sm text-gray-400">
              Layer controls would be implemented here with real NASA Mars data layers.
            </div>
          </div>
        </div>
      )}

      {/* Search Panel */}
      {searchPanelOpen && (
        <div className="absolute top-16 right-4 z-30 w-80 fade-in-right">
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
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search locations..."
              className="w-full px-3 py-2 bg-black/20 border border-white/10 rounded-lg text-white placeholder-gray-400"
            />
          </div>
        </div>
      )}

      {/* Bookmark Panel */}
      {bookmarkPanelOpen && (
        <div className="absolute top-16 right-4 z-30 w-80 fade-in-right">
          <div className="bg-black/30 backdrop-blur-xl rounded-xl border border-white/10 p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold flex items-center">
                <Bookmark className="w-5 h-5 mr-2" />
                Bookmarks
              </h3>
              <button
                onClick={() => setBookmarkPanelOpen(false)}
                className="p-1 rounded-md hover:bg-white/10 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-2">
              {bookmarks.length === 0 ? (
                <div className="text-sm text-gray-400">No bookmarks yet</div>
              ) : (
                bookmarks.map(bookmark => (
                  <div key={bookmark.id} className="p-2 bg-black/20 rounded-lg">
                    <div className="font-medium">{bookmark.name}</div>
                    <div className="text-xs text-gray-400">
                      {bookmark.lat.toFixed(4)}°, {bookmark.lon.toFixed(4)}°
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* Location Info Panel */}
      {locationPanelOpen && selectedLocation && (
        <div className="absolute bottom-20 left-4 z-30 w-80 scale-in">
          <div className="bg-black/30 backdrop-blur-xl rounded-xl border border-white/10 p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold flex items-center">
                <Info className="w-5 h-5 mr-2 text-orange-400" />
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

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-gray-400">Latitude</div>
                  <div className="font-mono">{selectedLocation.lat.toFixed(4)}°</div>
                </div>
                <div>
                  <div className="text-gray-400">Longitude</div>
                  <div className="font-mono">{selectedLocation.lon.toFixed(4)}°</div>
                </div>
                <div>
                  <div className="text-gray-400">Type</div>
                  <div className="capitalize">{selectedLocation.type}</div>
                </div>
                <div>
                  <div className="text-gray-400">Elevation</div>
                  <div>{selectedLocation.elevation?.toLocaleString() ?? 'Unknown'}m</div>
                </div>
              </div>

              <div className="flex gap-2 pt-2">
                <button
                  onClick={() => flyToLocation(selectedLocation.lat, selectedLocation.lon, 10)}
                  className="flex-1 px-3 py-2 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 rounded-lg text-blue-300 text-sm transition-colors"
                >
                  Fly To
                </button>
                <button
                  onClick={() => addBookmark(selectedLocation.name, selectedLocation.description)}
                  className="flex-1 px-3 py-2 bg-orange-500/20 hover:bg-orange-500/30 border border-orange-500/30 rounded-lg text-orange-300 text-sm transition-colors"
                >
                  Bookmark
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

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
