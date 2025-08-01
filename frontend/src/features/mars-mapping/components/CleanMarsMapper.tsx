// Clean Mars mapping interface without external animation dependencies
import {
    Activity,
    Bookmark,
    Download,
    Layers,
    MapPin,
    RotateCcw,
    Search,
    ZoomIn,
    ZoomOut,
} from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';

// Import shared modules
import { MARS_LAYERS, MARS_LOCATIONS } from '../../../shared/constants/mars-data';
import type { BookmarkType, LayerState, MarsLocation, ViewState } from '../../../shared/types/mars-types';
import { formatCoordinates, getLocationColor, searchMarsLocations } from '../../../shared/utils/mars-utils';

const CleanMarsMapper: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [viewState, setViewState] = useState<ViewState>({
    centerLat: 0,
    centerLon: 0,
    zoom: 1,
    rotation: 0,
  });

  const [layerStates, setLayerStates] = useState<Record<string, LayerState>>(() => {
    const initialStates: Record<string, LayerState> = {};
    Object.entries(MARS_LAYERS).forEach(([key, layer]) => {
      initialStates[key] = {
        id: key,
        visible: layer.visible,
        opacity: layer.opacity,
      };
    });
    return initialStates;
  });

  const [bookmarks, setBookmarks] = useState<BookmarkType[]>([]);
  const [selectedLocation, setSelectedLocation] = useState<MarsLocation | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const [showBookmarks, setShowBookmarks] = useState(false);
  const [showLayerPanel] = useState(true);

  // Canvas drawing functions
  const drawMarsBase = useCallback((ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    const { width, height } = canvas;

    // Draw Mars base terrain
    ctx.fillStyle = '#CD853F';
    ctx.fillRect(0, 0, width, height);

    // Add texture patterns
    ctx.fillStyle = '#A0522D';
    for (let i = 0; i < 50; i++) {
      ctx.beginPath();
      ctx.arc(
        Math.random() * width,
        Math.random() * height,
        Math.random() * 3 + 1,
        0,
        2 * Math.PI
      );
      ctx.fill();
    }
  }, []);

  const drawMarsLayer = useCallback((
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    layerName: string,
    opacity: number
  ) => {
    const { width, height } = canvas;
    const layerEntry = Object.entries(MARS_LAYERS).find(([key]) => key === layerName);
    if (!layerEntry) return;

    const [, layer] = layerEntry;
    ctx.globalAlpha = opacity;

    switch (layerName) {
      case 'elevation': {
        ctx.fillStyle = layer.color;
        ctx.fillRect(0, 0, width, height);
        break;
      }
      case 'thermal': {
        const gradient = ctx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, 'rgba(0, 0, 255, 0.3)');
        gradient.addColorStop(0.5, 'rgba(255, 255, 0, 0.3)');
        gradient.addColorStop(1, 'rgba(255, 0, 0, 0.3)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        break;
      }
      case 'imagery': {
        for (let i = 0; i < 10; i++) {
          ctx.strokeStyle = `rgba(139, 69, 19, ${0.1 + i * 0.05})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(0, (height / 10) * i);
          ctx.lineTo(width, (height / 10) * i);
          ctx.stroke();
        }
        break;
      }
      case 'geology': {
        ctx.fillStyle = 'rgba(160, 82, 45, 0.4)';
        for (let i = 0; i < 20; i++) {
          ctx.beginPath();
          ctx.arc(
            Math.random() * width,
            Math.random() * height,
            Math.random() * 20 + 5,
            0,
            2 * Math.PI
          );
          ctx.fill();
        }
        break;
      }
      case 'atmosphere': {
        ctx.fillStyle = 'rgba(0, 191, 255, 0.3)';
        for (let i = 0; i < 5; i++) {
          ctx.beginPath();
          ctx.arc(
            Math.random() * width,
            Math.random() * height,
            Math.random() * 30 + 10,
            0,
            2 * Math.PI
          );
          ctx.fill();
        }
        break;
      }
    }

    ctx.globalAlpha = 1;
  }, []);

  const drawMarsLocations = useCallback((ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    const { width, height } = canvas;

    MARS_LOCATIONS.forEach((location) => {
      // Convert lat/lng to canvas coordinates (simplified projection)
      const x = ((location.lon + 180) / 360) * width;
      const y = ((90 - location.lat) / 180) * height;

      // Draw location marker
      ctx.fillStyle = getLocationColor(location.type);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();

      // Draw border
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw label
      ctx.fillStyle = '#fff';
      ctx.font = '12px Arial';
      ctx.fillText(location.name, x + 8, y - 8);
    });
  }, []);

  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw base Mars terrain
    drawMarsBase(ctx, canvas);

    // Draw active layers
    Object.entries(layerStates).forEach(([layerName, layerState]) => {
      if (layerState.visible) {
        drawMarsLayer(ctx, canvas, layerName, layerState.opacity);
      }
    });

    // Draw Mars locations
    drawMarsLocations(ctx, canvas);
  }, [layerStates, drawMarsBase, drawMarsLayer, drawMarsLocations]);

  // Event handlers
  const handleCanvasClick = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Find clicked location
    const clickedLocation = MARS_LOCATIONS.find(location => {
      const locX = ((location.lon + 180) / 360) * canvas.width;
      const locY = ((90 - location.lat) / 180) * canvas.height;
      const distance = Math.sqrt((x - locX) ** 2 + (y - locY) ** 2);
      return distance < 10; // 10px radius
    });

    if (clickedLocation) {
      setSelectedLocation(clickedLocation);
    } else {
      setSelectedLocation(null);
    }
  }, []);

  const toggleLayer = useCallback((layerName: string) => {
    setLayerStates(prev => ({
      ...prev,
      [layerName]: {
        ...prev[layerName],
        visible: !prev[layerName].visible,
      },
    }));
  }, []);

  const updateLayerOpacity = useCallback((layerName: string, opacity: number) => {
    setLayerStates(prev => ({
      ...prev,
      [layerName]: {
        ...prev[layerName],
        opacity,
      },
    }));
  }, []);

  const zoomIn = useCallback(() => {
    setViewState(prev => ({ ...prev, zoom: Math.min(prev.zoom * 1.2, 10) }));
  }, []);

  const zoomOut = useCallback(() => {
    setViewState(prev => ({ ...prev, zoom: Math.max(prev.zoom / 1.2, 0.1) }));
  }, []);

  const resetView = useCallback(() => {
    setViewState({ centerLat: 0, centerLon: 0, zoom: 1, rotation: 0 });
  }, []);

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

  const exportData = useCallback(() => {
    const data = {
      viewState,
      layerStates,
      bookmarks,
      selectedLocation,
    };
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'mars-mapping-data.json';
    link.click();
    URL.revokeObjectURL(url);
  }, [viewState, layerStates, bookmarks, selectedLocation]);

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    redrawCanvas();
  }, [redrawCanvas]);

  // Redraw when layers change
  useEffect(() => {
    redrawCanvas();
  }, [redrawCanvas]);

  // Search functionality
  const searchResults = searchMarsLocations(searchQuery, MARS_LOCATIONS);

  return (
    <div className="w-full h-screen bg-black relative overflow-hidden">
      {/* Main Canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-crosshair"
        onClick={handleCanvasClick}
      />

      {/* Control Panel */}
      <div className="absolute top-4 left-4 z-30 space-y-2">
        <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-3 border border-gray-700">
          <h3 className="text-white text-sm font-semibold mb-2">Controls</h3>
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
              Reset
            </button>
          </div>
        </div>
      </div>

      {/* Layer Panel */}
      {showLayerPanel && (
        <div className="absolute left-4 top-1/2 transform -translate-y-1/2 z-30 space-y-2">
          <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-3 border border-gray-700 transition-opacity">
            <h3 className="text-white text-sm font-semibold mb-2 flex items-center">
              <Layers className="w-4 h-4 mr-2" />
              Layers
            </h3>
            {Object.entries(layerStates).map(([layerName, layerState]) => (
              <div key={layerName} className="mb-3">
                <div className="flex items-center justify-between mb-1">
                  <label className="text-white text-xs flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={layerState.visible}
                      onChange={() => toggleLayer(layerName)}
                      className="mr-2 rounded"
                    />
                    {layerName.charAt(0).toUpperCase() + layerName.slice(1)}
                  </label>
                </div>
                {layerState.visible && (
                  <div className="ml-6">
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={layerState.opacity}
                      onChange={(e) => updateLayerOpacity(layerName, parseFloat(e.target.value))}
                      className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer slider"
                    />
                    <div className="text-xs text-gray-400 mt-1">
                      Opacity: {Math.round(layerState.opacity * 100)}%
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
          className="p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          <Search className="w-5 h-5" />
        </button>
        <button
          onClick={() => setShowBookmarks(!showBookmarks)}
          className="p-3 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
        >
          <Bookmark className="w-5 h-5" />
        </button>
        <button
          onClick={exportData}
          className="p-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
        >
          <Download className="w-5 h-5" />
        </button>
      </div>

      {/* Search Panel */}
      {showSearch && (
        <div className="absolute top-16 right-4 z-30 w-80">
          <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700">
            <h3 className="text-white text-sm font-semibold mb-3 flex items-center">
              <Search className="w-4 h-4 mr-2" />
              Search Locations
            </h3>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search Mars locations..."
              className="w-full p-2 bg-gray-700 text-white border border-gray-600 rounded focus:border-blue-500 focus:outline-none"
            />
            {searchResults.length > 0 && (
              <div className="mt-3 max-h-60 overflow-y-auto">
                {searchResults.map((location) => (
                  <button
                    key={location.name}
                    onClick={() => setSelectedLocation(location)}
                    className="w-full text-left p-2 text-white hover:bg-gray-700 cursor-pointer rounded transition-colors"
                  >
                    <div className="font-medium">{location.name}</div>
                    <div className="text-xs text-gray-400">
                      {formatCoordinates(location.lat, location.lon)} • {location.type}
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
          <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700">
            <h3 className="text-white text-sm font-semibold mb-3 flex items-center">
              <Bookmark className="w-4 h-4 mr-2" />
              Bookmarks
            </h3>
            {selectedLocation && (
              <button
                onClick={addBookmark}
                className="w-full p-2 bg-green-600 hover:bg-green-700 text-white rounded mb-3 transition-colors"
              >
                Add Current Location
              </button>
            )}
            <div className="max-h-60 overflow-y-auto">
              {bookmarks.map((bookmark) => (
                <div key={bookmark.id} className="p-2 text-white hover:bg-gray-700 cursor-pointer rounded transition-colors">
                  <div className="font-medium">{bookmark.name}</div>
                  <div className="text-xs text-gray-400">
                    {formatCoordinates(bookmark.lat, bookmark.lon)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Location Info Panel */}
      {selectedLocation && (
        <div className="absolute bottom-4 left-4 z-30 w-80">
          <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700">
            <h3 className="text-white text-lg font-semibold mb-2 flex items-center">
              <MapPin className="w-5 h-5 mr-2" />
              {selectedLocation.name}
            </h3>
            <div className="text-gray-300 space-y-1">
              <div>Type: {selectedLocation.type}</div>
              <div>Coordinates: {formatCoordinates(selectedLocation.lat, selectedLocation.lon)}</div>
              <div>Elevation: {selectedLocation.elevation}m</div>
              {selectedLocation.description && (
                <div className="mt-2 text-sm">{selectedLocation.description}</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Status Bar */}
      <div className="absolute bottom-4 right-4 z-30">
        <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-3 border border-gray-700">
          <div className="text-white text-sm flex items-center space-x-4">
            <div className="flex items-center">
              <Activity className="w-4 h-4 mr-1" />
              Zoom: {viewState.zoom.toFixed(1)}x
            </div>
            <div>
              Center: {formatCoordinates(viewState.centerLat, viewState.centerLon)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CleanMarsMapper;
