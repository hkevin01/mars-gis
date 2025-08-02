// Enhanced 3D Mars Visualization Interface
import { Globe, Maximize2, Mountain, Settings } from 'lucide-react';
import React, { useCallback, useState } from 'react';
import Mars3DGlobe from './Mars3DGlobe';
import Mars3DTerrain from './Mars3DTerrain';

interface TerrainDataPoint {
  lat: number;
  lon: number;
  elevation: number;
}

interface SelectedRegion {
  bounds: [number, number, number, number];
  name: string;
}

interface Enhanced3DInterfaceProps {
  onLocationSelect?: (lat: number, lon: number) => void;
  initialView?: 'globe' | 'terrain';
  className?: string;
}

const Enhanced3DInterface: React.FC<Enhanced3DInterfaceProps> = ({
  onLocationSelect,
  initialView = 'globe',
  className = ''
}) => {
  const [currentView, setCurrentView] = useState<'globe' | 'terrain'>(initialView);
  const [selectedRegion, setSelectedRegion] = useState<SelectedRegion | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [elevationExaggeration, setElevationExaggeration] = useState(10);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [viewerOptions, setViewerOptions] = useState({
    showAtmosphere: true,
    showLighting: true,
    showLabels: false,
    autoRotate: false
  });

  // Predefined Mars regions of interest
  const marsRegions: SelectedRegion[] = [
    {
      bounds: [-15, -5, -10, 5],
      name: 'Valles Marineris'
    },
    {
      bounds: [20, -45, 25, -35],
      name: 'Olympus Mons'
    },
    {
      bounds: [-25, 15, -20, 25],
      name: 'Gale Crater'
    },
    {
      bounds: [-5, 175, 0, -175],
      name: 'Meridiani Planum'
    },
    {
      bounds: [10, -15, 15, -5],
      name: 'Chryse Planitia'
    }
  ];

  // Handle terrain point selection
  const handleTerrainClick = useCallback((point: TerrainDataPoint) => {
    if (onLocationSelect) {
      onLocationSelect(point.lat, point.lon);
    }
  }, [onLocationSelect]);

  // Handle Mars globe click to select region
  const handleGlobeClick = useCallback((lat: number, lon: number) => {
    // Create a region around the clicked point
    const regionSize = 5; // degrees
    const newRegion: SelectedRegion = {
      bounds: [
        lat - regionSize/2,
        lon - regionSize/2,
        lat + regionSize/2,
        lon + regionSize/2
      ],
      name: `Custom Region (${lat.toFixed(2)}°, ${lon.toFixed(2)}°)`
    };

    setSelectedRegion(newRegion);
    setCurrentView('terrain');

    if (onLocationSelect) {
      onLocationSelect(lat, lon);
    }
  }, [onLocationSelect]);

  // Toggle between views
  const toggleView = useCallback(() => {
    setCurrentView(prev => prev === 'globe' ? 'terrain' : 'globe');
  }, []);

  // Update viewer options
  const updateViewerOption = useCallback((option: keyof typeof viewerOptions, value: boolean) => {
    setViewerOptions(prev => ({ ...prev, [option]: value }));
  }, []);

  // Toggle fullscreen mode
  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => !prev);
  }, []);

  const containerClasses = `
    ${className}
    ${isFullscreen ? 'fixed inset-0 z-50 bg-black' : 'relative'}
    transition-all duration-300
  `;

  const viewerSize = isFullscreen
    ? { width: window.innerWidth, height: window.innerHeight }
    : { width: 800, height: 600 };

  return (
    <div className={containerClasses}>
      {/* Main 3D Viewer Area */}
      <div className="relative">
        {currentView === 'globe' ? (
          <Mars3DGlobe
            width={viewerSize.width}
            height={viewerSize.height}
            showAtmosphere={viewerOptions.showAtmosphere}
            autoRotate={viewerOptions.autoRotate}
            onLocationClick={handleGlobeClick}
          />
        ) : (
          <Mars3DTerrain
            width={viewerSize.width}
            height={viewerSize.height}
            selectedRegion={selectedRegion || undefined}
            onTerrainClick={handleTerrainClick}
            elevationExaggeration={elevationExaggeration}
          />
        )}

        {/* Main Control Panel */}
        <div className="absolute top-4 left-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-4 space-y-4">
          {/* View Switcher */}
          <div className="flex items-center space-x-2">
            <button
              onClick={toggleView}
              className={`p-2 rounded-lg transition-colors ${
                currentView === 'globe'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
              title="Switch to Globe View"
            >
              <Globe className="w-5 h-5" />
            </button>
            <button
              onClick={toggleView}
              className={`p-2 rounded-lg transition-colors ${
                currentView === 'terrain'
                  ? 'bg-orange-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
              title="Switch to Terrain View"
            >
              <Mountain className="w-5 h-5" />
            </button>
          </div>

          {/* Current View Info */}
          <div className="text-white">
            <div className="text-sm font-medium">
              {currentView === 'globe' ? '3D Mars Globe' : '3D Mars Terrain'}
            </div>
            <div className="text-xs text-gray-300">
              {currentView === 'globe'
                ? 'Click surface to explore terrain'
                : selectedRegion?.name || 'No region selected'
              }
            </div>
          </div>

          {/* Region Quick Select (for terrain view) */}
          {currentView === 'terrain' && (
            <div className="space-y-2">
              <div className="text-xs text-gray-300 font-medium">Quick Regions</div>
              <div className="grid grid-cols-1 gap-1">
                {marsRegions.map((region) => (
                  <button
                    key={region.name}
                    onClick={() => setSelectedRegion(region)}
                    className={`text-xs p-2 rounded text-left transition-colors ${
                      selectedRegion?.name === region.name
                        ? 'bg-orange-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    {region.name}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Elevation Controls (terrain view) */}
          {currentView === 'terrain' && (
            <div className="space-y-2">
              <div className="text-xs text-gray-300 font-medium">Elevation Scale</div>
              <input
                type="range"
                min="1"
                max="50"
                value={elevationExaggeration}
                onChange={(e) => setElevationExaggeration(Number(e.target.value))}
                className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              />
              <div className="text-xs text-gray-400">{elevationExaggeration}x exaggeration</div>
            </div>
          )}
        </div>

        {/* Settings Panel */}
        <div className="absolute top-4 right-4 space-y-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 bg-gray-900/90 backdrop-blur-sm text-white rounded-lg hover:bg-gray-800 transition-colors"
            title="View Settings"
          >
            <Settings className="w-5 h-5" />
          </button>

          <button
            onClick={toggleFullscreen}
            className="p-2 bg-gray-900/90 backdrop-blur-sm text-white rounded-lg hover:bg-gray-800 transition-colors"
            title="Toggle Fullscreen"
          >
            <Maximize2 className="w-5 h-5" />
          </button>
        </div>

        {/* Advanced Settings */}
        {showSettings && (
          <div className="absolute top-16 right-4 bg-gray-900/95 backdrop-blur-sm rounded-lg p-4 space-y-3 min-w-64">
            <div className="text-white font-medium text-sm border-b border-gray-700 pb-2">
              Visualization Settings
            </div>

            {/* Globe Settings */}
            {currentView === 'globe' && (
              <div className="space-y-2">
                <label className="flex items-center text-sm text-gray-300">
                  <input
                    type="checkbox"
                    checked={viewerOptions.showAtmosphere}
                    onChange={(e) => updateViewerOption('showAtmosphere', e.target.checked)}
                    className="mr-2 w-4 h-4"
                  />
                  Show Atmosphere
                </label>
                <label className="flex items-center text-sm text-gray-300">
                  <input
                    type="checkbox"
                    checked={viewerOptions.autoRotate}
                    onChange={(e) => updateViewerOption('autoRotate', e.target.checked)}
                    className="mr-2 w-4 h-4"
                  />
                  Auto Rotate
                </label>
              </div>
            )}

            {/* Terrain Settings */}
            {currentView === 'terrain' && (
              <div className="space-y-2">
                <label className="flex items-center text-sm text-gray-300">
                  <input
                    type="checkbox"
                    checked={viewerOptions.showLabels}
                    onChange={(e) => updateViewerOption('showLabels', e.target.checked)}
                    className="mr-2 w-4 h-4"
                  />
                  Show Labels
                </label>
              </div>
            )}
          </div>
        )}

        {/* Instructions */}
        <div className="absolute bottom-4 left-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-3">
          <div className="text-xs text-gray-300 space-y-1">
            {currentView === 'globe' ? (
              <>
                <div>• Drag to rotate Mars globe</div>
                <div>• Scroll to zoom in/out</div>
                <div>• Click surface to explore terrain</div>
              </>
            ) : (
              <>
                <div>• Drag to rotate terrain view</div>
                <div>• Scroll to zoom elevation</div>
                <div>• Click terrain for details</div>
              </>
            )}
          </div>
        </div>

        {/* Performance Info */}
        <div className="absolute bottom-4 right-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-2">
          <div className="text-xs text-gray-400">
            3D Visualization • Mars-GIS v3.0
          </div>
        </div>
      </div>

      {/* Exit Fullscreen Overlay */}
      {isFullscreen && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-gray-900/90 backdrop-blur-sm rounded-lg px-4 py-2">
          <div className="text-white text-sm">Press ESC or click X to exit fullscreen</div>
        </div>
      )}
    </div>
  );
};

export default Enhanced3DInterface;
