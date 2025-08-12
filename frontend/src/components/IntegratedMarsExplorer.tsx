// Integrated Mars Explorer - Seamless 2D/3D Navigation
import { ChevronLeft, ChevronRight, Eye, Globe, Map, Maximize2, Navigation } from 'lucide-react';
import React, { useCallback, useState } from 'react';
import OpenLayersMarsMapper from '../features/mars-mapping/components/OpenLayersMarsMapper';
import Mars3DViewer from '../views/Mars3DViewer';

type ViewModeId = '2d' | '3d' | 'split';

interface ViewMode {
  id: ViewModeId;
  name: string;
  description: string;
}

const VIEW_MODES: ViewMode[] = [
  {
    id: '2d',
    name: '2D Map View',
    description: 'Traditional OpenLayers Mars mapping interface'
  },
  {
    id: '3d',
    name: '3D Globe View',
    description: 'Immersive 3D Mars globe and terrain visualization'
  },
  {
    id: 'split',
    name: 'Split View',
    description: 'Side-by-side 2D mapping and 3D visualization'
  }
];

interface IntegratedMarsExplorerProps {
  className?: string;
  initialView?: ViewModeId;
}

const IntegratedMarsExplorer: React.FC<IntegratedMarsExplorerProps> = ({
  className = '',
  initialView = '2d'
}) => {
  const [currentView, setCurrentView] = useState<ViewModeId>(initialView);
  const [showModeSelector, setShowModeSelector] = useState(false);
  const [syncViews, setSyncViews] = useState(true);

  // Handle location selection from either view
  // Note: 3D Cesium view does not yet emit selection events.
  // Future: wire Cesium ScreenSpaceEventHandler to update selectedLocation.

  // Switch between view modes
  const switchViewMode = useCallback((mode: ViewModeId) => {
    setCurrentView(mode);
    setShowModeSelector(false);
  }, []);

  const getCurrentModeInfo = () => {
    return VIEW_MODES.find(mode => mode.id === currentView) || VIEW_MODES[0];
  };

  const renderViewContent = () => {
    switch (currentView) {
      case '2d':
        return (
          <div className="w-full h-full">
            <OpenLayersMarsMapper />
          </div>
        );

      case '3d':
        return (
          <div className="w-full h-full">
            <Mars3DViewer />
          </div>
        );

      case 'split':
        return (
          <div className="w-full h-full flex">
            {/* 2D View - Left Side */}
            <div className="w-1/2 h-full border-r-2 border-gray-600">
              <div className="relative w-full h-full">
                <OpenLayersMarsMapper />
                <div className="absolute top-4 left-4 bg-gray-900/90 backdrop-blur-sm rounded-lg px-3 py-1">
                  <div className="text-white text-sm font-medium flex items-center">
                    <Map className="w-4 h-4 mr-2 text-blue-400" />
                    2D Mars Map
                  </div>
                </div>
              </div>
            </div>

            {/* 3D View - Right Side */}
            <div className="w-1/2 h-full">
              <div className="relative w-full h-full">
                <Mars3DViewer />
                <div className="absolute top-4 left-4 bg-gray-900/90 backdrop-blur-sm rounded-lg px-3 py-1">
                  <div className="text-white text-sm font-medium flex items-center">
                    <Globe className="w-4 h-4 mr-2 text-orange-400" />
                    3D Mars Globe
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      default:
        return <div>Invalid view mode</div>;
    }
  };

  return (
    <div className={`relative w-full h-screen bg-black overflow-hidden ${className}`}>
      {/* Main View Content */}
      <div className="w-full h-full">
        {renderViewContent()}
      </div>

      {/* View Mode Selector */}
      <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-50">
        <div className="bg-gray-900/95 backdrop-blur-sm rounded-lg border border-gray-700 shadow-lg">
          {/* Current Mode Display */}
          <button
            onClick={() => setShowModeSelector(!showModeSelector)}
            className="flex items-center justify-between p-3 w-full text-left hover:bg-gray-800/50 rounded-lg transition-colors min-w-64"
          >
            <div className="flex items-center">
              {currentView === '2d' && <Map className="w-5 h-5 mr-3 text-blue-400" />}
              {currentView === '3d' && <Globe className="w-5 h-5 mr-3 text-orange-400" />}
              {currentView === 'split' && <Eye className="w-5 h-5 mr-3 text-purple-400" />}
              <div>
                <div className="text-white text-sm font-medium">{getCurrentModeInfo().name}</div>
                <div className="text-gray-400 text-xs">{getCurrentModeInfo().description}</div>
              </div>
            </div>
            {showModeSelector ? (
              <ChevronLeft className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-gray-400" />
            )}
          </button>

          {/* Mode Options */}
          {showModeSelector && (
            <div className="border-t border-gray-700 p-2 space-y-1">
              {VIEW_MODES.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => switchViewMode(mode.id)}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    currentView === mode.id
                      ? 'bg-blue-600/20 border border-blue-500/30'
                      : 'hover:bg-gray-700/50'
                  }`}
                >
                  <div className="flex items-center">
                    {mode.id === '2d' && <Map className="w-4 h-4 mr-3 text-blue-400" />}
                    {mode.id === '3d' && <Globe className="w-4 h-4 mr-3 text-orange-400" />}
                    {mode.id === 'split' && <Eye className="w-4 h-4 mr-3 text-purple-400" />}
                    <div>
                      <div className="text-white text-sm font-medium">{mode.name}</div>
                      <div className="text-gray-400 text-xs">{mode.description}</div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

            {/* Main Controls */}
      <div className="absolute top-4 right-[316px] z-50 space-y-2">
        {/* Sync Toggle (for split view) */}
        {currentView === 'split' && (
          <button
            onClick={() => setSyncViews(!syncViews)}
            className={`p-2 ${
              syncViews
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : 'bg-gray-600 hover:bg-gray-700 text-white'
            }`}
            title="Sync Views"
          >
            <Navigation className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Fullscreen Button - Fixed in top right */}
      <div className="absolute top-4 right-4 z-50">
        <button
          onClick={() => {
            if (!document.fullscreenElement) {
              document.documentElement.requestFullscreen();
            } else {
              document.exitFullscreen();
            }
          }}
          className="p-2 bg-gray-900/90 backdrop-blur-sm text-white rounded-lg hover:bg-gray-800 transition-colors"
          title="Toggle Fullscreen"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

  {/* Selected Location Info - future enhancement when 3D emits selection */}

      {/* Instructions */}
      <div className="absolute bottom-4 left-4 z-50">
        <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 border border-gray-700">
          <div className="text-white text-xs space-y-1">
            <div className="font-medium mb-2">Mars-GIS v3.0 - Integrated Explorer</div>
            {currentView === '2d' && (
              <>
                <div>• Click locations to explore</div>
                <div>• Use layer controls for NASA data</div>
                <div>• Switch to 3D for immersive view</div>
              </>
            )}
            {currentView === '3d' && (
              <>
                <div>• Drag to rotate Mars globe</div>
                <div>• Click surface to analyze terrain</div>
                <div>• Switch views at top center</div>
              </>
            )}
            {currentView === 'split' && (
              <>
                <div>• Use both 2D mapping and 3D visualization</div>
                <div>• Toggle sync to coordinate views</div>
                <div>• Click any location to explore</div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Mars-GIS Branding */}
      <div className="absolute top-4 left-[216px] z-50">
        <div className="bg-gradient-to-r from-red-900/80 to-orange-900/80 backdrop-blur-sm rounded-lg px-4 py-2 border border-red-700/50">
          <div className="text-white font-bold text-lg">Mars-GIS</div>
          <div className="text-orange-300 text-xs">Phase 3.0 - Enhanced Exploration</div>
        </div>
      </div>
    </div>
  );
};

export default IntegratedMarsExplorer;
