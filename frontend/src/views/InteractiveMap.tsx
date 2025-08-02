import { Layers, Loader } from 'lucide-react';
import Map from 'ol/Map';
import View from 'ol/View';
import { defaults as defaultControls, FullScreen, ScaleLine } from 'ol/control';
import VectorLayer from 'ol/layer/Vector';
import 'ol/ol.css';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import MapControls from '../components/MapControls';
import {
    createLandingSitesLayer,
    createMarsBaseLayer,
    createMarsElevationLayer,
    createMarsInfraredLayer,
    MarsLayer
} from '../services/marsLayers';

// Exported component
export const InteractiveMap: React.FC = () => {
  const mapElement = useRef<HTMLDivElement>(null);
  const map = useRef<Map | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingTiles, setLoadingTiles] = useState(0);
  const [activeTool, setActiveTool] = useState<string>('');
  const loadingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const updateLoadingState = useCallback((tilesLoading: number) => {
    if (loadingTimeoutRef.current) {
      clearTimeout(loadingTimeoutRef.current);
    }

    if (tilesLoading > 0) {
      setIsLoading(true);
    } else {
      loadingTimeoutRef.current = setTimeout(() => {
        setIsLoading(false);
      }, 300);
    }
  }, []);

  useEffect(() => {
    if (!mapElement.current) return;

    const baseLayer = createMarsBaseLayer();
    const elevationLayer = createMarsElevationLayer();
    const infraredLayer = createMarsInfraredLayer();
    const landingSitesLayer = createLandingSitesLayer();

    [baseLayer, elevationLayer, infraredLayer].forEach(layer => {
      layer.getSource()?.on('tileloadstart', () => {
        setLoadingTiles(prev => {
          const newCount = prev + 1;
          updateLoadingState(newCount);
          return newCount;
        });
      });

      layer.getSource()?.on('tileloadend', () => {
        setLoadingTiles(prev => {
          const newCount = Math.max(0, prev - 1);
          updateLoadingState(newCount);
          return newCount;
        });
      });

      layer.getSource()?.on('tileloaderror', () => {
        setLoadingTiles(prev => {
          const newCount = Math.max(0, prev - 1);
          updateLoadingState(newCount);
          return newCount;
        });
      });
    });

    map.current = new Map({
      target: mapElement.current,
      controls: defaultControls({
        zoom: false,
        rotate: false,
        attribution: false
      }).extend([
        new ScaleLine(),
        new FullScreen(),
      ]),
      layers: [baseLayer, elevationLayer, infraredLayer, landingSitesLayer],
      view: new View({
        center: [0, 0],
        zoom: 1,
        maxZoom: 8,
        minZoom: 0,
        constrainResolution: true,
        projection: 'EPSG:3857'
      }),
    });

    map.current.getView().setProperties({
      enableRotation: false,
      constrainResolution: true,
      smoothResolutionConstraint: true,
      smoothExtentConstraint: true
    });

    return () => {
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current);
      }
      if (map.current) {
        map.current.setTarget(undefined);
      }
    };
  }, [updateLoadingState]);

  const handleLayerToggle = (layerType: string, enabled: boolean) => {
    if (!map.current) return;

    const layers = map.current.getLayers();
    layers.forEach((layer) => {
      if (layerType === 'landing-sites' && layer instanceof VectorLayer) {
        layer.setVisible(enabled);
      } else {
        const marsLayer = layer as MarsLayer;
        if (marsLayer.type === layerType) {
          marsLayer.setVisible(enabled);
        }
      }
    });
  };

  const handleToolSelect = (tool: string) => {
    setActiveTool(tool);
    if (!map.current) return;

    const view = map.current.getView();
    const currentZoom = view.getZoom();

    if (!currentZoom) return;

    switch (tool) {
      case 'zoom-in':
        if (currentZoom < 9) {
          view.animate({
            zoom: currentZoom + 1,
            duration: 300,
            easing: (t) => t * t * (3 - 2 * t)
          });
        }
        break;
      case 'zoom-out':
        if (currentZoom > 0) {
          view.animate({
            zoom: currentZoom - 1,
            duration: 300,
            easing: (t) => t * t * (3 - 2 * t)
          });
        }
        break;
    }
  };

  const handleExport = () => {
    if (!map.current) return;

    const mapCanvas = mapElement.current?.querySelector('canvas');
    if (!mapCanvas) return;

    const link = document.createElement('a');
    link.download = 'mars-map.png';
    link.href = (mapCanvas as HTMLCanvasElement).toDataURL();
    link.click();
  };

  return (
    <div className="relative h-full">
      <div className="absolute inset-0">
        {/* Map Controls - Top Center Horizontal */}
        <MapControls onToolSelect={handleToolSelect} onExport={handleExport} />

        {/* Main Map Container */}
        <div
          ref={mapElement}
          className="w-full h-full"
        />

        {/* Mars Data Layers Panel - Right Side */}
        <div className="absolute top-4 right-4 bg-black bg-opacity-80 backdrop-blur-sm rounded-lg p-4 min-w-64 max-w-80 z-10">
          <h3 className="text-white text-lg font-semibold mb-3 flex items-center">
            <Layers className="mr-2" size={20} />
            Mars Data Layers
          </h3>
          <div className="space-y-3">
            <label className="flex items-center justify-between">
              <span className="text-white text-sm">Base Surface</span>
              <input
                type="checkbox"
                className="form-checkbox h-4 w-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-blue-500"
                defaultChecked
                onChange={(e) => handleLayerToggle('base', e.target.checked)}
              />
            </label>
            <label className="flex items-center justify-between">
              <span className="text-white text-sm">Elevation (MOLA)</span>
              <input
                type="checkbox"
                className="form-checkbox h-4 w-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-blue-500"
                onChange={(e) => handleLayerToggle('elevation', e.target.checked)}
              />
            </label>
            <label className="flex items-center justify-between">
              <span className="text-white text-sm">Infrared Data</span>
              <input
                type="checkbox"
                className="form-checkbox h-4 w-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-blue-500"
                onChange={(e) => handleLayerToggle('infrared', e.target.checked)}
              />
            </label>
            <label className="flex items-center justify-between">
              <span className="text-white text-sm">Landing Sites</span>
              <input
                type="checkbox"
                className="form-checkbox h-4 w-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-blue-500"
                onChange={(e) => handleLayerToggle('landing-sites', e.target.checked)}
              />
            </label>
          </div>
        </div>

        {/* Loading Indicator */}
        {isLoading && (
          <div className="absolute bottom-4 right-4 bg-black bg-opacity-80 backdrop-blur-sm px-4 py-2 rounded-lg flex items-center gap-2 transition-opacity duration-200">
            <Loader className="animate-spin text-white" size={20} />
            <span className="text-sm text-white">Loading tiles...</span>
          </div>
        )}
      </div>
    </div>
  );
};

// Default export
export default InteractiveMap;
