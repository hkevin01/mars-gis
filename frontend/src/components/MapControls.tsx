import { Download, Layers, Maximize, Move, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import React from 'react';

// Properly exported interface
export interface MapControlsProps {
  onToolSelect: (tool: string) => void;
  onExport: () => void;
}

// Exported as a named export
export const MapControls: React.FC<MapControlsProps> = ({ onToolSelect, onExport }) => {
  return (
    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-80 backdrop-blur-sm rounded-lg px-6 py-3 z-10">
      <div className="flex items-center space-x-4">
        {/* Zoom In Button */}
        <button
          onClick={() => onToolSelect('zoom-in')}
          className="text-white hover:text-blue-400 transition-colors p-2 hover:bg-white hover:bg-opacity-10 rounded"
          title="Zoom In"
        >
          <ZoomIn size={20} />
        </button>

        {/* Zoom Out Button */}
        <button
          onClick={() => onToolSelect('zoom-out')}
          className="text-white hover:text-blue-400 transition-colors p-2 hover:bg-white hover:bg-opacity-10 rounded"
          title="Zoom Out"
        >
          <ZoomOut size={20} />
        </button>

        {/* Separator */}
        <div className="h-6 w-px bg-gray-600"></div>

        {/* Pan Tool */}
        <button
          onClick={() => onToolSelect('pan')}
          className="text-white hover:text-blue-400 transition-colors p-2 hover:bg-white hover:bg-opacity-10 rounded"
          title="Pan Tool"
        >
          <Move size={20} />
        </button>

        {/* Reset View Button */}
        <button
          onClick={() => onToolSelect('reset-view')}
          className="text-white hover:text-blue-400 transition-colors p-2 hover:bg-white hover:bg-opacity-10 rounded"
          title="Reset View"
        >
          <RotateCcw size={20} />
        </button>

        {/* Separator */}
        <div className="h-6 w-px bg-gray-600"></div>

        {/* Toggle Layers */}
        <button
          onClick={() => onToolSelect('layers')}
          className="text-white hover:text-blue-400 transition-colors p-2 hover:bg-white hover:bg-opacity-10 rounded"
          title="Toggle Layers"
        >
          <Layers size={20} />
        </button>

        {/* Fullscreen Toggle */}
        <button
          onClick={() => onToolSelect('fullscreen')}
          className="text-white hover:text-blue-400 transition-colors p-2 hover:bg-white hover:bg-opacity-10 rounded"
          title="Toggle Fullscreen"
        >
          <Maximize size={20} />
        </button>

        {/* Separator */}
        <div className="h-6 w-px bg-gray-600"></div>

        {/* Export Button */}
        <button
          onClick={onExport}
          className="text-white hover:text-blue-400 transition-colors p-2 hover:bg-white hover:bg-opacity-10 rounded"
          title="Export Map"
        >
          <Download size={20} />
        </button>
      </div>
    </div>
  );
};

// Default export
export default MapControls;
