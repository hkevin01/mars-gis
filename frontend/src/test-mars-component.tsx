import React from 'react';
import OpenLayersMarsMapper from './features/mars-mapping/components/OpenLayersMarsMapper';

// Simple test component to verify Mars mapper functionality
const TestMarsComponent: React.FC = () => {
  return (
    <div className="w-full h-screen">
      <h1 className="text-white text-xl p-4 absolute top-0 left-0 z-50 bg-black/70 rounded">
        Mars GIS Test - Smooth Loading & Mars Background
      </h1>
      <OpenLayersMarsMapper />
    </div>
  );
};

export default TestMarsComponent;
