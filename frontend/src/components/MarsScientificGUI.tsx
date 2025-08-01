import {
    Activity,
    Brain,
    Calculator,
    Compass,
    Cpu,
    Database,
    Download,
    Eye,
    Filter,
    Globe,
    Layers,
    Mountain,
    RefreshCw,
    RotateCcw,
    Satellite,
    Search,
    Settings,
    Target,
    Thermometer,
    Timer,
    Upload,
    Wind,
    Zap
} from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';

// Types for Mars Scientific Data
interface MarsCoordinate {
  latitude: number;
  longitude: number;
  elevation?: number;
}

interface LandingSite {
  id: string;
  name: string;
  coordinates: MarsCoordinate;
  terrainType: string;
  hazardLevel: 'low' | 'medium' | 'high';
  scientificValue: number;
  safetyScore: number;
  accessibility: number;
  resources: string[];
  aiRecommendation: boolean;
}

interface MissionTrajectory {
  id: string;
  name: string;
  algorithm: 'A*' | 'RRT' | 'Dijkstra';
  startPoint: MarsCoordinate;
  endPoint: MarsCoordinate;
  waypoints: MarsCoordinate[];
  distance: number;
  duration: number;
  energyCost: number;
  riskLevel: number;
}

interface DataLayer {
  id: string;
  name: string;
  type: 'terrain' | 'atmospheric' | 'thermal' | 'geological' | 'mineral';
  visible: boolean;
  opacity: number;
  resolution: number;
  lastUpdated: string;
  dataSource: string;
}

const MarsScientificGUI: React.FC = () => {
  // State Management
  const [selectedTab, setSelectedTab] = useState<'globe' | 'planning' | 'ai' | 'data'>('globe');
  const [marsGlobeRotation, setMarsGlobeRotation] = useState({ x: 0, y: 0 });
  const [selectedRegion, setSelectedRegion] = useState<MarsCoordinate | null>(null);
  const [activeLayers, setActiveLayers] = useState<DataLayer[]>([]);
  const [landingSites, setLandingSites] = useState<LandingSite[]>([]);
  const [trajectories, setTrajectories] = useState<MissionTrajectory[]>([]);
  const [realTimeData, setRealTimeData] = useState<{
    atmosphericPressure?: number;
    temperature?: number;
    windSpeed?: number;
    dustOpacity?: number;
    solarIrradiance?: number;
    lastUpdate?: string;
  }>({});
  const [isProcessing, setIsProcessing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Initialize sample data
  useEffect(() => {
    // Initialize data layers
    setActiveLayers([
      {
        id: 'elevation',
        name: 'Elevation (MOLA)',
        type: 'terrain',
        visible: true,
        opacity: 0.8,
        resolution: 128,
        lastUpdated: '2025-08-01T10:00:00Z',
        dataSource: 'NASA MOLA'
      },
      {
        id: 'thermal',
        name: 'Thermal Infrared',
        type: 'thermal',
        visible: false,
        opacity: 0.6,
        resolution: 64,
        lastUpdated: '2025-08-01T09:30:00Z',
        dataSource: 'MRO THEMIS'
      },
      {
        id: 'atmospheric',
        name: 'Atmospheric Density',
        type: 'atmospheric',
        visible: false,
        opacity: 0.5,
        resolution: 32,
        lastUpdated: '2025-08-01T08:15:00Z',
        dataSource: 'MRO MCS'
      },
      {
        id: 'geological',
        name: 'Geological Units',
        type: 'geological',
        visible: false,
        opacity: 0.7,
        resolution: 256,
        lastUpdated: '2025-07-31T16:45:00Z',
        dataSource: 'USGS Geological Map'
      }
    ]);

    // Initialize sample landing sites
    setLandingSites([
      {
        id: 'site-001',
        name: 'Jezero Crater Delta',
        coordinates: { latitude: 18.85, longitude: 77.52, elevation: -2540 },
        terrainType: 'Ancient Lake Delta',
        hazardLevel: 'medium',
        scientificValue: 0.95,
        safetyScore: 0.78,
        accessibility: 0.85,
        resources: ['Water ice', 'Clay minerals', 'Organic compounds'],
        aiRecommendation: true
      },
      {
        id: 'site-002',
        name: 'Gale Crater Central Peak',
        coordinates: { latitude: -5.4, longitude: 137.8, elevation: -4500 },
        terrainType: 'Impact Crater',
        hazardLevel: 'low',
        scientificValue: 0.88,
        safetyScore: 0.92,
        accessibility: 0.75,
        resources: ['Sedimentary layers', 'Sulfates', 'Perchlorates'],
        aiRecommendation: true
      },
      {
        id: 'site-003',
        name: 'Valles Marineris Floor',
        coordinates: { latitude: -14.0, longitude: -59.2, elevation: -7000 },
        terrainType: 'Canyon System',
        hazardLevel: 'high',
        scientificValue: 0.92,
        safetyScore: 0.45,
        accessibility: 0.55,
        resources: ['Hydrated minerals', 'Layered deposits', 'Fault systems'],
        aiRecommendation: false
      }
    ]);

    // Initialize sample trajectories
    setTrajectories([
      {
        id: 'traj-001',
        name: 'Optimal Path A*',
        algorithm: 'A*',
        startPoint: { latitude: 18.85, longitude: 77.52 },
        endPoint: { latitude: 18.95, longitude: 77.65 },
        waypoints: [
          { latitude: 18.87, longitude: 77.55 },
          { latitude: 18.91, longitude: 77.60 }
        ],
        distance: 15.2,
        duration: 240,
        energyCost: 0.75,
        riskLevel: 0.25
      }
    ]);

    // Start real-time updates
    const interval = setInterval(() => {
      setRealTimeData(prev => ({
        ...prev,
        atmosphericPressure: 600 + Math.random() * 50,
        temperature: -80 + Math.random() * 10,
        windSpeed: Math.random() * 25,
        dustOpacity: Math.random() * 0.5,
        solarIrradiance: 580 + Math.random() * 40,
        lastUpdate: new Date().toISOString()
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  // Mars Globe Component
  const MarsGlobe3D = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
      if (!canvasRef.current) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Simple Mars globe representation
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const radius = Math.min(centerX, centerY) - 20;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw Mars sphere
      const gradient = ctx.createRadialGradient(
        centerX - radius/3, centerY - radius/3, 0,
        centerX, centerY, radius
      );
      gradient.addColorStop(0, '#ff6b47');
      gradient.addColorStop(0.7, '#cc4125');
      gradient.addColorStop(1, '#8b2e0f');

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fill();

      // Draw surface features
      ctx.strokeStyle = '#a0522d';
      ctx.lineWidth = 2;

      // Simplified topographic lines
      for (let i = 0; i < 8; i++) {
        const angle = (i * Math.PI * 2) / 8 + marsGlobeRotation.y;
        const x1 = centerX + Math.cos(angle) * (radius * 0.3);
        const y1 = centerY + Math.sin(angle) * (radius * 0.3);
        const x2 = centerX + Math.cos(angle) * (radius * 0.8);
        const y2 = centerY + Math.sin(angle) * (radius * 0.8);

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }

      // Draw landing sites
      landingSites.forEach(site => {
        const siteAngle = (site.coordinates.longitude * Math.PI) / 180 + marsGlobeRotation.y;
        const siteLat = (site.coordinates.latitude * Math.PI) / 180;

        const x = centerX + Math.cos(siteAngle) * Math.cos(siteLat) * radius * 0.9;
        const y = centerY + Math.sin(siteLat) * radius * 0.9;

        // Site marker
        ctx.fillStyle = site.aiRecommendation ? '#00ff00' : '#ffff00';
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();

        // Site border
        ctx.strokeStyle = site.hazardLevel === 'high' ? '#ff0000' :
                         site.hazardLevel === 'medium' ? '#ffaa00' : '#00aa00';
        ctx.lineWidth = 2;
        ctx.stroke();
      });

      // Draw selected region
      if (selectedRegion) {
        const regionAngle = (selectedRegion.longitude * Math.PI) / 180 + marsGlobeRotation.y;
        const regionLat = (selectedRegion.latitude * Math.PI) / 180;

        const x = centerX + Math.cos(regionAngle) * Math.cos(regionLat) * radius * 0.9;
        const y = centerY + Math.sin(regionLat) * radius * 0.9;

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.arc(x, y, 15, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

    }, []);  // Only run once on mount, canvas will be redrawn when needed

    return (
      <div className="relative bg-black rounded-lg overflow-hidden">
        <canvas
          ref={canvasRef}
          width={400}
          height={400}
          className="cursor-pointer"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Convert click to Mars coordinates (simplified)
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const longitude = ((x - centerX) / centerX) * 180;
            const latitude = -((y - centerY) / centerY) * 90;

            setSelectedRegion({ latitude, longitude });
          }}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            if (e.buttons === 1) { // Left mouse button
              const deltaX = (x - rect.width / 2) * 0.01;
              const deltaY = (y - rect.height / 2) * 0.01;

              setMarsGlobeRotation(prev => ({
                x: prev.x + deltaY,
                y: prev.y + deltaX
              }));
            }
          }}
        />

        {/* Globe Controls */}
        <div className="absolute top-4 right-4 space-y-2">
          <button
            onClick={() => setMarsGlobeRotation({ x: 0, y: 0 })}
            className="p-2 bg-black bg-opacity-50 text-white rounded hover:bg-opacity-70"
            title="Reset View"
          >
            <RotateCcw size={16} />
          </button>
        </div>

        {/* Region Info Overlay */}
        {selectedRegion && (
          <div className="absolute bottom-4 left-4 bg-black bg-opacity-80 text-white p-3 rounded max-w-xs">
            <h4 className="font-semibold mb-1">Selected Region</h4>
            <p className="text-sm">
              Lat: {selectedRegion.latitude.toFixed(2)}°<br/>
              Lon: {selectedRegion.longitude.toFixed(2)}°
            </p>
            {selectedRegion.elevation && (
              <p className="text-sm">Elevation: {selectedRegion.elevation}m</p>
            )}
          </div>
        )}
      </div>
    );
  };

  // Layer Control Panel
  const LayerControlPanel = () => (
    <div className="bg-white rounded-lg shadow-lg p-4">
      <h3 className="text-lg font-semibold mb-3 flex items-center">
        <Layers className="mr-2" size={20} />
        Data Layers
      </h3>

      <div className="space-y-3">
        {activeLayers.map(layer => (
          <div key={layer.id} className="border rounded p-3">
            <div className="flex items-center justify-between mb-2">
              <label className="flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={layer.visible}
                  onChange={(e) => {
                    setActiveLayers(prev => prev.map(l =>
                      l.id === layer.id ? { ...l, visible: e.target.checked } : l
                    ));
                  }}
                  className="mr-2"
                />
                <span className="font-medium">{layer.name}</span>
              </label>
              <span className={`px-2 py-1 rounded text-xs ${
                layer.type === 'terrain' ? 'bg-brown-100 text-brown-800' :
                layer.type === 'thermal' ? 'bg-red-100 text-red-800' :
                layer.type === 'atmospheric' ? 'bg-blue-100 text-blue-800' :
                'bg-gray-100 text-gray-800'
              }`}>
                {layer.type}
              </span>
            </div>

            <div className="space-y-2">
              <div>
                <label className="block text-sm text-gray-600 mb-1">
                  Opacity: {Math.round(layer.opacity * 100)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={layer.opacity}
                  onChange={(e) => {
                    setActiveLayers(prev => prev.map(l =>
                      l.id === layer.id ? { ...l, opacity: parseFloat(e.target.value) } : l
                    ));
                  }}
                  className="w-full"
                />
              </div>

              <div className="flex justify-between text-xs text-gray-500">
                <span>Source: {layer.dataSource}</span>
                <span>Res: {layer.resolution}px</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  // Landing Site Selection Tool
  const LandingSiteSelector = () => (
    <div className="bg-white rounded-lg shadow-lg p-4">
      <h3 className="text-lg font-semibold mb-3 flex items-center">
        <Target className="mr-2" size={20} />
        Landing Site Selection
      </h3>

      <div className="mb-4">
        <button
          onClick={() => {
            // Simulate AI optimization
            setIsProcessing(true);
            setTimeout(() => {
              setIsProcessing(false);
              // Add new optimized sites
            }, 2000);
          }}
          disabled={isProcessing}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center"
        >
          {isProcessing ? (
            <>
              <RefreshCw className="animate-spin mr-2" size={16} />
              AI Optimizing...
            </>
          ) : (
            <>
              <Brain className="mr-2" size={16} />
              AI-Powered Optimization
            </>
          )}
        </button>
      </div>

      <div className="space-y-3 max-h-96 overflow-y-auto">
        {landingSites.map(site => (
          <div
            key={site.id}
            className={`border rounded p-3 cursor-pointer transition-colors ${
              selectedRegion?.latitude === site.coordinates.latitude ?
              'border-blue-500 bg-blue-50' : 'hover:bg-gray-50'
            }`}
            onClick={() => setSelectedRegion(site.coordinates)}
          >
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium">{site.name}</h4>
              <div className="flex items-center space-x-1">
                {site.aiRecommendation && (
                <span title="AI Recommended">
                  <Brain className="text-blue-500" size={16} />
                </span>
                )}
                <span className={`w-3 h-3 rounded-full ${
                  site.hazardLevel === 'low' ? 'bg-green-500' :
                  site.hazardLevel === 'medium' ? 'bg-yellow-500' : 'bg-red-500'
                }`} title={`${site.hazardLevel} hazard`} />
              </div>
            </div>

            <div className="text-sm text-gray-600 mb-2">
              {site.coordinates.latitude.toFixed(2)}°, {site.coordinates.longitude.toFixed(2)}°
              <br />
              {site.terrainType}
            </div>

            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Science</span>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full"
                    style={{ width: `${site.scientificValue * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <span className="text-gray-500">Safety</span>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{ width: `${site.safetyScore * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <span className="text-gray-500">Access</span>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-purple-500 h-2 rounded-full"
                    style={{ width: `${site.accessibility * 100}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="mt-2">
              <div className="text-xs text-gray-500 mb-1">Available Resources:</div>
              <div className="flex flex-wrap gap-1">
                {site.resources.map(resource => (
                  <span key={resource} className="px-2 py-1 bg-gray-100 text-xs rounded">
                    {resource}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  // Mission Planning Dashboard
  const MissionPlanningPanel = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trajectory Planning */}
        <div className="bg-white rounded-lg shadow-lg p-4">
          <h3 className="text-lg font-semibold mb-3 flex items-center">
            <Compass className="mr-2" size={20} />
            Trajectory Planning
          </h3>

          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-2">
              <button
                className="p-2 border rounded text-center hover:bg-gray-50"
                onClick={() => {/* Generate A* path */}}
              >
                A* Algorithm
              </button>
              <button
                className="p-2 border rounded text-center hover:bg-gray-50"
                onClick={() => {/* Generate RRT path */}}
              >
                RRT Algorithm
              </button>
              <button
                className="p-2 border rounded text-center hover:bg-gray-50"
                onClick={() => {/* Generate Dijkstra path */}}
              >
                Dijkstra
              </button>
            </div>

            {trajectories.map(traj => (
              <div key={traj.id} className="border rounded p-3">
                <div className="flex justify-between items-center mb-2">
                  <h4 className="font-medium">{traj.name}</h4>
                  <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                    {traj.algorithm}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Distance:</span> {traj.distance} km
                  </div>
                  <div>
                    <span className="text-gray-600">Duration:</span> {traj.duration} min
                  </div>
                  <div>
                    <span className="text-gray-600">Energy:</span> {traj.energyCost} kWh
                  </div>
                  <div>
                    <span className="text-gray-600">Risk:</span>
                    <span className={`ml-1 ${
                      traj.riskLevel < 0.3 ? 'text-green-600' :
                      traj.riskLevel < 0.7 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {Math.round(traj.riskLevel * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Resource Optimization */}
        <div className="bg-white rounded-lg shadow-lg p-4">
          <h3 className="text-lg font-semibold mb-3 flex items-center">
            <Calculator className="mr-2" size={20} />
            Resource Optimization
          </h3>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-blue-50 rounded">
                <div className="text-sm text-gray-600">Power Consumption</div>
                <div className="text-xl font-semibold">2.4 kW</div>
                <div className="text-xs text-green-600">↓ 15% optimized</div>
              </div>
              <div className="p-3 bg-green-50 rounded">
                <div className="text-sm text-gray-600">Fuel Efficiency</div>
                <div className="text-xl font-semibold">89%</div>
                <div className="text-xs text-green-600">↑ 8% improved</div>
              </div>
              <div className="p-3 bg-yellow-50 rounded">
                <div className="text-sm text-gray-600">Communication</div>
                <div className="text-xl font-semibold">94%</div>
                <div className="text-xs text-blue-600">Signal quality</div>
              </div>
              <div className="p-3 bg-purple-50 rounded">
                <div className="text-sm text-gray-600">Mission Duration</div>
                <div className="text-xl font-semibold">45 Sol</div>
                <div className="text-xs text-purple-600">Estimated</div>
              </div>
            </div>

            <div className="border-t pt-4">
              <h4 className="font-medium mb-2">Earth vs Mars Comparison</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Gravity:</span>
                  <span>Earth: 9.8 m/s² | Mars: 3.7 m/s²</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Atmosphere:</span>
                  <span>Mars: 1% of Earth density</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Day Length:</span>
                  <span>Mars: 24h 37m (Sol)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mission Timeline */}
      <div className="bg-white rounded-lg shadow-lg p-4">
        <h3 className="text-lg font-semibold mb-3 flex items-center">
          <Timer className="mr-2" size={20} />
          Mission Timeline
        </h3>

        <div className="relative">
          <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-300"></div>

          <div className="space-y-4 ml-8">
            {[
              { phase: 'Pre-deployment', status: 'completed', time: 'Sol 0-5' },
              { phase: 'Landing sequence', status: 'completed', time: 'Sol 6' },
              { phase: 'System checkout', status: 'active', time: 'Sol 7-10' },
              { phase: 'Science operations', status: 'planned', time: 'Sol 11-40' },
              { phase: 'Extended mission', status: 'planned', time: 'Sol 41+' }
            ].map((phase, idx) => (
              <div key={idx} className="relative flex items-center">
                <div className={`absolute -left-8 w-3 h-3 rounded-full ${
                  phase.status === 'completed' ? 'bg-green-500' :
                  phase.status === 'active' ? 'bg-blue-500' : 'bg-gray-300'
                }`}></div>
                <div className="flex-1">
                  <div className="font-medium">{phase.phase}</div>
                  <div className="text-sm text-gray-600">{phase.time}</div>
                </div>
                <div className={`px-2 py-1 rounded text-xs ${
                  phase.status === 'completed' ? 'bg-green-100 text-green-800' :
                  phase.status === 'active' ? 'bg-blue-100 text-blue-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {phase.status}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  // AI/ML Analysis Panel
  const AIAnalysisPanel = () => (
    <div className="space-y-6">
      {/* Foundation Model Results */}
      <div className="bg-white rounded-lg shadow-lg p-4">
        <h3 className="text-lg font-semibold mb-3 flex items-center">
          <Brain className="mr-2" size={20} />
          Foundation Model Results
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            {
              name: 'Earth-Mars Transfer',
              type: 'earth-mars-transfer',
              status: 'active',
              confidence: 0.94,
              lastRun: '2 min ago',
              insights: 'Found 12 similar geological features'
            },
            {
              name: 'Multi-Modal Fusion',
              type: 'multimodal',
              status: 'processing',
              confidence: 0.88,
              lastRun: '5 min ago',
              insights: 'Integrating thermal + visual data'
            },
            {
              name: 'Self-Supervised Learning',
              type: 'self-supervised',
              status: 'completed',
              confidence: 0.91,
              lastRun: '1 hour ago',
              insights: 'Discovered 3 new terrain patterns'
            },
            {
              name: 'Planetary-Scale Embeddings',
              type: 'planetary-scale',
              status: 'queued',
              confidence: 0.96,
              lastRun: '3 hours ago',
              insights: 'Regional similarity mapping ready'
            }
          ].map(model => (
            <div key={model.type} className="border rounded p-4">
              <div className="flex justify-between items-center mb-2">
                <h4 className="font-medium">{model.name}</h4>
                <span className={`px-2 py-1 rounded text-xs ${
                  model.status === 'active' ? 'bg-green-100 text-green-800' :
                  model.status === 'processing' ? 'bg-blue-100 text-blue-800' :
                  model.status === 'completed' ? 'bg-gray-100 text-gray-800' :
                  'bg-yellow-100 text-yellow-800'
                }`}>
                  {model.status}
                </span>
              </div>

              <div className="mb-3">
                <div className="flex justify-between text-sm mb-1">
                  <span>Confidence</span>
                  <span>{Math.round(model.confidence * 100)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{ width: `${model.confidence * 100}%` }}
                  />
                </div>
              </div>

              <div className="text-sm text-gray-600 mb-2">
                Last run: {model.lastRun}
              </div>

              <div className="text-sm bg-gray-50 p-2 rounded">
                {model.insights}
              </div>

              <button className="mt-2 w-full bg-blue-600 text-white py-1 px-3 rounded text-sm hover:bg-blue-700">
                View Details
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Real-time Analytics */}
      <div className="bg-white rounded-lg shadow-lg p-4">
        <h3 className="text-lg font-semibold mb-3 flex items-center">
          <Activity className="mr-2" size={20} />
          Real-time Environmental Data
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="text-center p-3 bg-blue-50 rounded">
            <Thermometer className="mx-auto mb-2 text-blue-600" size={24} />
            <div className="text-lg font-semibold">
              {realTimeData.temperature?.toFixed(1) || '--'}°C
            </div>
            <div className="text-xs text-gray-600">Temperature</div>
          </div>

          <div className="text-center p-3 bg-green-50 rounded">
            <Activity className="mx-auto mb-2 text-green-600" size={24} />
            <div className="text-lg font-semibold">
              {realTimeData.atmosphericPressure?.toFixed(0) || '--'} Pa
            </div>
            <div className="text-xs text-gray-600">Pressure</div>
          </div>

          <div className="text-center p-3 bg-yellow-50 rounded">
            <Wind className="mx-auto mb-2 text-yellow-600" size={24} />
            <div className="text-lg font-semibold">
              {realTimeData.windSpeed?.toFixed(1) || '--'} m/s
            </div>
            <div className="text-xs text-gray-600">Wind Speed</div>
          </div>

          <div className="text-center p-3 bg-orange-50 rounded">
            <Eye className="mx-auto mb-2 text-orange-600" size={24} />
            <div className="text-lg font-semibold">
              {realTimeData.dustOpacity?.toFixed(2) || '--'}
            </div>
            <div className="text-xs text-gray-600">Dust Opacity</div>
          </div>

          <div className="text-center p-3 bg-purple-50 rounded">
            <Zap className="mx-auto mb-2 text-purple-600" size={24} />
            <div className="text-lg font-semibold">
              {realTimeData.solarIrradiance?.toFixed(0) || '--'} W/m²
            </div>
            <div className="text-xs text-gray-600">Solar Flux</div>
          </div>

          <div className="text-center p-3 bg-red-50 rounded">
            <Satellite className="mx-auto mb-2 text-red-600" size={24} />
            <div className="text-lg font-semibold">Active</div>
            <div className="text-xs text-gray-600">Comm Status</div>
          </div>
        </div>

        {realTimeData.lastUpdate && (
          <div className="mt-4 text-xs text-gray-500 text-center">
            Last updated: {new Date(realTimeData.lastUpdate).toLocaleTimeString()}
          </div>
        )}
      </div>

      {/* Terrain Classification */}
      <div className="bg-white rounded-lg shadow-lg p-4">
        <h3 className="text-lg font-semibold mb-3 flex items-center">
          <Mountain className="mr-2" size={20} />
          Real-time Terrain Classification
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium mb-2">Current Analysis Region</h4>
            {selectedRegion ? (
              <div className="bg-gray-50 p-3 rounded">
                <div className="text-sm">
                  <strong>Coordinates:</strong> {selectedRegion.latitude.toFixed(3)}°, {selectedRegion.longitude.toFixed(3)}°
                </div>
                <div className="mt-2 space-y-1">
                  <div className="flex justify-between">
                    <span>Terrain Type:</span>
                    <span className="font-medium">Ancient Plains</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Hazard Level:</span>
                    <span className="text-green-600 font-medium">Low</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Composition:</span>
                    <span className="font-medium">Basaltic</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-gray-50 p-3 rounded text-center text-gray-500">
                Click on the Mars globe to select a region for analysis
              </div>
            )}
          </div>

          <div>
            <h4 className="font-medium mb-2">Classification Confidence</h4>
            <div className="space-y-2">
              {[
                { type: 'Plains', confidence: 0.78 },
                { type: 'Crater rim', confidence: 0.15 },
                { type: 'Channel', confidence: 0.05 },
                { type: 'Dune field', confidence: 0.02 }
              ].map(classification => (
                <div key={classification.type} className="flex items-center">
                  <span className="w-20 text-sm">{classification.type}</span>
                  <div className="flex-1 bg-gray-200 rounded-full h-2 mx-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${classification.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-sm w-12 text-right">
                    {Math.round(classification.confidence * 100)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Data Management Interface
  const DataManagementPanel = () => (
    <div className="space-y-6">
      {/* Dataset Browser */}
      <div className="bg-white rounded-lg shadow-lg p-4">
        <h3 className="text-lg font-semibold mb-3 flex items-center">
          <Database className="mr-2" size={20} />
          NASA/USGS Dataset Browser
        </h3>

        <div className="mb-4 flex space-x-2">
          <div className="flex-1">
            <input
              type="text"
              placeholder="Search datasets..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full p-2 border rounded"
            />
          </div>
          <button className="p-2 border rounded hover:bg-gray-50">
            <Search size={20} />
          </button>
          <button className="p-2 border rounded hover:bg-gray-50">
            <Filter size={20} />
          </button>
        </div>

        <div className="space-y-3 max-h-96 overflow-y-auto">
          {[
            {
              name: 'Mars Reconnaissance Orbiter CTX',
              type: 'Imaging',
              size: '2.4 TB',
              resolution: '6 m/pixel',
              coverage: 'Global',
              updated: '2025-07-30',
              source: 'NASA PDS'
            },
            {
              name: 'MOLA Elevation Model',
              type: 'Topography',
              size: '850 GB',
              resolution: '128 px/deg',
              coverage: 'Global',
              updated: '2025-07-25',
              source: 'NASA GSFC'
            },
            {
              name: 'THEMIS Thermal Infrared',
              type: 'Thermal',
              size: '1.1 TB',
              resolution: '100 m/pixel',
              coverage: 'Global',
              updated: '2025-08-01',
              source: 'NASA JPL'
            },
            {
              name: 'USGS Geological Map',
              type: 'Geological',
              size: '45 GB',
              resolution: 'Vector',
              coverage: 'Selected regions',
              updated: '2025-06-15',
              source: 'USGS'
            }
          ].map((dataset, idx) => (
            <div key={idx} className="border rounded p-4 hover:bg-gray-50">
              <div className="flex justify-between items-start mb-2">
                <h4 className="font-medium">{dataset.name}</h4>
                <div className="flex space-x-1">
                  <button className="p-1 text-blue-600 hover:bg-blue-50 rounded">
                    <Eye size={16} />
                  </button>
                  <button className="p-1 text-green-600 hover:bg-green-50 rounded">
                    <Download size={16} />
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm text-gray-600">
                <div>Type: {dataset.type}</div>
                <div>Size: {dataset.size}</div>
                <div>Resolution: {dataset.resolution}</div>
                <div>Coverage: {dataset.coverage}</div>
              </div>

              <div className="mt-2 flex justify-between text-xs text-gray-500">
                <span>Source: {dataset.source}</span>
                <span>Updated: {dataset.updated}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Processing Jobs */}
      <div className="bg-white rounded-lg shadow-lg p-4">
        <h3 className="text-lg font-semibold mb-3 flex items-center">
          <Cpu className="mr-2" size={20} />
          Data Processing Status
        </h3>

        <div className="space-y-3">
          {[
            {
              id: 'job-001',
              name: 'Multi-resolution terrain analysis',
              type: 'Terrain Processing',
              status: 'processing',
              progress: 67,
              startTime: '10:30 AM',
              estimatedCompletion: '11:45 AM',
              dataSize: 2.4
            },
            {
              id: 'job-002',
              name: 'Atmospheric density modeling',
              type: 'Atmospheric Analysis',
              status: 'queued',
              progress: 0,
              startTime: '11:00 AM',
              estimatedCompletion: '12:30 PM',
              dataSize: 1.8
            },
            {
              id: 'job-003',
              name: 'Landing site hazard assessment',
              type: 'Safety Analysis',
              status: 'completed',
              progress: 100,
              startTime: '09:15 AM',
              estimatedCompletion: '10:20 AM',
              dataSize: 0.9
            }
          ].map(job => (
            <div key={job.id} className="border rounded p-4">
              <div className="flex justify-between items-center mb-2">
                <h4 className="font-medium">{job.name}</h4>
                <span className={`px-2 py-1 rounded text-xs ${
                  job.status === 'processing' ? 'bg-blue-100 text-blue-800' :
                  job.status === 'queued' ? 'bg-yellow-100 text-yellow-800' :
                  job.status === 'completed' ? 'bg-green-100 text-green-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {job.status}
                </span>
              </div>

              <div className="mb-2">
                <div className="flex justify-between text-sm mb-1">
                  <span>{job.type}</span>
                  <span>{job.progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      job.status === 'completed' ? 'bg-green-500' :
                      job.status === 'processing' ? 'bg-blue-500' : 'bg-gray-400'
                    }`}
                    style={{ width: `${job.progress}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 text-xs text-gray-600">
                <div>Started: {job.startTime}</div>
                <div>ETA: {job.estimatedCompletion}</div>
                <div>Size: {job.dataSize} GB</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Export Tools */}
      <div className="bg-white rounded-lg shadow-lg p-4">
        <h3 className="text-lg font-semibold mb-3 flex items-center">
          <Upload className="mr-2" size={20} />
          Export & Visualization Tools
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium mb-2">Export Formats</h4>
            <div className="space-y-2">
              {['GeoTIFF', 'HDF5', 'NetCDF', 'Shapefile', 'KML'].map(format => (
                <button
                  key={format}
                  className="w-full p-2 border rounded text-left hover:bg-gray-50"
                >
                  Export as {format}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h4 className="font-medium mb-2">Web Visualization</h4>
            <div className="space-y-2">
              <button className="w-full p-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                Generate Web Map
              </button>
              <button className="w-full p-2 border rounded hover:bg-gray-50">
                Create 3D Scene
              </button>
              <button className="w-full p-2 border rounded hover:bg-gray-50">
                Scientific Report
              </button>
              <button className="w-full p-2 border rounded hover:bg-gray-50">
                Mission Briefing
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Globe className="text-red-600 mr-3" size={32} />
              <div>
                <h1 className="text-xl font-bold text-gray-900">MARS-GIS Scientific Platform</h1>
                <p className="text-sm text-gray-600">Mars Exploration & Geospatial Analysis System</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                Sol 2847 | Earth: {new Date().toLocaleDateString()}
              </div>
              <button className="p-2 text-gray-600 hover:bg-gray-100 rounded">
                <Settings size={20} />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'globe', label: 'Mars Analysis', icon: Globe },
              { id: 'planning', label: 'Mission Planning', icon: Target },
              { id: 'ai', label: 'AI/ML Analysis', icon: Brain },
              { id: 'data', label: 'Data Management', icon: Database }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setSelectedTab(tab.id as any)}
                className={`flex items-center px-3 py-4 text-sm font-medium border-b-2 transition-colors ${
                  selectedTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="mr-2" size={18} />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {selectedTab === 'globe' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <MarsGlobe3D />
              <LayerControlPanel />
            </div>
            <div>
              <LandingSiteSelector />
            </div>
          </div>
        )}

        {selectedTab === 'planning' && <MissionPlanningPanel />}
        {selectedTab === 'ai' && <AIAnalysisPanel />}
        {selectedTab === 'data' && <DataManagementPanel />}
      </main>
    </div>
  );
};

export default MarsScientificGUI;
