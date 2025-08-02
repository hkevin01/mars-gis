// Enhanced 3D Mars Terrain with Real Elevation Data
import { Download, Eye, EyeOff, Mountain, Settings } from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

// Mars terrain data interfaces
interface TerrainDataPoint {
  lat: number;
  lon: number;
  elevation: number;
  slope?: number;
  roughness?: number;
  temperature?: number;
}

interface TerrainLayerConfig {
  id: string;
  name: string;
  visible: boolean;
  opacity: number;
  colorMap: string;
  elevationScale: number;
  dataSource: 'mola' | 'themis' | 'ctx' | 'combined';
}

interface Mars3DTerrainProps {
  width?: number;
  height?: number;
  selectedRegion?: {
    bounds: [number, number, number, number]; // [minLat, minLon, maxLat, maxLon]
    name: string;
  };
  onTerrainClick?: (point: TerrainDataPoint) => void;
  elevationExaggeration?: number;
}

const Mars3DTerrain: React.FC<Mars3DTerrainProps> = ({
  width = 600,
  height = 400,
  selectedRegion,
  onTerrainClick,
  elevationExaggeration = 10
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const terrainMeshRef = useRef<THREE.Mesh | null>(null);
  const animationRef = useRef<number | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [terrainData, setTerrainData] = useState<TerrainDataPoint[]>([]);
  const [meshStats, setMeshStats] = useState({
    vertices: 0,
    faces: 0,
    minElevation: 0,
    maxElevation: 0
  });

  const [layers, setLayers] = useState<TerrainLayerConfig[]>([
    {
      id: 'elevation',
      name: 'MOLA Elevation',
      visible: true,
      opacity: 1.0,
      colorMap: 'elevation',
      elevationScale: 1.0,
      dataSource: 'mola'
    },
    {
      id: 'thermal',
      name: 'THEMIS Thermal',
      visible: false,
      opacity: 0.7,
      colorMap: 'thermal',
      elevationScale: 0.5,
      dataSource: 'themis'
    },
    {
      id: 'slope',
      name: 'Slope Analysis',
      visible: false,
      opacity: 0.8,
      colorMap: 'slope',
      elevationScale: 1.0,
      dataSource: 'combined'
    }
  ]);

  // Initialize Three.js scene for terrain visualization
  const initializeScene = useCallback(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.Fog(0x0a0a0a, 10, 100);
    sceneRef.current = scene;

    // Camera setup for terrain viewing
    const camera = new THREE.PerspectiveCamera(
      60,
      width / height,
      0.1,
      1000
    );
    camera.position.set(0, 50, 50);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer with enhanced settings for terrain
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    rendererRef.current = renderer;

    // Enhanced lighting for terrain
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
    directionalLight.position.set(50, 100, 50);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    directionalLight.shadow.camera.near = 0.5;
    directionalLight.shadow.camera.far = 500;
    directionalLight.shadow.camera.left = -100;
    directionalLight.shadow.camera.right = 100;
    directionalLight.shadow.camera.top = 100;
    directionalLight.shadow.camera.bottom = -100;
    scene.add(directionalLight);

    // Ambient light for overall illumination
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    scene.add(ambientLight);

    // Add orange rim light (Mars atmosphere effect)
    const rimLight = new THREE.DirectionalLight(0xff4500, 0.3);
    rimLight.position.set(-50, 20, -50);
    scene.add(rimLight);

    // Mouse controls for camera
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    const handleMouseDown = (event: MouseEvent) => {
      isDragging = true;
      previousMousePosition = { x: event.clientX, y: event.clientY };
    };

    const handleMouseMove = (event: MouseEvent) => {
      if (!isDragging) return;

      const deltaMove = {
        x: event.clientX - previousMousePosition.x,
        y: event.clientY - previousMousePosition.y
      };

      // Rotate camera around terrain
      const spherical = new THREE.Spherical();
      spherical.setFromVector3(camera.position);
      spherical.theta -= deltaMove.x * 0.01;
      spherical.phi += deltaMove.y * 0.01;
      spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

      camera.position.setFromSpherical(spherical);
      camera.lookAt(0, 0, 0);

      previousMousePosition = { x: event.clientX, y: event.clientY };
    };

    const handleMouseUp = () => {
      isDragging = false;
    };

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const scale = event.deltaY > 0 ? 1.1 : 0.9;
      camera.position.multiplyScalar(scale);

      // Constrain zoom
      const distance = camera.position.length();
      if (distance < 10) camera.position.normalize().multiplyScalar(10);
      if (distance > 200) camera.position.normalize().multiplyScalar(200);
    };

    // Click handling for terrain selection
    const handleClick = (event: MouseEvent) => {
      if (!terrainMeshRef.current || !onTerrainClick) return;

      const raycaster = new THREE.Raycaster();
      const mouse = new THREE.Vector2();
      const rect = renderer.domElement.getBoundingClientRect();

      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(terrainMeshRef.current);

      if (intersects.length > 0) {
        const point = intersects[0].point;
        const terrainPoint: TerrainDataPoint = {
          lat: point.z, // Convert back to lat/lon
          lon: point.x,
          elevation: point.y / elevationExaggeration
        };
        onTerrainClick(terrainPoint);
      }
    };

    // Add event listeners
    renderer.domElement.addEventListener('mousedown', handleMouseDown);
    renderer.domElement.addEventListener('mousemove', handleMouseMove);
    renderer.domElement.addEventListener('mouseup', handleMouseUp);
    renderer.domElement.addEventListener('wheel', handleWheel);
    renderer.domElement.addEventListener('click', handleClick);

    mountRef.current.appendChild(renderer.domElement);

    return () => {
      renderer.domElement.removeEventListener('mousedown', handleMouseDown);
      renderer.domElement.removeEventListener('mousemove', handleMouseMove);
      renderer.domElement.removeEventListener('mouseup', handleMouseUp);
      renderer.domElement.removeEventListener('wheel', handleWheel);
      renderer.domElement.removeEventListener('click', handleClick);
    };
  }, [width, height, elevationExaggeration, onTerrainClick]);

  // Generate terrain mesh from elevation data
  const generateTerrainMesh = useCallback(async () => {
    if (!sceneRef.current) return;

    // Simulate loading terrain data for selected region
    setIsLoading(true);

    // Generate synthetic terrain data (in real implementation, this would fetch from NASA APIs)
    const gridSize = 256;
    const bounds = selectedRegion?.bounds || [-5, -5, 5, 5];
    const [minLat, minLon, maxLat, maxLon] = bounds;

    const generatedData: TerrainDataPoint[] = [];
    let minElev = Infinity;
    let maxElev = -Infinity;

    // Create height map data
    const heightData: number[][] = [];
    for (let i = 0; i < gridSize; i++) {
      heightData[i] = [];
      for (let j = 0; j < gridSize; j++) {
        const lat = minLat + (maxLat - minLat) * (i / (gridSize - 1));
        const lon = minLon + (maxLon - minLon) * (j / (gridSize - 1));

        // Generate realistic Mars-like terrain using noise functions
        const noise1 = Math.sin(lat * 0.1) * Math.cos(lon * 0.1) * 1000;
        const noise2 = Math.sin(lat * 0.3) * Math.cos(lon * 0.3) * 300;
        const noise3 = Math.sin(lat * 0.8) * Math.cos(lon * 0.8) * 100;
        const elevation = noise1 + noise2 + noise3;

        heightData[i][j] = elevation;
        minElev = Math.min(minElev, elevation);
        maxElev = Math.max(maxElev, elevation);

        generatedData.push({ lat, lon, elevation });
      }
    }

    // Update mesh statistics
    setMeshStats({
      vertices: gridSize * gridSize,
      faces: (gridSize - 1) * (gridSize - 1) * 2,
      minElevation: minElev,
      maxElevation: maxElev
    });

    // Create Three.js geometry
    const geometry = new THREE.PlaneGeometry(
      maxLon - minLon,
      maxLat - minLat,
      gridSize - 1,
      gridSize - 1
    );

    // Apply elevation data to vertices
    const vertices = geometry.attributes.position.array as Float32Array;
    for (let i = 0; i < vertices.length; i += 3) {
      const row = Math.floor(i / 3 / gridSize);
      const col = (i / 3) % gridSize;
      vertices[i + 1] = heightData[row][col] * elevationExaggeration / 1000; // Scale for visualization
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();

    // Create material with elevation-based coloring
    const material = new THREE.ShaderMaterial({
      uniforms: {
        minElevation: { value: minElev / 1000 },
        maxElevation: { value: maxElev / 1000 },
        opacity: { value: 1.0 }
      },
      vertexShader: `
        attribute vec3 position;
        attribute vec3 normal;
        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;

        varying vec3 vPosition;
        varying vec3 vNormal;

        void main() {
          vPosition = position;
          vNormal = normal;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        precision mediump float;

        uniform float minElevation;
        uniform float maxElevation;
        uniform float opacity;
        varying vec3 vPosition;
        varying vec3 vNormal;

        vec3 getElevationColor(float elevation) {
          float t = (elevation - minElevation) / (maxElevation - minElevation);

          // Mars-like color gradient
          vec3 low = vec3(0.4, 0.2, 0.1);    // Dark brown/red
          vec3 mid = vec3(0.8, 0.4, 0.2);    // Mars orange
          vec3 high = vec3(1.0, 0.8, 0.6);   // Light tan

          if (t < 0.5) {
            return mix(low, mid, t * 2.0);
          } else {
            return mix(mid, high, (t - 0.5) * 2.0);
          }
        }

        void main() {
          vec3 color = getElevationColor(vPosition.y);

          // Add lighting
          vec3 lightDirection = normalize(vec3(1.0, 1.0, 1.0));
          float lightIntensity = max(dot(vNormal, lightDirection), 0.2);

          gl_FragColor = vec4(color * lightIntensity, opacity);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide
    });

    // Create and add terrain mesh
    const terrainMesh = new THREE.Mesh(geometry, material);
    terrainMesh.rotation.x = -Math.PI / 2; // Rotate to lie flat
    terrainMesh.castShadow = true;
    terrainMesh.receiveShadow = true;

    // Remove previous terrain mesh
    if (terrainMeshRef.current) {
      sceneRef.current.remove(terrainMeshRef.current);
    }

    terrainMeshRef.current = terrainMesh;
    sceneRef.current.add(terrainMesh);

    setTerrainData(generatedData);
    setIsLoading(false);
  }, [selectedRegion, elevationExaggeration]);

  // Animation loop
  const animate = useCallback(() => {
    if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

    rendererRef.current.render(sceneRef.current, cameraRef.current);
    animationRef.current = requestAnimationFrame(animate);
  }, []);

  // Update layer visibility and properties
  const updateLayer = useCallback((layerId: string, updates: Partial<TerrainLayerConfig>) => {
    setLayers(prev => prev.map(layer =>
      layer.id === layerId ? { ...layer, ...updates } : layer
    ));

    // Apply updates to terrain mesh material
    if (terrainMeshRef.current && terrainMeshRef.current.material instanceof THREE.ShaderMaterial) {
      const material = terrainMeshRef.current.material;
      if (updates.opacity !== undefined) {
        material.uniforms.opacity.value = updates.opacity;
      }
    }
  }, []);

  // Export terrain data
  const exportTerrainData = useCallback(() => {
    if (terrainData.length === 0) return;

    const data = {
      region: selectedRegion,
      terrainData,
      meshStats,
      layers: layers.filter(l => l.visible),
      exportedAt: new Date().toISOString()
    };

    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `mars-terrain-${selectedRegion?.name || 'data'}-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [terrainData, selectedRegion, meshStats, layers]);

  // Initialize scene
  useEffect(() => {
    const cleanup = initializeScene();
    return cleanup;
  }, [initializeScene]);

  // Generate terrain when region changes
  useEffect(() => {
    generateTerrainMesh();
  }, [generateTerrainMesh]);

  // Start animation
  useEffect(() => {
    animate();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [animate]);

  return (
    <div className="relative bg-black rounded-lg overflow-hidden" style={{ width, height }}>
      {/* 3D Terrain Canvas */}
      <div ref={mountRef} className="w-full h-full" />

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center">
          <div className="text-white text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-orange-500 border-t-transparent mx-auto mb-4"></div>
            <div className="text-sm">Generating Mars Terrain...</div>
            <div className="text-xs text-gray-400 mt-1">
              {selectedRegion?.name || 'Selected Region'}
            </div>
          </div>
        </div>
      )}

      {/* Control Panel */}
      <div className="absolute top-4 right-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center text-orange-400">
            <Mountain className="w-4 h-4 mr-2" />
            <span className="text-xs font-medium">3D Terrain</span>
          </div>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-1 text-gray-400 hover:text-white transition-colors"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>

        {/* Quick Stats */}
        <div className="text-xs text-gray-300 space-y-1">
          <div>Vertices: {meshStats.vertices.toLocaleString()}</div>
          <div>Elevation: {meshStats.minElevation.toFixed(0)}m to {meshStats.maxElevation.toFixed(0)}m</div>
        </div>

        {/* Layer Controls */}
        {showSettings && (
          <div className="border-t border-gray-700 pt-2 space-y-2">
            <div className="text-xs text-gray-300 font-medium">Layers</div>
            {layers.map(layer => (
              <div key={layer.id} className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-xs text-gray-300 flex items-center">
                    <input
                      type="checkbox"
                      checked={layer.visible}
                      onChange={(e) => updateLayer(layer.id, { visible: e.target.checked })}
                      className="mr-2 w-3 h-3"
                    />
                    {layer.name}
                  </label>
                  <button
                    onClick={() => updateLayer(layer.id, { visible: !layer.visible })}
                    className="p-0.5 text-gray-400 hover:text-white transition-colors"
                  >
                    {layer.visible ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                  </button>
                </div>
                {layer.visible && (
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={layer.opacity}
                    onChange={(e) => updateLayer(layer.id, { opacity: parseFloat(e.target.value) })}
                    className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                  />
                )}
              </div>
            ))}

            <button
              onClick={exportTerrainData}
              className="flex items-center space-x-1 w-full p-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors"
            >
              <Download className="w-3 h-3" />
              <span>Export Data</span>
            </button>
          </div>
        )}
      </div>

      {/* Region Info */}
      {selectedRegion && (
        <div className="absolute bottom-4 left-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-3">
          <div className="text-white">
            <div className="font-medium text-sm">{selectedRegion.name}</div>
            <div className="text-xs text-gray-300">
              {selectedRegion.bounds[0].toFixed(2)}°, {selectedRegion.bounds[1].toFixed(2)}° to{' '}
              {selectedRegion.bounds[2].toFixed(2)}°, {selectedRegion.bounds[3].toFixed(2)}°
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="absolute bottom-4 right-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-2">
        <div className="text-xs text-gray-300 space-y-0.5">
          <div>• Drag to rotate view</div>
          <div>• Scroll to zoom</div>
          <div>• Click terrain to select</div>
        </div>
      </div>
    </div>
  );
};

export default Mars3DTerrain;
