// Enhanced 3D Mars Globe with Three.js Integration
import {
    Moon,
    Navigation,
    Pause,
    Play,
    RotateCcw,
    Settings,
    Sun
} from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';

// Three.js imports
import * as THREE from 'three';

// Mars-specific constants
const MARS_RADIUS = 3389.5; // km
const MARS_TEXTURE_URLS = {
  base: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/textures/4k_mars.jpg',
  elevation: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_ClrShade_merge_global_463m/textures/4k_mars_elevation.jpg',
  normal: 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/textures/4k_mars_normal.jpg'
};

interface Mars3DGlobeProps {
  width?: number;
  height?: number;
  autoRotate?: boolean;
  showAtmosphere?: boolean;
  elevationScale?: number;
  onLocationClick?: (lat: number, lon: number) => void;
  selectedLocation?: { lat: number; lon: number; name: string } | null;
}

interface LayerConfig {
  id: string;
  name: string;
  visible: boolean;
  opacity: number;
  textureUrl: string;
  blendMode: THREE.Blending;
}

const Mars3DGlobe: React.FC<Mars3DGlobeProps> = ({
  width = 800,
  height = 600,
  autoRotate = true,
  showAtmosphere = true,
  elevationScale = 1.0,
  onLocationClick,
  selectedLocation
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<any>(null);
  const marsRef = useRef<THREE.Mesh | null>(null);
  const atmosphereRef = useRef<THREE.Mesh | null>(null);
  const animationRef = useRef<number | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [isRotating, setIsRotating] = useState(autoRotate);
  const [showControls, setShowControls] = useState(false);
  const [lightingMode, setLightingMode] = useState<'day' | 'night' | 'terminator'>('day');

  const [layers, setLayers] = useState<LayerConfig[]>([
    {
      id: 'base',
      name: 'Mars Surface',
      visible: true,
      opacity: 1.0,
      textureUrl: MARS_TEXTURE_URLS.base,
      blendMode: THREE.NormalBlending
    },
    {
      id: 'elevation',
      name: 'Elevation',
      visible: false,
      opacity: 0.5,
      textureUrl: MARS_TEXTURE_URLS.elevation,
      blendMode: THREE.MultiplyBlending
    }
  ]);

  // Initialize Three.js scene
  const initializeScene = useCallback(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      width / height,
      0.1,
      10000
    );
    camera.position.set(0, 0, MARS_RADIUS * 3);
    cameraRef.current = camera;

    // Renderer setup
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
    renderer.toneMappingExposure = 1.2;
    rendererRef.current = renderer;

    // Basic mouse controls setup
    let isMouseDown = false;
    let mouseX = 0;
    let mouseY = 0;
    let targetRotationX = 0;
    let targetRotationY = 0;

    const handleMouseDown = (event: MouseEvent) => {
      isMouseDown = true;
      mouseX = event.clientX;
      mouseY = event.clientY;
    };

    const handleMouseMove = (event: MouseEvent) => {
      if (!isMouseDown) return;

      const deltaX = event.clientX - mouseX;
      const deltaY = event.clientY - mouseY;

      targetRotationX += deltaY * 0.01;
      targetRotationY += deltaX * 0.01;

      // Constrain vertical rotation
      targetRotationX = Math.max(-Math.PI/2, Math.min(Math.PI/2, targetRotationX));

      // Apply rotation to camera
      const radius = camera.position.length();
      camera.position.x = radius * Math.sin(targetRotationY) * Math.cos(targetRotationX);
      camera.position.y = radius * Math.sin(targetRotationX);
      camera.position.z = radius * Math.cos(targetRotationY) * Math.cos(targetRotationX);
      camera.lookAt(0, 0, 0);

      mouseX = event.clientX;
      mouseY = event.clientY;
    };

    const handleMouseUp = () => {
      isMouseDown = false;
    };

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const scale = event.deltaY > 0 ? 1.1 : 0.9;
      camera.position.multiplyScalar(scale);

      // Constrain zoom
      const distance = camera.position.length();
      if (distance < MARS_RADIUS * 1.2) camera.position.normalize().multiplyScalar(MARS_RADIUS * 1.2);
      if (distance > MARS_RADIUS * 8) camera.position.normalize().multiplyScalar(MARS_RADIUS * 8);
    };

    // Add event listeners
    renderer.domElement.addEventListener('mousedown', handleMouseDown);
    renderer.domElement.addEventListener('mousemove', handleMouseMove);
    renderer.domElement.addEventListener('mouseup', handleMouseUp);
    renderer.domElement.addEventListener('wheel', handleWheel);

    // Store controls reference for cleanup
    controlsRef.current = {
      dispose: () => {
        renderer.domElement.removeEventListener('mousedown', handleMouseDown);
        renderer.domElement.removeEventListener('mousemove', handleMouseMove);
        renderer.domElement.removeEventListener('mouseup', handleMouseUp);
        renderer.domElement.removeEventListener('wheel', handleWheel);
      }
    };

    // Add click handler for location selection
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const handleClick = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);

      if (marsRef.current) {
        const intersects = raycaster.intersectObject(marsRef.current);
        if (intersects.length > 0) {
          const point = intersects[0].point;
          const lat = Math.asin(point.y / MARS_RADIUS) * (180 / Math.PI);
          const lon = Math.atan2(point.z, point.x) * (180 / Math.PI);
          onLocationClick?.(lat, lon);
        }
      }
    };

    renderer.domElement.addEventListener('click', handleClick);
    mountRef.current.appendChild(renderer.domElement);

    return () => {
      renderer.domElement.removeEventListener('click', handleClick);
    };
  }, [width, height, onLocationClick]);

  // Create Mars atmosphere
  const createAtmosphere = useCallback(() => {
    if (!sceneRef.current) return;

    const atmosphereGeometry = new THREE.SphereGeometry(MARS_RADIUS * 1.015, 64, 32);
    const atmosphereMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        opacity: { value: 0.3 }
      },
      vertexShader: `
        attribute vec3 position;
        attribute vec3 normal;
        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;
        uniform mat3 normalMatrix;

        varying vec3 vNormal;
        varying vec3 vPosition;

        void main() {
          vNormal = normalize(normalMatrix * normal);
          vPosition = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        precision mediump float;
        uniform float time;
        uniform float opacity;
        varying vec3 vNormal;
        varying vec3 vPosition;

        void main() {
          float intensity = pow(0.8 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
          vec3 atmosphere = vec3(0.8, 0.4, 0.2) * intensity;
          gl_FragColor = vec4(atmosphere, opacity * intensity);
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true
    });

    const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
    atmosphereRef.current = atmosphere;
    sceneRef.current.add(atmosphere);
  }, []);

  // Setup dynamic lighting based on mode
  const setupLighting = useCallback(() => {
    if (!sceneRef.current) return;

    // Remove existing lights
    const lights = sceneRef.current.children.filter((child: any) => child instanceof THREE.Light);
    lights.forEach((light: any) => sceneRef.current!.remove(light));

    switch (lightingMode) {
      case 'day': {
        // Bright sunlight
        const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
        sunLight.position.set(5000, 0, 0);
        sunLight.castShadow = true;
        sunLight.shadow.mapSize.width = 2048;
        sunLight.shadow.mapSize.height = 2048;
        sceneRef.current.add(sunLight);

        // Ambient light for global illumination
        const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
        sceneRef.current.add(ambientLight);
        break;
      }

      case 'night': {
        // Low ambient lighting
        const nightAmbient = new THREE.AmbientLight(0x202040, 0.1);
        sceneRef.current.add(nightAmbient);
        break;
      }

      case 'terminator': {
        // Dramatic side lighting
        const terminatorLight = new THREE.DirectionalLight(0xffaa44, 0.8);
        terminatorLight.position.set(0, 0, 5000);
        sceneRef.current.add(terminatorLight);

        const fillLight = new THREE.AmbientLight(0x404040, 0.2);
        sceneRef.current.add(fillLight);
        break;
      }
    }
  }, [lightingMode]);

  // Create Mars sphere with enhanced materials
  const createMarsGlobe = useCallback(async () => {
    if (!sceneRef.current) return;

    try {
      // Load textures
      const loader = new THREE.TextureLoader();
      const baseTexture = await new Promise<THREE.Texture>((resolve, reject) => {
        loader.load(MARS_TEXTURE_URLS.base, resolve, undefined, reject);
      });

      const normalTexture = await new Promise<THREE.Texture>((resolve, reject) => {
        loader.load(MARS_TEXTURE_URLS.normal, resolve, undefined, reject);
      });

      // Configure textures
      [baseTexture, normalTexture].forEach(texture => {
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
      });

      // Mars geometry with enhanced detail
      const geometry = new THREE.SphereGeometry(MARS_RADIUS, 128, 64);

      // Enhanced Mars material
      const material = new THREE.MeshPhongMaterial({
        map: baseTexture,
        normalMap: normalTexture,
        normalScale: new THREE.Vector2(0.3, 0.3),
        shininess: 1,
        transparent: false,
        color: 0xffffff
      });

      const mars = new THREE.Mesh(geometry, material);
      mars.castShadow = true;
      mars.receiveShadow = true;
      marsRef.current = mars;
      sceneRef.current.add(mars);

      // Create atmosphere if enabled
      if (showAtmosphere) {
        createAtmosphere();
      }

      // Setup lighting
      setupLighting();

      setIsLoading(false);
    } catch {
      // Fallback to basic material
      const geometry = new THREE.SphereGeometry(MARS_RADIUS, 64, 32);
      const material = new THREE.MeshPhongMaterial({
        color: 0xcd853f, // Mars-like color
        shininess: 1
      });
      const mars = new THREE.Mesh(geometry, material);
      marsRef.current = mars;
      sceneRef.current.add(mars);
      setIsLoading(false);
    }
  }, [showAtmosphere, createAtmosphere, setupLighting]);

  // Animation loop
  const animate = useCallback(() => {
    if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

    // Auto-rotation when enabled
    if (isRotating && marsRef.current) {
      marsRef.current.rotation.y += 0.005;
    }

    // Update atmosphere animation
    if (atmosphereRef.current && atmosphereRef.current.material instanceof THREE.ShaderMaterial) {
      atmosphereRef.current.material.uniforms.time.value += 0.01;
    }

    // Render scene
    rendererRef.current.render(sceneRef.current, cameraRef.current);

    animationRef.current = requestAnimationFrame(animate);
  }, [isRotating]);

  // Toggle layer visibility
  const toggleLayer = useCallback((layerId: string) => {
    setLayers(prev => prev.map(layer =>
      layer.id === layerId
        ? { ...layer, visible: !layer.visible }
        : layer
    ));
  }, []);

  // Update layer opacity
  const updateLayerOpacity = useCallback((layerId: string, opacity: number) => {
    setLayers(prev => prev.map(layer =>
      layer.id === layerId
        ? { ...layer, opacity }
        : layer
    ));
  }, []);

  // Reset camera position
  const resetCamera = useCallback(() => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(0, 0, MARS_RADIUS * 3);
      controlsRef.current.reset();
    }
  }, []);

  // Initialize scene on mount
  useEffect(() => {
    const cleanup = initializeScene();
    createMarsGlobe();

    return cleanup;
  }, [initializeScene, createMarsGlobe]);

  // Start animation loop
  useEffect(() => {
    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [animate]);

  // Update lighting when mode changes
  useEffect(() => {
    setupLighting();
  }, [lightingMode, setupLighting]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <div className="relative" style={{ width, height }}>
      {/* 3D Canvas Container */}
      <div ref={mountRef} className="w-full h-full bg-black rounded-lg overflow-hidden" />

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center rounded-lg">
          <div className="text-white text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-orange-500 border-t-transparent mx-auto mb-4"></div>
            <div className="text-sm">Loading Mars 3D Globe...</div>
          </div>
        </div>
      )}

      {/* Control Panel */}
      <div className="absolute top-4 right-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 space-y-2">
        {/* Main Controls */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsRotating(!isRotating)}
            className={`p-2 rounded ${isRotating ? 'bg-orange-600' : 'bg-gray-600'} text-white hover:opacity-80 transition-opacity`}
            title={isRotating ? 'Pause Rotation' : 'Start Rotation'}
          >
            {isRotating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>

          <button
            onClick={resetCamera}
            className="p-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
            title="Reset Camera"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <button
            onClick={() => setShowControls(!showControls)}
            className="p-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
            title="Toggle Settings"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>

        {/* Lighting Controls */}
        <div className="flex items-center space-x-1">
          <button
            onClick={() => setLightingMode('day')}
            className={`p-1.5 rounded ${lightingMode === 'day' ? 'bg-yellow-600' : 'bg-gray-600'} text-white text-xs`}
            title="Day Lighting"
          >
            <Sun className="w-3 h-3" />
          </button>
          <button
            onClick={() => setLightingMode('night')}
            className={`p-1.5 rounded ${lightingMode === 'night' ? 'bg-blue-600' : 'bg-gray-600'} text-white text-xs`}
            title="Night Lighting"
          >
            <Moon className="w-3 h-3" />
          </button>
          <button
            onClick={() => setLightingMode('terminator')}
            className={`p-1.5 rounded ${lightingMode === 'terminator' ? 'bg-orange-600' : 'bg-gray-600'} text-white text-xs`}
            title="Terminator Lighting"
          >
            <Navigation className="w-3 h-3" />
          </button>
        </div>

        {/* Extended Controls */}
        {showControls && (
          <div className="border-t border-gray-700 pt-2 space-y-2">
            <div className="text-xs text-gray-300 font-medium">Layers</div>
            {layers.map(layer => (
              <div key={layer.id} className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-xs text-gray-300 flex items-center">
                    <input
                      type="checkbox"
                      checked={layer.visible}
                      onChange={() => toggleLayer(layer.id)}
                      className="mr-2 w-3 h-3"
                    />
                    {layer.name}
                  </label>
                </div>
                {layer.visible && (
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={layer.opacity}
                    onChange={(e) => updateLayerOpacity(layer.id, parseFloat(e.target.value))}
                    className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                  />
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Location Info */}
      {selectedLocation && (
        <div className="absolute bottom-4 left-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-3">
          <div className="text-white">
            <div className="font-medium">{selectedLocation.name}</div>
            <div className="text-sm text-gray-300">
              {selectedLocation.lat.toFixed(4)}°, {selectedLocation.lon.toFixed(4)}°
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="absolute bottom-4 right-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-2">
        <div className="text-xs text-gray-300">
          <div>• Drag to rotate</div>
          <div>• Scroll to zoom</div>
          <div>• Click surface to select</div>
        </div>
      </div>
    </div>
  );
};

export default Mars3DGlobe;
