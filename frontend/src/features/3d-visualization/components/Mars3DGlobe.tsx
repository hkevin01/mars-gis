import { Moon, Navigation, Pause, Play, RotateCcw, Settings, Sun } from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import {
    ACESFilmicToneMapping,
    AdditiveBlending,
    AmbientLight,
    BackSide,
    ClampToEdgeWrapping,
    DirectionalLight,
    LinearFilter,
    Mesh,
    MeshPhongMaterial,
    MeshStandardMaterial,
    MultiplyBlending,
    NormalBlending,
    PCFSoftShadowMap,
    PerspectiveCamera,
    Raycaster,
    RepeatWrapping,
    Scene,
    ShaderMaterial,
    SphereGeometry,
    Texture,
    TextureLoader,
    Vector2,
    WebGLRenderer
} from 'three';

// Types
interface MarsLocation {
  id: string;
  name: string;
  lat: number;
  lon: number;
  type: string;
  description?: string;
}

interface Mars3DGlobeProps {
  width?: number;
  height?: number;
  autoRotate?: boolean;
  showAtmosphere?: boolean;
  elevationScale?: number;
  onLocationClick?: (lat: number, lon: number) => void;
  selectedLocation?: MarsLocation | null;
}

interface LayerConfig {
  id: string;
  name: string;
  visible: boolean;
  opacity: number;
  textureUrl?: string;
  blendMode?: THREE.Blending;
}

// Constants
const MARS_RADIUS = 3396.2; // km
const ROTATION_SPEED = 0.001;
const MARS_COLOR = 0xb86434;

const MARS_TEXTURE_URLS = {
  base: '/textures/mars_base.jpg',
  elevation: '/textures/mars_elevation.jpg',
  normal: '/textures/mars_normal.jpg'
};

const Mars3DGlobe: React.FC<Mars3DGlobeProps> = ({
  width = 800,
  height = 600,
  autoRotate = true,
  showAtmosphere = true,
  elevationScale = 1,
  onLocationClick,
  selectedLocation
}) => {
  // Scene refs
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<Scene | null>(null);
  const rendererRef = useRef<WebGLRenderer | null>(null);
  const cameraRef = useRef<PerspectiveCamera | null>(null);
  const marsRef = useRef<Mesh | null>(null);
  const atmosphereRef = useRef<Mesh | null>(null);
  const animationRef = useRef<number | null>(null);
  const textureLoader = useRef<TextureLoader>(new TextureLoader());
  const controlsRef = useRef<any>(null);
  const isMouseDown = useRef<boolean>(false);
  const previousMousePosition = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Component state
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
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
      blendMode: NormalBlending
    },
    {
      id: 'elevation',
      name: 'Elevation',
      visible: false,
      opacity: 0.5,
      textureUrl: MARS_TEXTURE_URLS.elevation,
      blendMode: MultiplyBlending
    }
  ]);

  // Create Mars atmosphere
  const createAtmosphere = useCallback(() => {
    if (!sceneRef.current) return;

    const atmosphereGeometry = new SphereGeometry(MARS_RADIUS * 1.015, 64, 32);
    const atmosphereMaterial = new ShaderMaterial({
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
      blending: AdditiveBlending,
      side: BackSide,
      transparent: true
    });

    const atmosphere = new Mesh(atmosphereGeometry, atmosphereMaterial);
    atmosphereRef.current = atmosphere;
    sceneRef.current.add(atmosphere);
  }, []);

  // Setup lighting based on mode
  const setupLighting = useCallback(() => {
    if (!sceneRef.current) return;

    // Remove existing lights
    const lights = sceneRef.current.children.filter(child => child instanceof DirectionalLight || child instanceof AmbientLight);
    lights.forEach(light => sceneRef.current!.remove(light));

    switch (lightingMode) {
      case 'day': {
        // Bright sunlight
        const sunLight = new DirectionalLight(0xffffff, 1.2);
        sunLight.position.set(5000, 0, 0);
        sunLight.castShadow = true;
        sunLight.shadow.mapSize.width = 2048;
        sunLight.shadow.mapSize.height = 2048;
        sceneRef.current.add(sunLight);

        // Ambient light for global illumination
        const ambientLight = new AmbientLight(0x404040, 0.3);
        sceneRef.current.add(ambientLight);
        break;
      }

      case 'night': {
        // Low ambient lighting
        const nightAmbient = new AmbientLight(0x202040, 0.1);
        sceneRef.current.add(nightAmbient);
        break;
      }

      case 'terminator': {
        // Dramatic side lighting
        const terminatorLight = new DirectionalLight(0xffaa44, 0.8);
        terminatorLight.position.set(0, 0, 5000);
        sceneRef.current.add(terminatorLight);

        const fillLight = new AmbientLight(0x404040, 0.2);
        sceneRef.current.add(fillLight);
        break;
      }
    }
  }, [lightingMode]);

  // Create Mars sphere with materials
  const createMarsGlobe = useCallback(async () => {
    if (!sceneRef.current) return;

    const createBasicMars = () => {
      const geometry = new SphereGeometry(MARS_RADIUS, 64, 32);
      const material = new MeshStandardMaterial({
        color: MARS_COLOR,
        roughness: 0.7,
        metalness: 0.3
      });
      return new Mesh(geometry, material);
    };

    const loadTextures = async () => {
      const [baseTexture, normalTexture] = await Promise.all([
        new Promise<Texture>((resolve, reject) =>
          textureLoader.current.load(MARS_TEXTURE_URLS.base, resolve, undefined, reject)
        ),
        new Promise<Texture>((resolve, reject) =>
          textureLoader.current.load(MARS_TEXTURE_URLS.normal, resolve, undefined, reject)
        )
      ]);

      // Configure textures
      [baseTexture, normalTexture].forEach(texture => {
        texture.wrapS = RepeatWrapping;
        texture.wrapT = ClampToEdgeWrapping;
        texture.minFilter = LinearFilter;
        texture.magFilter = LinearFilter;
      });

      return { baseTexture, normalTexture };
    };

    try {
      // Load textures
      const { baseTexture, normalTexture } = await loadTextures();

      // Mars geometry with enhanced detail
      const geometry = new SphereGeometry(MARS_RADIUS, 128, 64);

      // Enhanced Mars material
      const material = new MeshPhongMaterial({
        map: baseTexture,
        normalMap: normalTexture,
        normalScale: new Vector2(0.3, 0.3),
        shininess: 1,
        transparent: false,
        color: 0xffffff
      });

      const mars = new Mesh(geometry, material);
      mars.castShadow = true;
      mars.receiveShadow = true;
      marsRef.current = mars;
      sceneRef.current.add(mars);

      // Create atmosphere if enabled
      if (showAtmosphere) {
        createAtmosphere();
      }

      setIsLoading(false);
    } catch {
      // Use fallback material on texture load error
      const mars = createBasicMars();
      marsRef.current = mars;
      sceneRef.current.add(mars);
      setIsLoading(false);
      setError('Using simplified Mars model - high quality textures could not be loaded');
    }
  }, [showAtmosphere, createAtmosphere]);

  // Setup mouse controls
  const setupMouseControls = useCallback(() => {
    if (!mountRef.current || !cameraRef.current || !rendererRef.current) return;

    const handleMouseDown = (event: MouseEvent) => {
      isMouseDown.current = true;
      previousMousePosition.current = {
        x: event.clientX,
        y: event.clientY
      };
    };

    const handleMouseMove = (event: MouseEvent) => {
      if (!isMouseDown.current || !marsRef.current) return;

      const deltaMove = {
        x: event.clientX - previousMousePosition.current.x,
        y: event.clientY - previousMousePosition.current.y
      };

      marsRef.current.rotation.y += deltaMove.x * 0.005;
      marsRef.current.rotation.x += deltaMove.y * 0.005;

      previousMousePosition.current = {
        x: event.clientX,
        y: event.clientY
      };
    };

    const handleMouseUp = () => {
      isMouseDown.current = false;
    };

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      if (!cameraRef.current) return;

      const scale = event.deltaY > 0 ? 1.1 : 0.9;
      const pos = cameraRef.current.position.clone();
      pos.multiplyScalar(scale);

      // Constrain zoom
      const distance = pos.length();
      if (distance < MARS_RADIUS * 1.2) {
        pos.normalize().multiplyScalar(MARS_RADIUS * 1.2);
      }
      if (distance > MARS_RADIUS * 8) {
        pos.normalize().multiplyScalar(MARS_RADIUS * 8);
      }

      cameraRef.current.position.copy(pos);
    };

    // Add event listeners
    rendererRef.current.domElement.addEventListener('mousedown', handleMouseDown);
    rendererRef.current.domElement.addEventListener('mousemove', handleMouseMove);
    rendererRef.current.domElement.addEventListener('mouseup', handleMouseUp);
    rendererRef.current.domElement.addEventListener('wheel', handleWheel);

    // Store cleanup function
    controlsRef.current = {
      dispose: () => {
        rendererRef.current?.domElement.removeEventListener('mousedown', handleMouseDown);
        rendererRef.current?.domElement.removeEventListener('mousemove', handleMouseMove);
        rendererRef.current?.domElement.removeEventListener('mouseup', handleMouseUp);
        rendererRef.current?.domElement.removeEventListener('wheel', handleWheel);
      }
    };
  }, []);

  // Initialize scene
  const initScene = useCallback(async () => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new Scene();
    scene.background = new THREE.Color(0x000000);
    sceneRef.current = scene;

    // Camera setup
    const aspectRatio = width / height;
    const camera = new PerspectiveCamera(
      60, // FOV - reduced for better perspective
      aspectRatio,
      0.1, // Near plane
      MARS_RADIUS * 10 // Far plane
    );
    camera.position.set(0, 0, MARS_RADIUS * 3);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = PCFSoftShadowMap;
    renderer.toneMapping = ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Create Mars globe with materials
    await createMarsGlobe();

    // Setup lighting
    setupLighting();

    // Setup mouse controls
    setupMouseControls();

  }, [width, height, createMarsGlobe, setupLighting, setupMouseControls]);

  // Animation loop
  const animate = useCallback(() => {
    if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

    // Auto-rotation when enabled
    if (isRotating && marsRef.current) {
      marsRef.current.rotation.y += ROTATION_SPEED;
    }

    // Update atmosphere animation
    if (atmosphereRef.current && atmosphereRef.current.material instanceof ShaderMaterial) {
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
    if (!cameraRef.current) return;

    cameraRef.current.position.set(0, 0, MARS_RADIUS * 3);
    cameraRef.current.lookAt(0, 0, 0);
  }, []);

  // Handle location selection
  const handleLocationClick = useCallback((event: MouseEvent) => {
    if (!marsRef.current || !cameraRef.current || !rendererRef.current || !onLocationClick) return;

    const raycaster = new Raycaster();
    const mouse = new Vector2();

    const rect = rendererRef.current.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, cameraRef.current);

    const intersects = raycaster.intersectObject(marsRef.current);
    if (intersects.length > 0) {
      const point = intersects[0].point.clone();
      const lat = Math.asin(point.y / MARS_RADIUS) * (180 / Math.PI);
      const lon = Math.atan2(point.z, point.x) * (180 / Math.PI);
      onLocationClick(lat, lon);
    }
  }, [onLocationClick]);

  // Initialize scene on mount
  useEffect(() => {
    initScene();
  }, [initScene]);

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

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (!cameraRef.current || !rendererRef.current) return;

      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(width, height);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [width, height]);

  // Add location click handler
  useEffect(() => {
    if (!mountRef.current || !onLocationClick) return;

    const element = mountRef.current;
    element.addEventListener('click', handleLocationClick);
    return () => {
      element.removeEventListener('click', handleLocationClick);
    };
  }, [onLocationClick, handleLocationClick]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      controlsRef.current?.dispose();
      rendererRef.current?.dispose();
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
            {selectedLocation.description && (
              <div className="text-xs text-gray-400 mt-1">{selectedLocation.description}</div>
            )}
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

      {/* Error Message */}
      {error && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-red-600/90 backdrop-blur-sm rounded-lg p-2 text-white text-sm">
          {error}
        </div>
      )}
    </div>
  );
};

export default Mars3DGlobe;
