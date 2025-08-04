import { useCallback, useEffect, useRef, useState } from 'react';
import {
    ACESFilmicToneMapping,
    AdditiveBlending,
    AmbientLight,
    BackSide,
    Clock,
    Color,
    DirectionalLight,
    LinearFilter,
    Mesh,
    MeshStandardMaterial,
    Object3D,
    PCFSoftShadowMap,
    PerspectiveCamera,
    RepeatWrapping,
    Scene,
    ShaderMaterial,
    SphereGeometry,
    TextureLoader,
    Vector3,
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
  blendMode?: number;
}

// Constants
const MARS_RADIUS = 3396.2; // km
const ROTATION_SPEED = 0.001;
const MARS_COLOR = 0xb86434;

const MARS_TEXTURE_URLS = {
  base: '/mars_base.jpg',
  elevation: '/mars_elevation.jpg',
  normal: '/mars_normal.jpg',
  fallback: '/mars_simplified.jpg'
};

export const Mars3DGlobe = ({
  width = 800,
  height = 600,
  autoRotate = true,
  showAtmosphere = true,
  elevationScale = 1.0,
  onLocationClick,
  selectedLocation
}: Mars3DGlobeProps): JSX.Element => {
  // Component refs
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<Scene | null>(null);
  const rendererRef = useRef<WebGLRenderer | null>(null);
  const cameraRef = useRef<PerspectiveCamera | null>(null);
  const marsRef = useRef<Mesh | null>(null);
  const atmosphereRef = useRef<Mesh | null>(null);
  const animationRef = useRef<number | null>(null);
  const textureLoader = useRef<TextureLoader>(new TextureLoader());
  const controlsRef = useRef<any>(null);
  const clock = useRef<Clock>(new Clock());

  // Component state
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
      blendMode: 1
    },
    {
      id: 'elevation',
      name: 'Elevation',
      visible: false,
      opacity: 0.5,
      textureUrl: MARS_TEXTURE_URLS.elevation,
      blendMode: 2
    }
  ]);

  // Add error state
  const [loadError, setLoadError] = useState<string | null>(null);

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

  // Setup dynamic lighting based on mode
  const setupLighting = useCallback(() => {
    if (!sceneRef.current) return;

    // Remove existing lights
    const lights = sceneRef.current.children.filter((child: Object3D) => child instanceof DirectionalLight || child instanceof AmbientLight);
    lights.forEach((light: Object3D) => sceneRef.current!.remove(light));

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

  // Create Mars sphere with enhanced materials
  const createMarsGlobe = useCallback(async () => {
    if (!sceneRef.current) return;

    const geometry = new SphereGeometry(MARS_RADIUS, 64, 32);

    try {
      // Try loading the simplified texture first
      const texture = await new Promise((resolve, reject) => {
        textureLoader.current.load(
          MARS_TEXTURE_URLS.fallback,
          (texture) => {
            texture.wrapS = RepeatWrapping;
            texture.wrapT = RepeatWrapping;
            texture.minFilter = LinearFilter;
            resolve(texture);
          },
          undefined,
          (error) => reject(error)
        );
      });

      const material = new MeshStandardMaterial({
        map: texture,
        bumpScale: 0.05,
        roughness: 0.8,
        metalness: 0.1,
        color: 0xffffff
      });
      material.needsUpdate = true;

      const mars = new Mesh(geometry, material);
      marsRef.current = mars;
      sceneRef.current.add(mars);

      // Create atmosphere if enabled
      if (showAtmosphere) {
        createAtmosphere();
      }

      // Setup lighting
      setupLighting();
    } catch (error) {
      console.error("Failed to load Mars texture:", error);
      setLoadError("High quality textures could not be loaded. Using basic visualization.");

      // Fallback to basic colored material
      const material = new MeshStandardMaterial({
        color: MARS_COLOR,
        roughness: 0.7,
        metalness: 0.3
      });

      const mars = new Mesh(geometry, material);
      marsRef.current = mars;
      sceneRef.current.add(mars);

      // Still create atmosphere and lighting even with fallback
      if (showAtmosphere) {
        createAtmosphere();
      }
      setupLighting();
    } finally {
      setIsLoading(false);
    }
  }, [showAtmosphere, createAtmosphere, setupLighting]);

  // Initialize scene
  const initScene = useCallback(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new Scene();
    scene.background = new Color(0x000000);
    sceneRef.current = scene;

    // Camera setup
    const aspectRatio = width / height;
    const camera = new PerspectiveCamera(60, aspectRatio, 0.1, MARS_RADIUS * 10);
    camera.position.z = MARS_RADIUS * 3;
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
      if (cameraRef.current) {
        const radius = cameraRef.current.position.length();
        const newPos = new Vector3();
        newPos.x = radius * Math.sin(targetRotationY) * Math.cos(targetRotationX);
        newPos.y = radius * Math.sin(targetRotationX);
        newPos.z = radius * Math.cos(targetRotationY) * Math.cos(targetRotationX);
        cameraRef.current.position.copy(newPos);
        cameraRef.current.lookAt(0, 0, 0);
      }

      mouseX = event.clientX;
      mouseY = event.clientY;
    };

    const handleMouseUp = () => {
      isMouseDown = false;
    };

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      if (!cameraRef.current) return;

      const scale = event.deltaY > 0 ? 1.1 : 0.9;
      cameraRef.current.position.multiplyScalar(scale);

      // Constrain zoom
      const distance = cameraRef.current.position.length();
      if (distance < MARS_RADIUS * 1.2) {
        cameraRef.current.position.normalize().multiplyScalar(MARS_RADIUS * 1.2);
      }
      if (distance > MARS_RADIUS * 8) {
        cameraRef.current.position.normalize().multiplyScalar(MARS_RADIUS * 8);
      }
    };

    // Add event listeners
    renderer.domElement.addEventListener('mousedown', handleMouseDown);
    renderer.domElement.addEventListener('mousemove', handleMouseMove);
    renderer.domElement.addEventListener('mouseup', handleMouseUp);
    renderer.domElement.addEventListener('wheel', handleWheel);
    mountRef.current.appendChild(renderer.domElement);

    // Store controls reference for cleanup
    controlsRef.current = {
      reset: () => {
        if (cameraRef.current) {
          cameraRef.current.position.set(0, 0, MARS_RADIUS * 3);
          cameraRef.current.lookAt(0, 0, 0);
        }
        targetRotationX = 0;
        targetRotationY = 0;
      },
      dispose: () => {
        renderer.domElement.removeEventListener('mousedown', handleMouseDown);
        renderer.domElement.removeEventListener('mousemove', handleMouseMove);
        renderer.domElement.removeEventListener('mouseup', handleMouseUp);
        renderer.domElement.removeEventListener('wheel', handleWheel);
      }
    };

    // Animation loop
    const animate = () => {
      if (!marsRef.current || !rendererRef.current || !sceneRef.current || !cameraRef.current) return;

      const delta = clock.current.getDelta();

      if (isRotating) {
        marsRef.current.rotation.y += ROTATION_SPEED * delta;
      }

      if (atmosphereRef.current && atmosphereRef.current.material instanceof ShaderMaterial) {
        atmosphereRef.current.material.uniforms.time.value += delta;
      }

      rendererRef.current.render(sceneRef.current, cameraRef.current);
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    // Cleanup function
    return () => {
      if (mountRef.current && rendererRef.current) {
        mountRef.current.removeChild(rendererRef.current.domElement);
      }
      controlsRef.current?.dispose();
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, [width, height, isRotating]);

  // Initialize everything
  useEffect(() => {
    const cleanup = initScene();
    createMarsGlobe();
    return cleanup;
  }, [initScene, createMarsGlobe]);

  // Update lighting when mode changes
  useEffect(() => {
    setupLighting();
  }, [lightingMode, setupLighting]);

  // Reset when props change
  useEffect(() => {
    if (marsRef.current) {
      marsRef.current.rotation.set(0, 0, 0);
    }
  }, [showAtmosphere, elevationScale]);

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
    if (controlsRef.current) {
      controlsRef.current.reset();
    }
  }, []);

  return (
    <div className="relative" style={{ width, height }}>
      {/* 3D Canvas Container */}
      <div ref={mountRef} className="w-full h-full bg-black rounded-lg overflow-hidden" />

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
          <div className="text-white text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-white border-t-transparent mx-auto mb-2" />
            <div>Loading Mars Surface...</div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {loadError && (
        <div className="absolute bottom-4 left-4 right-4 bg-red-900/90 text-white p-3 rounded-lg text-sm">
          <div className="font-semibold mb-1">Using simplified Mars model</div>
          <div className="text-red-200">{loadError}</div>
        </div>
      )}

      {/* Control Panel */}
      <div className="absolute top-4 right-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 space-y-2">
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsRotating(!isRotating)}
            className={`p-2 rounded ${isRotating ? 'bg-orange-600' : 'bg-gray-600'} text-white hover:opacity-80 transition-opacity`}
            title={isRotating ? 'Pause Rotation' : 'Start Rotation'}
          >
            {isRotating ? '⏸' : '▶️'}
          </button>

          <button
            onClick={resetCamera}
            className="p-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
            title="Reset Camera"
          >
            {'↺'}
          </button>

          <button
            onClick={() => setShowControls(!showControls)}
            className="p-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
            title="Toggle Settings"
          >
            {'⚙️'}
          </button>
        </div>

        {/* Extended Controls */}
        {showControls && (
          <div className="border-t border-gray-700 pt-2 space-y-2">
            <div className="text-xs text-gray-300 font-medium">Layer Controls</div>
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

      {/* Instructions */}
      <div className="absolute top-4 left-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-2">
        <div className="text-xs text-gray-300">
          <div>• Drag to rotate</div>
          <div>• Scroll to zoom</div>
          <div>• Click surface to select</div>
        </div>
      </div>

      {/* Location Info */}
      {selectedLocation && (
        <div className="absolute bottom-4 right-4 bg-gray-900/90 backdrop-blur-sm rounded-lg p-3">
          <div className="text-white">
            <div className="font-medium">{selectedLocation.name}</div>
            <div className="text-sm text-gray-300">
              {selectedLocation.lat.toFixed(4)}°, {selectedLocation.lon.toFixed(4)}°
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Mars3DGlobe;
