// WebXR VR Interface for Mars Exploration
import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Eye,
  EyeOff,
  Smartphone,
  Headphones
} from 'lucide-react';

// Import WebXR types
import '../types/webxr.d.ts';

interface VRMarsExplorerProps {
  marsGlobeRef?: React.RefObject<any>;
  onVRStateChange?: (isVRActive: boolean) => void;
  selectedLocation?: { lat: number; lon: number; name: string } | null;
}

interface VRControlsState {
  isVRSupported: boolean;
  isVRActive: boolean;
  isLoading: boolean;
  error: string | null;
  deviceType: 'headset' | 'mobile' | 'unknown';
}

const VRMarsExplorer: React.FC<VRMarsExplorerProps> = ({
  marsGlobeRef,
  onVRStateChange,
  selectedLocation
}) => {
  const xrSessionRef = useRef<XRSession | null>(null);
  const animationIdRef = useRef<number | null>(null);

  const [vrState, setVRState] = useState<VRControlsState>({
    isVRSupported: false,
    isVRActive: false,
    isLoading: false,
    error: null,
    deviceType: 'unknown'
  });

  // Check WebXR support on mount
  useEffect(() => {
    const checkVRSupport = async () => {
      if (!navigator.xr) {
        setVRState(prev => ({
          ...prev,
          error: 'WebXR not supported in this browser'
        }));
        return;
      }

      try {
        // Check for immersive VR support
        const isVRSupported = await navigator.xr.isSessionSupported('immersive-vr');
        const isARSupported = await navigator.xr.isSessionSupported('immersive-ar');

        // Detect device type
        let deviceType: 'headset' | 'mobile' | 'unknown' = 'unknown';
        if (isVRSupported) {
          deviceType = 'headset';
        } else if (isARSupported || /Mobi|Android/i.test(navigator.userAgent)) {
          deviceType = 'mobile';
        }

        setVRState(prev => ({
          ...prev,
          isVRSupported: isVRSupported || isARSupported,
          deviceType
        }));
      } catch (error) {
        setVRState(prev => ({
          ...prev,
          error: `WebXR error: ${error instanceof Error ? error.message : 'Unknown error'}`
        }));
      }
    };

    checkVRSupport();
  }, []);

  // Update Mars globe for VR viewing
  const updateMarsGlobeForVR = useCallback((frame: XRFrame, referenceSpace: XRReferenceSpace) => {
    const pose = frame.getViewerPose(referenceSpace);
    if (!pose || !marsGlobeRef?.current) return;

    const marsGlobe = marsGlobeRef.current;
    const position = pose.transform.position;

    // Scale Mars for comfortable VR viewing
    const vrScale = 0.5;
    marsGlobe.scale?.set(vrScale, vrScale, vrScale);

    // Position Mars in front of user
    marsGlobe.position?.set(
      position.x,
      position.y - 1,
      position.z - 3
    );
  }, [marsGlobeRef]);

  // Initialize VR session
  const startVRSession = useCallback(async () => {
    if (!navigator.xr || !vrState.isVRSupported) {
      setVRState(prev => ({ ...prev, error: 'VR not supported' }));
      return;
    }

    setVRState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const sessionType = vrState.deviceType === 'mobile' ? 'immersive-ar' : 'immersive-vr';
      const session = await navigator.xr.requestSession(sessionType, {
        requiredFeatures: ['local'],
        optionalFeatures: ['bounded-floor', 'hand-tracking']
      });

      xrSessionRef.current = session;
      const referenceSpace = await session.requestReferenceSpace('local');

      const onXRFrame = (time: number, frame: XRFrame) => {
        const currentSession = frame.session;

        if (currentSession && currentSession === xrSessionRef.current) {
          if (marsGlobeRef?.current) {
            updateMarsGlobeForVR(frame, referenceSpace);
          }
          animationIdRef.current = currentSession.requestAnimationFrame(onXRFrame);
        }
      };

      animationIdRef.current = session.requestAnimationFrame(onXRFrame);

      session.addEventListener('end', () => {
        if (animationIdRef.current) {
          session.cancelAnimationFrame(animationIdRef.current);
          animationIdRef.current = null;
        }
        xrSessionRef.current = null;
        setVRState(prev => ({ ...prev, isVRActive: false, isLoading: false }));
        onVRStateChange?.(false);
      });

      setVRState(prev => ({ ...prev, isVRActive: true, isLoading: false }));
      onVRStateChange?.(true);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setVRState(prev => ({
        ...prev,
        isLoading: false,
        error: `Failed to start VR: ${errorMessage}`
      }));
    }
  }, [vrState.isVRSupported, vrState.deviceType, marsGlobeRef, onVRStateChange, updateMarsGlobeForVR]);

  // End VR session
  const endVRSession = useCallback(async () => {
    if (xrSessionRef.current) {
      try {
        await xrSessionRef.current.end();
      } catch {
        // Session may already be ended - ignore error
      }
    }
  }, []);

  // Get appropriate icon for device type
  const getDeviceIcon = () => {
    switch (vrState.deviceType) {
      case 'headset':
        return <Headphones className="w-4 h-4" />;
      case 'mobile':
        return <Smartphone className="w-4 h-4" />;
      default:
        return <Eye className="w-4 h-4" />;
    }
  };

  // Get device-specific label
  const getDeviceLabel = () => {
    switch (vrState.deviceType) {
      case 'headset':
        return 'Enter VR';
      case 'mobile':
        return 'AR Mode';
      default:
        return 'Immersive View';
    }
  };

  return (
    <div className="vr-mars-explorer">
      {/* VR Controls Panel */}
      <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 space-y-2">
        {/* VR Status Display */}
        <div className="text-xs text-gray-300 font-medium">
          {vrState.deviceType === 'headset' ? 'VR Experience' : 'AR Experience'}
        </div>

        {/* Main VR Button */}
        {vrState.isVRSupported ? (
          <button
            onClick={vrState.isVRActive ? endVRSession : startVRSession}
            disabled={vrState.isLoading}
            className={`flex items-center space-x-2 w-full p-2 rounded text-white font-medium transition-colors ${
              vrState.isVRActive
                ? 'bg-red-600 hover:bg-red-700'
                : 'bg-purple-600 hover:bg-purple-700'
            } ${vrState.isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
            title={vrState.isVRActive ? 'Exit VR' : getDeviceLabel()}
          >
            {vrState.isLoading && (
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
            )}
            {!vrState.isLoading && vrState.isVRActive && <EyeOff className="w-4 h-4" />}
            {!vrState.isLoading && !vrState.isVRActive && getDeviceIcon()}
            <span>
              {vrState.isLoading && 'Initializing...'}
              {!vrState.isLoading && vrState.isVRActive && 'Exit VR'}
              {!vrState.isLoading && !vrState.isVRActive && getDeviceLabel()}
            </span>
          </button>
        ) : (
          <div className="bg-gray-700 text-gray-300 p-2 rounded text-sm text-center">
            VR/AR not available
          </div>
        )}

        {/* VR Status Indicators */}
        {vrState.isVRActive && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-green-400">● VR Active</span>
              <span className="text-gray-400">{vrState.deviceType}</span>
            </div>

            {selectedLocation && (
              <div className="bg-purple-900/30 border border-purple-600/50 rounded p-2">
                <div className="text-xs text-purple-300">Viewing:</div>
                <div className="text-sm text-white font-medium">
                  {selectedLocation.name}
                </div>
                <div className="text-xs text-purple-200">
                  {selectedLocation.lat.toFixed(4)}°, {selectedLocation.lon.toFixed(4)}°
                </div>
              </div>
            )}
          </div>
        )}

        {/* Error Display */}
        {vrState.error && (
          <div className="bg-red-900/30 border border-red-600/50 rounded p-2">
            <div className="text-xs text-red-300">Error:</div>
            <div className="text-sm text-red-200">{vrState.error}</div>
          </div>
        )}

        {/* VR Instructions */}
        {vrState.isVRSupported && !vrState.isVRActive && (
          <div className="text-xs text-gray-400 space-y-1">
            <div className="font-medium text-gray-300">Instructions:</div>
            {vrState.deviceType === 'headset' ? (
              <>
                <div>• Put on VR headset</div>
                <div>• Use controllers to interact</div>
                <div>• Look around to explore Mars</div>
              </>
            ) : (
              <>
                <div>• Hold device steady</div>
                <div>• Move to explore Mars</div>
                <div>• Tap to select locations</div>
              </>
            )}
          </div>
        )}

        {/* Browser Compatibility */}
        {!vrState.isVRSupported && (
          <div className="text-xs text-gray-400 space-y-1">
            <div className="text-gray-300 font-medium">Requirements:</div>
            <div>• Chrome/Edge with WebXR</div>
            <div>• VR headset or mobile device</div>
            <div>• Enable experimental features</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VRMarsExplorer;
