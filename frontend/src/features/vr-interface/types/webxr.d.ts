// WebXR Type Definitions for Mars GIS
// This extends the standard WebXR types with additional properties we need

declare global {
  interface Navigator {
    xr?: XRSystem;
  }

  interface XRSystem {
    isSessionSupported(mode: XRSessionMode): Promise<boolean>;
    requestSession(mode: XRSessionMode, options?: XRSessionInit): Promise<XRSession>;
  }

  type XRSessionMode = 'inline' | 'immersive-vr' | 'immersive-ar';

  interface XRSessionInit {
    requiredFeatures?: string[];
    optionalFeatures?: string[];
  }

  interface XRSession extends EventTarget {
    inputSources: XRInputSource[];
    requestReferenceSpace(type: XRReferenceSpaceType): Promise<XRReferenceSpace>;
    requestAnimationFrame(callback: XRFrameRequestCallback): number;
    cancelAnimationFrame(handle: number): void;
    end(): Promise<void>;
    renderState: XRRenderState;
    updateRenderState(state?: XRRenderStateInit): void;
  }

  interface XRInputSource {
    handedness: XRHandedness;
    targetRayMode: XRTargetRayMode;
    targetRaySpace: XRSpace;
    gripSpace?: XRSpace;
    gamepad?: Gamepad;
    profiles: string[];
  }

  type XRHandedness = 'none' | 'left' | 'right';
  type XRTargetRayMode = 'gaze' | 'tracked-pointer' | 'screen';

  interface XRFrame {
    session: XRSession;
    getViewerPose(referenceSpace: XRReferenceSpace): XRViewerPose | null;
    getPose(space: XRSpace, baseSpace: XRSpace): XRPose | null;
  }

  interface XRViewerPose {
    transform: XRRigidTransform;
    views: XRView[];
  }

  interface XRPose {
    transform: XRRigidTransform;
  }

  interface XRRigidTransform {
    position: DOMPointReadOnly;
    orientation: DOMPointReadOnly;
    matrix: Float32Array;
    inverse: XRRigidTransform;
  }

  interface XRView {
    eye: XREye;
    projectionMatrix: Float32Array;
    transform: XRRigidTransform;
  }

  type XREye = 'left' | 'right' | 'none';

  interface XRReferenceSpace extends XRSpace {
    getOffsetReferenceSpace(originOffset: XRRigidTransform): XRReferenceSpace;
  }

  interface XRSpace extends EventTarget {}

  type XRReferenceSpaceType = 'viewer' | 'local' | 'local-floor' | 'bounded-floor' | 'unbounded';

  interface XRRenderState {
    baseLayer?: XRWebGLLayer;
    depthFar: number;
    depthNear: number;
    inlineVerticalFieldOfView?: number;
  }

  interface XRRenderStateInit {
    baseLayer?: XRWebGLLayer;
    depthFar?: number;
    depthNear?: number;
    inlineVerticalFieldOfView?: number;
  }

  interface XRWebGLLayer {
    framebuffer: WebGLFramebuffer | null;
    framebufferWidth: number;
    framebufferHeight: number;
    getViewport(view: XRView): XRViewport;
  }

  interface XRViewport {
    x: number;
    y: number;
    width: number;
    height: number;
  }

  type XRFrameRequestCallback = (time: DOMHighResTimeStamp, frame: XRFrame) => void;
}

export {};
