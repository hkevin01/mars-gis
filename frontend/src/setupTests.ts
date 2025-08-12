// Test setup file for Mars-GIS Frontend
import '@testing-library/jest-dom';
import { server } from './__mocks__/server';

// Mock Service Worker setup
beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock IntersectionObserver
global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock WebGL context for Three.js / Cesium testing with minimal required methods
const webGLContextMock: Partial<WebGLRenderingContext> = {
  canvas: {} as any,
  drawingBufferWidth: 1024,
  drawingBufferHeight: 768,
  viewport: jest.fn(),
  clear: jest.fn(),
  clearColor: jest.fn(),
  enable: jest.fn(),
  disable: jest.fn(),
  getExtension: jest.fn(),
  getParameter: jest.fn(),
  createShader: jest.fn(),
  shaderSource: jest.fn(),
  compileShader: jest.fn(),
  createProgram: jest.fn(),
  attachShader: jest.fn(),
  linkProgram: jest.fn(),
  useProgram: jest.fn(),
  getShaderParameter: jest.fn(),
  getShaderInfoLog: jest.fn(),
  createBuffer: jest.fn(),
  bindBuffer: jest.fn(),
  bufferData: jest.fn(),
  drawArrays: jest.fn(),
};

(HTMLCanvasElement.prototype.getContext as any) = jest.fn((contextId: string) => {
  if (contextId === 'webgl' || contextId === 'webgl2') {
    return webGLContextMock as WebGLRenderingContext;
  }
  return null;
});

// Mock WebSocket (class with required static constants)
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  readyState = MockWebSocket.OPEN;
  url: string;
  protocol = '';
  binaryType: BinaryType = 'blob';
  onopen: ((this: WebSocket, ev: Event) => any) | null = null;
  onmessage: ((this: WebSocket, ev: MessageEvent) => any) | null = null;
  onclose: ((this: WebSocket, ev: CloseEvent) => any) | null = null;
  onerror: ((this: WebSocket, ev: Event) => any) | null = null;
  constructor(url: string | URL) { this.url = url.toString(); }
  send = jest.fn();
  close = jest.fn();
  addEventListener = jest.fn();
  removeEventListener = jest.fn();
  dispatchEvent = jest.fn();
}
// Override global WebSocket for test environment
(global as any).WebSocket = MockWebSocket as any;

// Mock localStorage & sessionStorage with full Storage interface shape
const createStorageMock = (): Storage => {
  const store = new Map<string, string>();
  return {
    length: 0,
    clear: jest.fn(() => { store.clear(); }),
    getItem: jest.fn((key: string) => store.get(key) ?? null),
    key: jest.fn((index: number) => Array.from(store.keys())[index] ?? null),
    removeItem: jest.fn((key: string) => { store.delete(key); }),
    setItem: jest.fn((key: string, value: string) => { store.set(key, value); }),
  } as unknown as Storage;
};
(global as any).localStorage = createStorageMock();
(global as any).sessionStorage = createStorageMock();

// Mock URL.createObjectURL
global.URL.createObjectURL = jest.fn(() => 'mocked-url');
global.URL.revokeObjectURL = jest.fn();

// Mock fetch if not already available
if (!global.fetch) {
  global.fetch = jest.fn();
}

// Console error/warning suppression for known issues
// eslint-disable-next-line no-console
const originalError = console.error;
// eslint-disable-next-line no-console
const originalWarn = console.warn;

beforeAll(() => {
  // eslint-disable-next-line no-console
  console.error = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('Warning: ReactDOM.render is deprecated')
    ) {
      return;
    }
    originalError.call(console, ...args);
  };

  // eslint-disable-next-line no-console
  console.warn = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('componentWillReceiveProps has been renamed')
    ) {
      return;
    }
    originalWarn.call(console, ...args);
  };
});

afterAll(() => {
  // eslint-disable-next-line no-console
  console.error = originalError;
  // eslint-disable-next-line no-console
  console.warn = originalWarn;
});

// Set up environment variables for testing
process.env.REACT_APP_API_URL = 'http://localhost:3001/api/v1';
process.env.REACT_APP_WS_URL = 'ws://localhost:3001';
