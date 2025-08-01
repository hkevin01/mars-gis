// API Configuration
export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1',
  TIMEOUT: 30000,
  RETRY_ATTEMPTS: 3,
};

// Request/Response Types
export interface ApiResponse<T = any> {
  data: T;
  message?: string;
  status: number;
  timestamp: string;
}

export interface ApiError {
  message: string;
  code: string;
  details?: any;
  status: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

// Mission Types
export interface Mission {
  id: string;
  name: string;
  description: string;
  status: 'planned' | 'active' | 'completed' | 'failed' | 'cancelled';
  asset_id: string;
  created_at: string;
  updated_at: string;
  start_date?: string;
  end_date?: string;
  progress: number;
  tasks: Task[];
  metadata: Record<string, any>;
}

export interface Task {
  id: string;
  name: string;
  type: 'navigation' | 'sampling' | 'analysis' | 'communication' | 'maintenance';
  status: 'pending' | 'active' | 'completed' | 'failed' | 'skipped';
  coordinates?: [number, number];
  parameters: Record<string, any>;
  created_at: string;
  updated_at: string;
  estimated_duration?: number;
  actual_duration?: number;
}

export interface CreateMissionRequest {
  name: string;
  description: string;
  asset_id: string;
  start_date?: string;
  tasks: Omit<Task, 'id' | 'created_at' | 'updated_at' | 'status'>[];
  metadata?: Record<string, any>;
}

// Asset Types
export interface Asset {
  id: string;
  name: string;
  type: 'rover' | 'lander' | 'orbiter';
  status: 'online' | 'offline' | 'maintenance' | 'charging';
  location: {
    latitude: number;
    longitude: number;
    altitude?: number;
  };
  battery_level: number;
  last_contact: string;
  capabilities: string[];
  metadata: Record<string, any>;
}

// Analysis Types
export interface AnalysisResult {
  id: string;
  type: 'terrain' | 'atmospheric' | 'geological' | 'hazard';
  status: 'queued' | 'processing' | 'completed' | 'failed';
  region: string;
  coordinates: {
    latitude: number;
    longitude: number;
    radius?: number;
  };
  parameters: Record<string, any>;
  results: Record<string, any>;
  created_at: string;
  completed_at?: string;
  confidence_score?: number;
}

export interface CreateAnalysisRequest {
  type: 'terrain' | 'atmospheric' | 'geological' | 'hazard';
  region: string;
  coordinates: {
    latitude: number;
    longitude: number;
    radius?: number;
  };
  parameters?: Record<string, any>;
}

// Data Types
export interface AtmosphericData {
  timestamp: string;
  temperature: number;
  pressure: number;
  humidity: number;
  wind_speed: number;
  wind_direction: number;
  dust_opacity: number;
  coordinates: [number, number];
}

export interface GeologicalData {
  id: string;
  sample_type: 'soil' | 'rock' | 'atmospheric';
  collection_date: string;
  coordinates: [number, number];
  composition: Record<string, number>;
  mineral_content: Record<string, number>;
  analysis_results: Record<string, any>;
  confidence_score: number;
}

export interface TerrainData {
  coordinates: [number, number];
  elevation: number;
  slope: number;
  surface_type: string;
  roughness: number;
  hazard_level: 'low' | 'medium' | 'high';
  classification: string;
  confidence_score: number;
}

// HTTP Client with error handling and retry logic
class ApiClient {
  private baseURL: string;
  private timeout: number;
  private retryAttempts: number;

  constructor() {
    this.baseURL = API_CONFIG.BASE_URL;
    this.timeout = API_CONFIG.TIMEOUT;
    this.retryAttempts = API_CONFIG.RETRY_ATTEMPTS;
  }

  private async request<T>(
    url: string,
    options: RequestInit = {},
    attempt: number = 1
  ): Promise<ApiResponse<T>> {
    const fullUrl = `${this.baseURL}${url}`;

    const defaultHeaders: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };

    // Add authentication token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      defaultHeaders['Authorization'] = `Bearer ${token}`;
    }

    const config: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    };

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(fullUrl, {
        ...config,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw {
          message: errorData.message || `HTTP ${response.status}: ${response.statusText}`,
          code: errorData.code || 'HTTP_ERROR',
          status: response.status,
          details: errorData,
        } as ApiError;
      }

      const data = await response.json();
      return {
        data,
        status: response.status,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      // Retry logic for network errors
      if (attempt < this.retryAttempts && this.shouldRetry(error)) {
        await this.delay(Math.pow(2, attempt) * 1000); // Exponential backoff
        return this.request<T>(url, options, attempt + 1);
      }

      // Handle different error types
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw {
            message: 'Request timeout',
            code: 'TIMEOUT',
            status: 408,
          } as ApiError;
        }

        if (error.message === 'Failed to fetch') {
          throw {
            message: 'Network error - please check your connection',
            code: 'NETWORK_ERROR',
            status: 0,
          } as ApiError;
        }
      }

      throw error;
    }
  }

  private shouldRetry(error: any): boolean {
    // Retry on network errors or server errors (5xx)
    return (
      error.message === 'Failed to fetch' ||
      error.name === 'AbortError' ||
      (error.status >= 500 && error.status < 600)
    );
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async get<T>(url: string): Promise<ApiResponse<T>> {
    return this.request<T>(url, { method: 'GET' });
  }

  async post<T>(url: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(url, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async put<T>(url: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(url, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T>(url: string): Promise<ApiResponse<T>> {
    return this.request<T>(url, { method: 'DELETE' });
  }

  async patch<T>(url: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(url, {
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    });
  }
}

// Create singleton instance
export const apiClient = new ApiClient();

// API Service Functions

// Mission Management
export const missionService = {
  async getAll(params?: {
    status?: string;
    asset?: string;
    page?: number;
    pageSize?: number;
  }): Promise<PaginatedResponse<Mission>> {
    const searchParams = new URLSearchParams();
    if (params?.status) searchParams.append('status', params.status);
    if (params?.asset) searchParams.append('asset', params.asset);
    if (params?.page) searchParams.append('page', params.page.toString());
    if (params?.pageSize) searchParams.append('page_size', params.pageSize.toString());

    const url = `/missions${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
    const response = await apiClient.get<PaginatedResponse<Mission>>(url);
    return response.data;
  },

  async getById(id: string): Promise<Mission> {
    const response = await apiClient.get<Mission>(`/missions/${id}`);
    return response.data;
  },

  async create(mission: CreateMissionRequest): Promise<Mission> {
    const response = await apiClient.post<Mission>('/missions', mission);
    return response.data;
  },

  async update(id: string, mission: Partial<Mission>): Promise<Mission> {
    const response = await apiClient.put<Mission>(`/missions/${id}`, mission);
    return response.data;
  },

  async delete(id: string): Promise<void> {
    await apiClient.delete(`/missions/${id}`);
  },

  async start(id: string): Promise<Mission> {
    const response = await apiClient.post<Mission>(`/missions/${id}/start`);
    return response.data;
  },

  async pause(id: string): Promise<Mission> {
    const response = await apiClient.post<Mission>(`/missions/${id}/pause`);
    return response.data;
  },

  async resume(id: string): Promise<Mission> {
    const response = await apiClient.post<Mission>(`/missions/${id}/resume`);
    return response.data;
  },

  async cancel(id: string): Promise<Mission> {
    const response = await apiClient.post<Mission>(`/missions/${id}/cancel`);
    return response.data;
  },
};

// Asset Management
export const assetService = {
  async getAll(): Promise<Asset[]> {
    const response = await apiClient.get<Asset[]>('/assets');
    return response.data;
  },

  async getById(id: string): Promise<Asset> {
    const response = await apiClient.get<Asset>(`/assets/${id}`);
    return response.data;
  },

  async getStatus(id: string): Promise<{ status: string; last_update: string }> {
    const response = await apiClient.get(`/assets/${id}/status`);
    return response.data;
  },

  async sendCommand(id: string, command: string, parameters?: Record<string, any>): Promise<void> {
    await apiClient.post(`/assets/${id}/commands`, { command, parameters });
  },
};

// Analysis Services
export const analysisService = {
  async getResults(params?: {
    type?: string;
    status?: string;
    region?: string;
    page?: number;
    pageSize?: number;
  }): Promise<PaginatedResponse<AnalysisResult>> {
    const searchParams = new URLSearchParams();
    if (params?.type) searchParams.append('type', params.type);
    if (params?.status) searchParams.append('status', params.status);
    if (params?.region) searchParams.append('region', params.region);
    if (params?.page) searchParams.append('page', params.page.toString());
    if (params?.pageSize) searchParams.append('page_size', params.pageSize.toString());

    const url = `/analysis/results${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
    const response = await apiClient.get<PaginatedResponse<AnalysisResult>>(url);
    return response.data;
  },

  async getById(id: string): Promise<AnalysisResult> {
    const response = await apiClient.get<AnalysisResult>(`/analysis/results/${id}`);
    return response.data;
  },

  async createTerrain(request: CreateAnalysisRequest): Promise<AnalysisResult> {
    const response = await apiClient.post<AnalysisResult>('/analysis/terrain', request);
    return response.data;
  },

  async createAtmospheric(request: CreateAnalysisRequest): Promise<AnalysisResult> {
    const response = await apiClient.post<AnalysisResult>('/analysis/atmospheric', request);
    return response.data;
  },

  async createGeological(request: CreateAnalysisRequest): Promise<AnalysisResult> {
    const response = await apiClient.post<AnalysisResult>('/analysis/geological', request);
    return response.data;
  },

  async createHazard(request: CreateAnalysisRequest): Promise<AnalysisResult> {
    const response = await apiClient.post<AnalysisResult>('/analysis/hazard', request);
    return response.data;
  },

  async cancel(id: string): Promise<void> {
    await apiClient.delete(`/analysis/results/${id}`);
  },
};

// Data Services
export const dataService = {
  async getAtmospheric(params?: {
    startDate?: string;
    endDate?: string;
    region?: string;
    limit?: number;
  }): Promise<AtmosphericData[]> {
    const searchParams = new URLSearchParams();
    if (params?.startDate) searchParams.append('start_date', params.startDate);
    if (params?.endDate) searchParams.append('end_date', params.endDate);
    if (params?.region) searchParams.append('region', params.region);
    if (params?.limit) searchParams.append('limit', params.limit.toString());

    const url = `/data/atmospheric${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
    const response = await apiClient.get<AtmosphericData[]>(url);
    return response.data;
  },

  async getGeological(params?: {
    sampleType?: string;
    region?: string;
    limit?: number;
  }): Promise<GeologicalData[]> {
    const searchParams = new URLSearchParams();
    if (params?.sampleType) searchParams.append('sample_type', params.sampleType);
    if (params?.region) searchParams.append('region', params.region);
    if (params?.limit) searchParams.append('limit', params.limit.toString());

    const url = `/data/geological${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
    const response = await apiClient.get<GeologicalData[]>(url);
    return response.data;
  },

  async getTerrain(params?: {
    bounds?: [number, number, number, number]; // [minLat, minLon, maxLat, maxLon]
    classification?: string;
    hazardLevel?: string;
    limit?: number;
  }): Promise<TerrainData[]> {
    const searchParams = new URLSearchParams();
    if (params?.bounds) searchParams.append('bounds', params.bounds.join(','));
    if (params?.classification) searchParams.append('classification', params.classification);
    if (params?.hazardLevel) searchParams.append('hazard_level', params.hazardLevel);
    if (params?.limit) searchParams.append('limit', params.limit.toString());

    const url = `/data/terrain${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
    const response = await apiClient.get<TerrainData[]>(url);
    return response.data;
  },

  async getMapTiles(layer: string, z: number, x: number, y: number): Promise<Blob> {
    const response = await fetch(`${API_CONFIG.BASE_URL}/data/tiles/${layer}/${z}/${x}/${y}.png`);
    if (!response.ok) {
      throw new Error(`Failed to fetch tile: ${response.statusText}`);
    }
    return response.blob();
  },
};

// System Services
export const systemService = {
  async getStatus(): Promise<{
    status: 'healthy' | 'degraded' | 'down';
    uptime: number;
    version: string;
    services: Record<string, 'healthy' | 'degraded' | 'down'>;
  }> {
    const response = await apiClient.get('/system/status');
    return response.data;
  },

  async getMetrics(): Promise<{
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    active_connections: number;
    request_rate: number;
  }> {
    const response = await apiClient.get('/system/metrics');
    return response.data;
  },

  async getLogs(params?: {
    level?: 'debug' | 'info' | 'warning' | 'error';
    limit?: number;
    since?: string;
  }): Promise<Array<{
    timestamp: string;
    level: string;
    message: string;
    component: string;
  }>> {
    const searchParams = new URLSearchParams();
    if (params?.level) searchParams.append('level', params.level);
    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.since) searchParams.append('since', params.since);

    const url = `/system/logs${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
    const response = await apiClient.get(url);
    return response.data;
  },
};

// WebSocket Service for Real-time Updates
export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  connect(): void {
    const wsUrl = API_CONFIG.BASE_URL.replace('http', 'ws') + '/ws';

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.reconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.reconnect();
    }
  }

  private reconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Attempting to reconnect WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect();
      }, this.reconnectDelay * Math.pow(2, this.reconnectAttempts));
    }
  }

  private handleMessage(message: { type: string; data: any }): void {
    const listeners = this.listeners.get(message.type);
    if (listeners) {
      listeners.forEach(callback => callback(message.data));
    }
  }

  subscribe(eventType: string, callback: (data: any) => void): void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    this.listeners.get(eventType)!.add(callback);
  }

  unsubscribe(eventType: string, callback: (data: any) => void): void {
    const listeners = this.listeners.get(eventType);
    if (listeners) {
      listeners.delete(callback);
      if (listeners.size === 0) {
        this.listeners.delete(eventType);
      }
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.listeners.clear();
  }
}

// Create singleton WebSocket service
export const webSocketService = new WebSocketService();
