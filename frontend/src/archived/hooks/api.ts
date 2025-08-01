import { useCallback, useEffect, useRef, useState } from 'react';
import {
    AnalysisResult,
    analysisService,
    ApiError,
    Asset,
    assetService,
    CreateAnalysisRequest,
    CreateMissionRequest,
    dataService,
    Mission,
    missionService,
    PaginatedResponse,
    systemService,
    webSocketService
} from '../services/api';

// Generic API hook for handling loading states and errors
export function useApiCall<T>(
  apiCall: () => Promise<T>,
  dependencies: any[] = [],
  options: {
    immediate?: boolean;
    onSuccess?: (data: T) => void;
    onError?: (error: ApiError) => void;
  } = {}
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const { immediate = true, onSuccess, onError } = options;

  const execute = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await apiCall();
      setData(result);
      if (onSuccess) {
        onSuccess(result);
      }
    } catch (err) {
      const error = err as ApiError;
      setError(error);
      if (onError) {
        onError(error);
      }
    } finally {
      setLoading(false);
    }
  }, dependencies);

  useEffect(() => {
    if (immediate) {
      execute();
    }
  }, [execute, immediate]);

  return {
    data,
    loading,
    error,
    execute,
    refresh: execute,
  };
}

// Mission Management Hooks
export function useMissions(params?: {
  status?: string;
  asset?: string;
  page?: number;
  pageSize?: number;
}) {
  return useApiCall(
    () => missionService.getAll(params),
    [params?.status, params?.asset, params?.page, params?.pageSize]
  );
}

export function useMission(id: string | null) {
  return useApiCall(
    () => (id ? missionService.getById(id) : Promise.resolve(null)),
    [id],
    { immediate: !!id }
  );
}

export function useMissionActions() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const createMission = useCallback(async (mission: CreateMissionRequest): Promise<Mission | null> => {
    try {
      setLoading(true);
      setError(null);
      const result = await missionService.create(mission);
      return result;
    } catch (err) {
      setError(err as ApiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const updateMission = useCallback(async (id: string, mission: Partial<Mission>): Promise<Mission | null> => {
    try {
      setLoading(true);
      setError(null);
      const result = await missionService.update(id, mission);
      return result;
    } catch (err) {
      setError(err as ApiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const deleteMission = useCallback(async (id: string): Promise<boolean> => {
    try {
      setLoading(true);
      setError(null);
      await missionService.delete(id);
      return true;
    } catch (err) {
      setError(err as ApiError);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const startMission = useCallback(async (id: string): Promise<Mission | null> => {
    try {
      setLoading(true);
      setError(null);
      const result = await missionService.start(id);
      return result;
    } catch (err) {
      setError(err as ApiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const pauseMission = useCallback(async (id: string): Promise<Mission | null> => {
    try {
      setLoading(true);
      setError(null);
      const result = await missionService.pause(id);
      return result;
    } catch (err) {
      setError(err as ApiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const resumeMission = useCallback(async (id: string): Promise<Mission | null> => {
    try {
      setLoading(true);
      setError(null);
      const result = await missionService.resume(id);
      return result;
    } catch (err) {
      setError(err as ApiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const cancelMission = useCallback(async (id: string): Promise<Mission | null> => {
    try {
      setLoading(true);
      setError(null);
      const result = await missionService.cancel(id);
      return result;
    } catch (err) {
      setError(err as ApiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loading,
    error,
    createMission,
    updateMission,
    deleteMission,
    startMission,
    pauseMission,
    resumeMission,
    cancelMission,
  };
}

// Asset Management Hooks
export function useAssets() {
  return useApiCall(() => assetService.getAll(), []);
}

export function useAsset(id: string | null) {
  return useApiCall(
    () => (id ? assetService.getById(id) : Promise.resolve(null)),
    [id],
    { immediate: !!id }
  );
}

export function useAssetStatus(id: string | null, refreshInterval: number = 30000) {
  const intervalRef = useRef<NodeJS.Timeout>();
  const { data, loading, error, execute } = useApiCall(
    () => (id ? assetService.getStatus(id) : Promise.resolve(null)),
    [id],
    { immediate: !!id }
  );

  useEffect(() => {
    if (id && refreshInterval > 0) {
      intervalRef.current = setInterval(() => {
        execute();
      }, refreshInterval);

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }
  }, [id, refreshInterval, execute]);

  return { data, loading, error, refresh: execute };
}

export function useAssetActions() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const sendCommand = useCallback(async (
    id: string,
    command: string,
    parameters?: Record<string, any>
  ): Promise<boolean> => {
    try {
      setLoading(true);
      setError(null);
      await assetService.sendCommand(id, command, parameters);
      return true;
    } catch (err) {
      setError(err as ApiError);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loading,
    error,
    sendCommand,
  };
}

// Analysis Hooks
export function useAnalysisResults(params?: {
  type?: string;
  status?: string;
  region?: string;
  page?: number;
  pageSize?: number;
}) {
  return useApiCall(
    () => analysisService.getResults(params),
    [params?.type, params?.status, params?.region, params?.page, params?.pageSize]
  );
}

export function useAnalysisResult(id: string | null) {
  return useApiCall(
    () => (id ? analysisService.getById(id) : Promise.resolve(null)),
    [id],
    { immediate: !!id }
  );
}

export function useAnalysisActions() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const createAnalysis = useCallback(async (
    type: 'terrain' | 'atmospheric' | 'geological' | 'hazard',
    request: CreateAnalysisRequest
  ): Promise<AnalysisResult | null> => {
    try {
      setLoading(true);
      setError(null);
      
      let result: AnalysisResult;
      switch (type) {
        case 'terrain':
          result = await analysisService.createTerrain(request);
          break;
        case 'atmospheric':
          result = await analysisService.createAtmospheric(request);
          break;
        case 'geological':
          result = await analysisService.createGeological(request);
          break;
        case 'hazard':
          result = await analysisService.createHazard(request);
          break;
        default:
          throw new Error(`Unsupported analysis type: ${type}`);
      }
      
      return result;
    } catch (err) {
      setError(err as ApiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const cancelAnalysis = useCallback(async (id: string): Promise<boolean> => {
    try {
      setLoading(true);
      setError(null);
      await analysisService.cancel(id);
      return true;
    } catch (err) {
      setError(err as ApiError);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loading,
    error,
    createAnalysis,
    cancelAnalysis,
  };
}

// Data Hooks
export function useAtmosphericData(params?: {
  startDate?: string;
  endDate?: string;
  region?: string;
  limit?: number;
}) {
  return useApiCall(
    () => dataService.getAtmospheric(params),
    [params?.startDate, params?.endDate, params?.region, params?.limit]
  );
}

export function useGeologicalData(params?: {
  sampleType?: string;
  region?: string;
  limit?: number;
}) {
  return useApiCall(
    () => dataService.getGeological(params),
    [params?.sampleType, params?.region, params?.limit]
  );
}

export function useTerrainData(params?: {
  bounds?: [number, number, number, number];
  classification?: string;
  hazardLevel?: string;
  limit?: number;
}) {
  return useApiCall(
    () => dataService.getTerrain(params),
    [
      params?.bounds?.join(','),
      params?.classification,
      params?.hazardLevel,
      params?.limit,
    ]
  );
}

// System Hooks
export function useSystemStatus(refreshInterval: number = 60000) {
  const intervalRef = useRef<NodeJS.Timeout>();
  const { data, loading, error, execute } = useApiCall(
    () => systemService.getStatus(),
    []
  );

  useEffect(() => {
    if (refreshInterval > 0) {
      intervalRef.current = setInterval(() => {
        execute();
      }, refreshInterval);

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }
  }, [refreshInterval, execute]);

  return { data, loading, error, refresh: execute };
}

export function useSystemMetrics(refreshInterval: number = 30000) {
  const intervalRef = useRef<NodeJS.Timeout>();
  const { data, loading, error, execute } = useApiCall(
    () => systemService.getMetrics(),
    []
  );

  useEffect(() => {
    if (refreshInterval > 0) {
      intervalRef.current = setInterval(() => {
        execute();
      }, refreshInterval);

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }
  }, [refreshInterval, execute]);

  return { data, loading, error, refresh: execute };
}

export function useSystemLogs(params?: {
  level?: 'debug' | 'info' | 'warning' | 'error';
  limit?: number;
  since?: string;
}) {
  return useApiCall(
    () => systemService.getLogs(params),
    [params?.level, params?.limit, params?.since]
  );
}

// WebSocket Hooks
export function useWebSocket() {
  const [connected, setConnected] = useState(false);
  const listenersRef = useRef<Map<string, (data: any) => void>>(new Map());

  useEffect(() => {
    webSocketService.connect();
    
    // Monitor connection status
    const checkConnection = () => {
      setConnected(webSocketService['ws']?.readyState === WebSocket.OPEN);
    };
    
    const interval = setInterval(checkConnection, 1000);
    checkConnection();

    return () => {
      clearInterval(interval);
      webSocketService.disconnect();
    };
  }, []);

  const subscribe = useCallback((eventType: string, callback: (data: any) => void) => {
    webSocketService.subscribe(eventType, callback);
    listenersRef.current.set(eventType, callback);
  }, []);

  const unsubscribe = useCallback((eventType: string) => {
    const callback = listenersRef.current.get(eventType);
    if (callback) {
      webSocketService.unsubscribe(eventType, callback);
      listenersRef.current.delete(eventType);
    }
  }, []);

  useEffect(() => {
    return () => {
      // Cleanup all subscriptions
      listenersRef.current.forEach((callback, eventType) => {
        webSocketService.unsubscribe(eventType, callback);
      });
      listenersRef.current.clear();
    };
  }, []);

  return {
    connected,
    subscribe,
    unsubscribe,
  };
}

// Real-time Mission Updates Hook
export function useMissionUpdates(missionId: string | null) {
  const [mission, setMission] = useState<Mission | null>(null);
  const { subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    if (missionId) {
      const handleMissionUpdate = (data: Mission) => {
        if (data.id === missionId) {
          setMission(data);
        }
      };

      subscribe('mission_update', handleMissionUpdate);

      return () => {
        unsubscribe('mission_update');
      };
    }
  }, [missionId, subscribe, unsubscribe]);

  return mission;
}

// Real-time Asset Updates Hook
export function useAssetUpdates() {
  const [assets, setAssets] = useState<Record<string, Asset>>({});
  const { subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    const handleAssetUpdate = (data: Asset) => {
      setAssets(prev => ({
        ...prev,
        [data.id]: data,
      }));
    };

    subscribe('asset_update', handleAssetUpdate);

    return () => {
      unsubscribe('asset_update');
    };
  }, [subscribe, unsubscribe]);

  return assets;
}

// Real-time System Alerts Hook
export function useSystemAlerts() {
  const [alerts, setAlerts] = useState<Array<{
    id: string;
    type: 'info' | 'warning' | 'error' | 'success';
    message: string;
    timestamp: string;
    dismissed?: boolean;
  }>>([]);
  const { subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    const handleAlert = (data: any) => {
      setAlerts(prev => [data, ...prev.slice(0, 49)]); // Keep last 50 alerts
    };

    subscribe('system_alert', handleAlert);

    return () => {
      unsubscribe('system_alert');
    };
  }, [subscribe, unsubscribe]);

  const dismissAlert = useCallback((id: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === id ? { ...alert, dismissed: true } : alert
    ));
  }, []);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  return {
    alerts: alerts.filter(alert => !alert.dismissed),
    dismissAlert,
    clearAlerts,
  };
}

// Pagination Hook
export function usePagination<T>(
  fetchFunction: (page: number, pageSize: number) => Promise<PaginatedResponse<T>>,
  initialPageSize: number = 20
) {
  const [data, setData] = useState<T[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(initialPageSize);
  const [total, setTotal] = useState(0);
  const [hasNext, setHasNext] = useState(false);
  const [hasPrev, setHasPrev] = useState(false);

  const fetchData = useCallback(async (newPage: number = page, newPageSize: number = pageSize) => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetchFunction(newPage, newPageSize);
      setData(response.items);
      setTotal(response.total);
      setHasNext(response.has_next);
      setHasPrev(response.has_prev);
      setPage(newPage);
      setPageSize(newPageSize);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, fetchFunction]);

  useEffect(() => {
    fetchData();
  }, []);

  const nextPage = useCallback(() => {
    if (hasNext) {
      fetchData(page + 1, pageSize);
    }
  }, [hasNext, page, pageSize, fetchData]);

  const prevPage = useCallback(() => {
    if (hasPrev) {
      fetchData(page - 1, pageSize);
    }
  }, [hasPrev, page, pageSize, fetchData]);

  const goToPage = useCallback((newPage: number) => {
    fetchData(newPage, pageSize);
  }, [pageSize, fetchData]);

  const changePageSize = useCallback((newPageSize: number) => {
    fetchData(1, newPageSize);
  }, [fetchData]);

  return {
    data,
    loading,
    error,
    page,
    pageSize,
    total,
    hasNext,
    hasPrev,
    nextPage,
    prevPage,
    goToPage,
    changePageSize,
    refresh: () => fetchData(page, pageSize),
  };
}
