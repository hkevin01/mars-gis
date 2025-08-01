import { useCallback, useEffect, useState } from 'react';

// Types
interface MarsDataQueryParams {
  region?: {
    minLat: number;
    maxLat: number;
    minLon: number;
    maxLon: number;
  };
  datasets?: string[];
  resolution?: number;
}

// Hook for Mars data querying
export const useMarsData = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const queryMarsData = useCallback(async (params: MarsDataQueryParams) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/mars-data/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setLoading(false);
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      setLoading(false);
      throw err;
    }
  }, []);

  return { queryMarsData, loading, error };
};

// Hook for ML model inference
export const useMLInference = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runInference = useCallback(async (modelType: string, inputData: any) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/inference/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_type: modelType,
          input_data: inputData,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setLoading(false);
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      setLoading(false);
      throw err;
    }
  }, []);

  const runBatchInference = useCallback(async (modelType: string, batchData: any[]) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/inference/batch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_type: modelType,
          batch_data: batchData,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setLoading(false);
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      setLoading(false);
      throw err;
    }
  }, []);

  return { runInference, runBatchInference, loading, error };
};

// Hook for mission planning
export const useMissionPlanning = () => {
  const [missions, setMissions] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchMissions = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/missions');

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setMissions(result.data || []);
      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch missions');
      setLoading(false);
    }
  }, []);

  const createMission = useCallback(async (missionData: any) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/missions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(missionData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setLoading(false);
      await fetchMissions(); // Refresh missions list
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create mission');
      setLoading(false);
      throw err;
    }
  }, [fetchMissions]);

  const updateMissionStatus = useCallback(async (missionId: string, status: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/v1/missions/${missionId}/status`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setLoading(false);
      await fetchMissions(); // Refresh missions list
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update mission status');
      setLoading(false);
      throw err;
    }
  }, [fetchMissions]);

  useEffect(() => {
    fetchMissions();
  }, [fetchMissions]);

  return {
    missions,
    loading,
    error,
    fetchMissions,
    createMission,
    updateMissionStatus,
  };
};

// Hook for real-time data streaming
export const useRealTimeData = (streamType: string) => {
  const [data, setData] = useState<any>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let eventSource: EventSource | null = null;

    const connectToStream = () => {
      try {
        eventSource = new EventSource(`/api/v1/streaming/${streamType}`);

        eventSource.onopen = () => {
          setConnected(true);
          setError(null);
        };

        eventSource.onmessage = (event) => {
          try {
            const newData = JSON.parse(event.data);
            setData(newData);
          } catch (err) {
            // Handle parsing error silently
            setError('Failed to parse streaming data');
          }
        };

        eventSource.onerror = () => {
          setConnected(false);
          setError('Connection to real-time stream failed');
        };
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to connect to stream');
      }
    };

    connectToStream();

    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [streamType]);

  return { data, connected, error };
};

// Hook for system health monitoring
export const useSystemHealth = () => {
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/health');

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setHealth(result);
      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Health check failed');
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, [checkHealth]);

  return { health, loading, error, checkHealth };
};

// Hook for dataset management
export const useDatasets = () => {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDatasets = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/mars-data/datasets');

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setDatasets(result.data || []);
      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch datasets');
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  return { datasets, loading, error, fetchDatasets };
};

// Utility function for error handling
export const handleAPIError = (error: any): string => {
  if (error?.response?.data?.message) {
    return error.response.data.message;
  }
  if (error?.message) {
    return error.message;
  }
  return 'An unexpected error occurred';
};
