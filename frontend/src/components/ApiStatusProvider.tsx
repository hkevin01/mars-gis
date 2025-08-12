import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';

interface ApiStatusContextValue {
  apiUp: boolean | null; // null = unknown / not configured
  lastChecked: number | null;
  checking: boolean;
  retry: () => void;
  configured: boolean; // whether an API base is configured
}

const ApiStatusContext = createContext<ApiStatusContextValue | undefined>(undefined);

// Derive API base from REACT_APP_API_URL (expected like http://host:port/api/v1) or explicit REACT_APP_API_BASE
const RAW_API_BASE = (process.env.REACT_APP_API_URL || process.env.REACT_APP_API_BASE || '').replace(/\/$/, '');
// If a custom ping path is provided, use it; else append /health to base if we have one.
const API_PING_PATH = process.env.REACT_APP_API_PING || (RAW_API_BASE ? `${RAW_API_BASE}/health` : '');
const CHECK_INTERVAL_MS = 15000; // base interval; we add simple backoff on repeated failures

export const ApiStatusProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [apiUp, setApiUp] = useState<boolean | null>(null);
  const [checking, setChecking] = useState(false);
  const [lastChecked, setLastChecked] = useState<number | null>(null);
  const [failCount, setFailCount] = useState(0);

  const configured = !!API_PING_PATH;

  const check = useCallback(async () => {
    if (!configured) {
      // No API configured; treat as unknown and don't attempt network
      setApiUp(null);
      setLastChecked(Date.now());
      return;
    }
    setChecking(true);
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      const res = await fetch(API_PING_PATH, { signal: controller.signal });
      clearTimeout(timeout);
      setApiUp(res.ok);
      if (res.ok) setFailCount(0); else setFailCount(c => c + 1);
    } catch (_) {
      setApiUp(false);
      setFailCount(c => c + 1);
    } finally {
      setLastChecked(Date.now());
      setChecking(false);
    }
  }, [configured]);

  useEffect(() => {
    // Initial check
    check();
    let interval = CHECK_INTERVAL_MS;
    let id: NodeJS.Timeout;
    const schedule = () => {
      // Simple linear backoff after failures (cap at 60s)
      interval = Math.min(CHECK_INTERVAL_MS + failCount * 5000, 60000);
      id = setTimeout(async () => {
        await check();
        schedule();
      }, interval);
    };
    schedule();
    return () => clearTimeout(id);
  }, [check, failCount]);

  return (
    <ApiStatusContext.Provider value={{ apiUp, lastChecked, checking, retry: check, configured }}>
      {children}
    </ApiStatusContext.Provider>
  );
};

export const useApiStatus = () => {
  const ctx = useContext(ApiStatusContext);
  if (!ctx) throw new Error('useApiStatus must be used within ApiStatusProvider');
  return ctx;
};

export const ApiStatusBanner: React.FC = () => {
  const { apiUp, checking, retry, configured } = useApiStatus();
  if (apiUp) return null; // apiUp === true means healthy
  let message: string;
  if (!configured) {
    message = 'No API configured (REACT_APP_API_URL). Running in frontend-only mode.';
  } else if (apiUp === false) {
    message = 'Backend API unreachable. Some functionality may be limited.';
  } else {
    message = 'Checking backend status...';
  }
  return (
    <div style={{
      position: 'fixed', bottom: 0, left: 0, right: 0, zIndex: 1000,
      background: 'linear-gradient(90deg,#7f1d1d,#991b1b)', color: 'white',
      padding: '8px 16px', fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'space-between'
    }}>
      <span>{message}</span>
      {configured && (
        <button
          onClick={retry}
          disabled={checking}
          style={{
            background: '#dc2626', border: 'none', color: 'white', padding: '4px 10px',
            borderRadius: 4, cursor: 'pointer', opacity: checking ? 0.6 : 1
          }}
        >{checking ? 'Retrying...' : 'Retry'}</button>
      )}
    </div>
  );
};
