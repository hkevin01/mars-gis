// Simple error boundary fallback component
import React from 'react';

interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

export const GlobalErrorFallback: React.FC<ErrorFallbackProps> = ({ error, resetErrorBoundary }) => (
  <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
    <div className="max-w-md w-full bg-gray-800 rounded-lg shadow-lg p-6 text-center">
      <div className="text-red-400 text-6xl mb-4">⚠️</div>
      <h1 className="text-2xl font-bold text-white mb-2">Oops! Something went wrong</h1>
      <p className="text-gray-400 mb-4">
        {error.message || 'An unexpected error occurred in the Mars GIS application.'}
      </p>
      <button
        onClick={resetErrorBoundary}
        className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
      >
        Try Again
      </button>
    </div>
  </div>
);

export default GlobalErrorFallback;
