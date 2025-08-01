// Simple web vitals reporting
export const reportWebVitals = (onPerfEntry?: (metric: any) => void) => {
  if (onPerfEntry && onPerfEntry instanceof Function) {
    // In a real app, you might use web-vitals library here
    // For now, just a simple implementation
    console.log('Web vitals reporting enabled');
  }
};
