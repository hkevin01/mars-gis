// Cypress E2E Test Suite for Mars-GIS Platform
// Test-Driven Development - End-to-End User Workflows

describe('Mars-GIS Platform - Complete User Workflows', () => {
  beforeEach(() => {
    // Set up intercepts for API calls
    cy.intercept('GET', '/api/v1/dashboard/metrics', {
      fixture: 'dashboard-metrics.json'
    }).as('getDashboardMetrics');

    cy.intercept('GET', '/api/v1/system/alerts', {
      fixture: 'system-alerts.json'
    }).as('getSystemAlerts');

    cy.intercept('POST', '/api/v1/auth/login', {
      fixture: 'auth-response.json'
    }).as('login');

    // Visit the application
    cy.visit('/');
  });

  describe('Authentication Flow', () => {
    it('should allow user to login successfully', () => {
      // Navigate to login
      cy.get('[data-testid="login-button"]').click();

      // Fill login form
      cy.get('[data-testid="username-input"]').type('test_user');
      cy.get('[data-testid="password-input"]').type('test_password');
      cy.get('[data-testid="submit-login"]').click();

      // Verify successful login
      cy.wait('@login');
      cy.url().should('include', '/dashboard');
      cy.get('[data-testid="user-avatar"]').should('be.visible');
    });

    it('should handle login errors gracefully', () => {
      cy.intercept('POST', '/api/v1/auth/login', {
        statusCode: 401,
        body: { error: 'Invalid credentials' }
      }).as('loginError');

      cy.get('[data-testid="login-button"]').click();
      cy.get('[data-testid="username-input"]').type('invalid_user');
      cy.get('[data-testid="password-input"]').type('invalid_password');
      cy.get('[data-testid="submit-login"]').click();

      cy.wait('@loginError');
      cy.get('[data-testid="error-message"]').should('contain', 'Invalid credentials');
    });
  });

  describe('Dashboard Workflow', () => {
    beforeEach(() => {
      // Mock authentication
      cy.window().then((win) => {
        win.localStorage.setItem('auth_token', 'mock_token');
      });
      cy.visit('/dashboard');
    });

    it('should display dashboard with all metrics', () => {
      cy.wait('@getDashboardMetrics');

      // Verify dashboard components
      cy.get('[data-testid="dashboard-container"]').should('be.visible');
      cy.get('[data-testid="metrics-grid"]').should('be.visible');
      cy.get('[data-testid="mission-status-card"]').should('be.visible');
      cy.get('[data-testid="system-health-card"]').should('be.visible');
      cy.get('[data-testid="recent-analysis-card"]').should('be.visible');

      // Verify metrics display
      cy.get('[data-testid="active-missions-metric"]').should('contain', '3');
      cy.get('[data-testid="data-processing-metric"]').should('contain', '75%');
      cy.get('[data-testid="system-health-indicator"]').should('contain', 'Optimal');
    });

    it('should refresh dashboard data', () => {
      cy.get('[data-testid="refresh-dashboard"]').click();
      cy.wait('@getDashboardMetrics');

      // Verify loading indicator appears and disappears
      cy.get('[data-testid="loading-indicator"]').should('be.visible');
      cy.get('[data-testid="loading-indicator"]').should('not.exist');
    });

    it('should handle real-time updates', () => {
      // Mock WebSocket connection
      cy.window().then((win) => {
        const mockWs = {
          onmessage: cy.stub(),
          send: cy.stub(),
          close: cy.stub()
        };
        win.WebSocket = cy.stub().returns(mockWs);
      });

      // Verify real-time indicator
      cy.get('[data-testid="real-time-indicator"]').should('be.visible');
    });
  });

  describe('Mars 3D Viewer Workflow', () => {
    beforeEach(() => {
      cy.window().then((win) => {
        win.localStorage.setItem('auth_token', 'mock_token');
      });
      cy.visit('/mars-3d');
    });

    it('should load and display 3D Mars globe', () => {
      // Wait for 3D scene to initialize
      cy.get('[data-testid="mars-3d-container"]', { timeout: 10000 }).should('be.visible');
      cy.get('[data-testid="globe-canvas"]').should('be.visible');
      cy.get('[data-testid="viewer-controls"]').should('be.visible');

      // Verify loading completes
      cy.get('[data-testid="loading-indicator"]').should('not.exist');
      cy.get('[data-testid="mars-surface-mesh"]').should('exist');
    });

    it('should handle mouse interactions', () => {
      cy.get('[data-testid="globe-canvas"]').should('be.visible');

      // Test mouse interactions
      cy.get('[data-testid="globe-canvas"]')
        .trigger('mousedown', { clientX: 100, clientY: 100 })
        .trigger('mousemove', { clientX: 150, clientY: 150 })
        .trigger('mouseup');

      // Verify rotation state
      cy.get('[data-testid="globe-canvas"]').should('have.attr', 'data-rotating', 'true');
    });

    it('should support zoom controls', () => {
      // Test zoom in
      cy.get('[data-testid="zoom-in-button"]').click();
      cy.get('[data-testid="zoom-level-display"]').should('not.contain', '1.0');

      // Test zoom out
      cy.get('[data-testid="zoom-out-button"]').click();

      // Test zoom reset
      cy.get('[data-testid="zoom-reset-button"]').click();
      cy.get('[data-testid="zoom-level-display"]').should('contain', '1.0');
    });
  });

  describe('Interactive Map Workflow', () => {
    beforeEach(() => {
      cy.window().then((win) => {
        win.localStorage.setItem('auth_token', 'mock_token');
      });
      cy.visit('/interactive-map');
    });

    it('should display interactive Mars map', () => {
      cy.get('[data-testid="interactive-map-container"]').should('be.visible');
      cy.get('[data-testid="leaflet-map"]').should('be.visible');
      cy.get('[data-testid="map-controls"]').should('be.visible');

      // Verify coordinate system display
      cy.get('[data-testid="coordinate-display"]').should('be.visible');
      cy.contains('Mars Coordinate System').should('be.visible');
    });

    it('should handle layer toggling', () => {
      // Test elevation layer toggle
      cy.get('[data-testid="elevation-layer-toggle"]').click();
      cy.get('[data-testid="elevation-layer-toggle"]').should('have.attr', 'aria-pressed', 'true');

      // Test geological layer toggle
      cy.get('[data-testid="geological-layer-toggle"]').click();
      cy.get('[data-testid="geological-layer-toggle"]').should('have.attr', 'aria-pressed', 'true');
    });

    it('should support marker placement', () => {
      // Click on map to place marker
      cy.get('[data-testid="leaflet-map"]').click(300, 200);

      // Verify marker appears
      cy.get('[data-testid="new-marker"]').should('be.visible');

      // Test marker popup
      cy.get('[data-testid="new-marker"]').click();
      cy.get('[data-testid="marker-popup"]').should('be.visible');
    });
  });

  describe('Mission Planning Workflow', () => {
    beforeEach(() => {
      cy.window().then((win) => {
        win.localStorage.setItem('auth_token', 'mock_token');
      });
      cy.visit('/mission-planner');
    });

    it('should create a new mission plan', () => {
      cy.get('[data-testid="mission-planner-container"]').should('be.visible');
      cy.get('[data-testid="mission-form"]').should('be.visible');

      // Fill mission form
      cy.get('[data-testid="mission-name-input"]').type('Test Mission E2E');
      cy.get('[data-testid="mission-type-select"]').select('landing');
      cy.get('[data-testid="target-coordinates-lat"]').type('-14.5684');
      cy.get('[data-testid="target-coordinates-lon"]').type('175.4726');
      cy.get('[data-testid="mission-duration"]').type('687');

      // Set constraints
      cy.get('[data-testid="max-slope-input"]').type('15');
      cy.get('[data-testid="min-elevation-input"]').type('-1000');

      // Submit mission
      cy.get('[data-testid="submit-mission-button"]').click();

      // Verify mission creation
      cy.get('[data-testid="mission-success-message"]').should('be.visible');
      cy.url().should('include', '/missions/');
    });

    it('should validate mission parameters', () => {
      // Try to submit empty form
      cy.get('[data-testid="submit-mission-button"]').click();

      // Verify validation errors
      cy.get('[data-testid="mission-name-error"]').should('contain', 'Mission name is required');
      cy.get('[data-testid="coordinates-error"]').should('contain', 'Target coordinates are required');
    });

    it('should optimize landing sites', () => {
      // First create a basic mission
      cy.get('[data-testid="mission-name-input"]').type('Optimization Test');
      cy.get('[data-testid="mission-type-select"]').select('landing');
      cy.get('[data-testid="target-coordinates-lat"]').type('-14.5684');
      cy.get('[data-testid="target-coordinates-lon"]').type('175.4726');

      // Run optimization
      cy.get('[data-testid="optimize-landing-sites-button"]').click();

      // Verify optimization results
      cy.get('[data-testid="optimization-results"]', { timeout: 30000 }).should('be.visible');
      cy.get('[data-testid="optimal-site-1"]').should('be.visible');
      cy.get('[data-testid="optimization-score"]').should('be.visible');
    });
  });

  describe('Data Analysis Workflow', () => {
    beforeEach(() => {
      cy.window().then((win) => {
        win.localStorage.setItem('auth_token', 'mock_token');
      });
      cy.visit('/data-analysis');
    });

    it('should load and analyze datasets', () => {
      cy.get('[data-testid="data-analysis-container"]').should('be.visible');

      // Select dataset
      cy.get('[data-testid="dataset-selector"]').select('mars-elevation');

      // Wait for dataset preview
      cy.get('[data-testid="dataset-preview"]', { timeout: 10000 }).should('be.visible');

      // Run analysis
      cy.get('[data-testid="run-analysis-button"]').click();

      // Verify analysis progress
      cy.get('[data-testid="analysis-progress"]').should('be.visible');

      // Wait for results
      cy.get('[data-testid="analysis-results"]', { timeout: 30000 }).should('be.visible');
    });

    it('should display analysis results', () => {
      // Mock analysis completion
      cy.intercept('POST', '/api/v1/analysis/run', {
        body: { analysis_id: 'test_analysis_123', status: 'completed' }
      }).as('runAnalysis');

      cy.intercept('GET', '/api/v1/analysis/test_analysis_123/results', {
        fixture: 'analysis-results.json'
      }).as('getResults');

      // Select dataset and run analysis
      cy.get('[data-testid="dataset-selector"]').select('mars-geological');
      cy.get('[data-testid="run-analysis-button"]').click();

      cy.wait('@runAnalysis');
      cy.wait('@getResults');

      // Verify results display
      cy.get('[data-testid="results-chart"]').should('be.visible');
      cy.get('[data-testid="results-table"]').should('be.visible');
      cy.get('[data-testid="download-results-button"]').should('be.visible');
    });
  });

  describe('Complete User Journey', () => {
    it('should complete full analysis workflow', () => {
      // 1. Login
      cy.window().then((win) => {
        win.localStorage.setItem('auth_token', 'mock_token');
      });

      // 2. Start from dashboard
      cy.visit('/dashboard');
      cy.wait('@getDashboardMetrics');

      // 3. Navigate to data analysis
      cy.get('[data-testid="nav-data-analysis"]').click();
      cy.url().should('include', '/data-analysis');

      // 4. Upload data
      const fileName = 'test-mars-data.tif';
      cy.get('[data-testid="file-upload-input"]').selectFile({
        contents: Cypress.Buffer.from('mock file content'),
        fileName,
        mimeType: 'image/tiff'
      }, { force: true });

      // 5. Run analysis
      cy.get('[data-testid="run-analysis-button"]').click();
      cy.get('[data-testid="analysis-progress"]').should('be.visible');

      // 6. View results
      cy.get('[data-testid="analysis-results"]', { timeout: 30000 }).should('be.visible');

      // 7. Create mission from results
      cy.get('[data-testid="create-mission-from-analysis"]').click();
      cy.url().should('include', '/mission-planner');

      // 8. Complete mission planning
      cy.get('[data-testid="mission-name-input"]').should('have.value', 'Analysis Mission');
      cy.get('[data-testid="submit-mission-button"]').click();

      // 9. Verify mission creation
      cy.get('[data-testid="mission-success-message"]').should('be.visible');
    });
  });

  describe('Performance Tests', () => {
    it('should load dashboard within performance threshold', () => {
      const startTime = Date.now();

      cy.window().then((win) => {
        win.localStorage.setItem('auth_token', 'mock_token');
      });

      cy.visit('/dashboard');
      cy.wait('@getDashboardMetrics');

      cy.get('[data-testid="dashboard-container"]').should('be.visible').then(() => {
        const loadTime = Date.now() - startTime;
        expect(loadTime).to.be.lessThan(3000); // 3 second threshold
      });
    });

    it('should handle large datasets without blocking UI', () => {
      cy.window().then((win) => {
        win.localStorage.setItem('auth_token', 'mock_token');
      });

      cy.visit('/data-analysis');

      // Mock large dataset
      cy.intercept('GET', '/api/v1/datasets/large-mars-data', {
        fixture: 'large-dataset.json'
      }).as('getLargeDataset');

      cy.get('[data-testid="dataset-selector"]').select('large-mars-data');
      cy.wait('@getLargeDataset');

      // UI should remain responsive
      cy.get('[data-testid="loading-indicator"]').should('be.visible');
      cy.get('[data-testid="cancel-loading-button"]').should('be.visible');

      // Should eventually load
      cy.get('[data-testid="dataset-preview"]', { timeout: 10000 }).should('be.visible');
    });
  });

  describe('Error Handling', () => {
    it('should handle API errors gracefully', () => {
      // Mock API error
      cy.intercept('GET', '/api/v1/dashboard/metrics', {
        statusCode: 500,
        body: { error: 'Internal server error' }
      }).as('getMetricsError');

      cy.window().then((win) => {
        win.localStorage.setItem('auth_token', 'mock_token');
      });

      cy.visit('/dashboard');
      cy.wait('@getMetricsError');

      // Verify error handling
      cy.get('[data-testid="error-boundary"]').should('be.visible');
      cy.get('[data-testid="retry-button"]').should('be.visible');

      // Test retry functionality
      cy.intercept('GET', '/api/v1/dashboard/metrics', {
        fixture: 'dashboard-metrics.json'
      }).as('getMetricsRetry');

      cy.get('[data-testid="retry-button"]').click();
      cy.wait('@getMetricsRetry');

      cy.get('[data-testid="dashboard-container"]').should('be.visible');
    });

    it('should handle network connectivity issues', () => {
      cy.window().then((win) => {
        win.localStorage.setItem('auth_token', 'mock_token');
      });

      // Simulate network failure
      cy.intercept('GET', '/api/v1/dashboard/metrics', { forceNetworkError: true }).as('networkError');

      cy.visit('/dashboard');
      cy.wait('@networkError');

      // Verify offline indicator
      cy.get('[data-testid="offline-indicator"]').should('be.visible');
      cy.get('[data-testid="offline-message"]').should('contain', 'Connection lost');
    });
  });
});

// Accessibility tests
describe('Accessibility Compliance', () => {
  beforeEach(() => {
    cy.window().then((win) => {
      win.localStorage.setItem('auth_token', 'mock_token');
    });
  });

  it('should pass accessibility audit on dashboard', () => {
    cy.visit('/dashboard');
    cy.injectAxe();
    cy.checkA11y();
  });

  it('should support keyboard navigation', () => {
    cy.visit('/dashboard');

    // Test Tab navigation
    cy.get('body').tab();
    cy.focused().should('have.attr', 'data-testid', 'skip-to-content');

    cy.focused().tab();
    cy.focused().should('have.attr', 'data-testid', 'main-navigation');
  });

  it('should have proper ARIA labels', () => {
    cy.visit('/dashboard');

    cy.get('[role="main"]').should('exist');
    cy.get('[role="navigation"]').should('exist');
    cy.get('[aria-label="Dashboard metrics"]').should('exist');
  });
});
