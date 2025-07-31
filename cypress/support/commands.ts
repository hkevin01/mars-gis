// Custom Cypress Commands for Mars-GIS Testing
// Implementation of all custom commands declared in e2e.ts

// Authentication Commands
Cypress.Commands.add('login', (email = 'test@marsgis.com', password = 'password123') => {
  cy.visit('/login')
  cy.get('[data-testid="email-input"]').type(email)
  cy.get('[data-testid="password-input"]').type(password)
  cy.get('[data-testid="login-button"]').click()
  cy.url().should('not.include', '/login')
  cy.get('[data-testid="user-menu"]').should('be.visible')
})

Cypress.Commands.add('logout', () => {
  cy.get('[data-testid="user-menu"]').click()
  cy.get('[data-testid="logout-button"]').click()
  cy.url().should('include', '/login')
})

// Navigation Commands
Cypress.Commands.add('navigateToMarsCoordinates', (lat: number, lng: number) => {
  cy.get('[data-testid="coordinate-input-lat"]').clear().type(lat.toString())
  cy.get('[data-testid="coordinate-input-lng"]').clear().type(lng.toString())
  cy.get('[data-testid="navigate-button"]').click()
  cy.get('[data-testid="mars-globe"]').should('be.visible')
})

Cypress.Commands.add('waitForMarsGlobe', () => {
  cy.get('[data-testid="mars-globe"]', { timeout: 30000 }).should('be.visible')
  cy.get('[data-testid="globe-loading"]').should('not.exist')
  // Wait for Three.js to finish rendering
  cy.wait(2000)
})

// Accessibility Commands
Cypress.Commands.add('checkA11y', () => {
  cy.injectAxe()
  cy.checkA11y(null, {
    includedImpacts: ['critical', 'serious'],
    rules: {
      'color-contrast': { enabled: true },
      'keyboard-navigation': { enabled: true },
      'focus-management': { enabled: true }
    }
  })
})

// Data Management Commands
Cypress.Commands.add('seedTestData', (dataType: string) => {
  cy.task('seedTestData', dataType)
})

Cypress.Commands.add('resetAppState', () => {
  // Clear local storage and session storage
  cy.clearLocalStorage()
  cy.clearCookies()

  // Reset database to clean state
  cy.task('resetTestDb', null, { timeout: 30000 })

  // Clear any cached API responses
  if (window.caches) {
    cy.window().then(async (win) => {
      const cacheNames = await win.caches.keys()
      await Promise.all(
        cacheNames.map(cacheName => win.caches.delete(cacheName))
      )
    })
  }
})

// Performance Commands
Cypress.Commands.add('takePerformanceSnapshot', (label: string) => {
  cy.window().then((win) => {
    const metrics = {
      label,
      timestamp: Date.now(),
      navigation: win.performance.getEntriesByType('navigation')[0],
      resources: win.performance.getEntriesByType('resource').length,
      memory: (win.performance as any).memory ? {
        usedJSHeapSize: (win.performance as any).memory.usedJSHeapSize,
        totalJSHeapSize: (win.performance as any).memory.totalJSHeapSize,
        jsHeapSizeLimit: (win.performance as any).memory.jsHeapSizeLimit
      } : null
    }

    cy.task('recordPerformance', metrics)
  })
})

// API Mocking Commands
Cypress.Commands.add('mockApiResponse', (endpoint: string, response: any) => {
  cy.intercept('GET', `**/api/v1${endpoint}`, response).as(`mock-${endpoint.replace(/[^a-zA-Z0-9]/g, '-')}`)
})

// Additional helper commands for Mars-GIS specific functionality

// Mission Planning Commands
Cypress.Commands.add('createMissionPlan', (missionData: any) => {
  cy.visit('/mission-planner')
  cy.get('[data-testid="new-mission-button"]').click()

  // Fill mission details
  cy.get('[data-testid="mission-name-input"]').type(missionData.name)
  cy.get('[data-testid="mission-description-input"]').type(missionData.description)

  // Set mission coordinates
  missionData.waypoints.forEach((waypoint: any, index: number) => {
    cy.get('[data-testid="add-waypoint-button"]').click()
    cy.get(`[data-testid="waypoint-${index}-lat"]`).type(waypoint.lat.toString())
    cy.get(`[data-testid="waypoint-${index}-lng"]`).type(waypoint.lng.toString())
  })

  cy.get('[data-testid="save-mission-button"]').click()
  cy.get('[data-testid="mission-created-notification"]').should('be.visible')
})

// Data Analysis Commands
Cypress.Commands.add('uploadDataFile', (filePath: string) => {
  cy.get('[data-testid="file-upload-input"]').selectFile(filePath)
  cy.get('[data-testid="upload-progress"]').should('be.visible')
  cy.get('[data-testid="upload-complete"]', { timeout: 60000 }).should('be.visible')
})

Cypress.Commands.add('runAnalysisWorkflow', (workflowType: string) => {
  cy.get(`[data-testid="workflow-${workflowType}"]`).click()
  cy.get('[data-testid="run-analysis-button"]').click()
  cy.get('[data-testid="analysis-running"]').should('be.visible')
  cy.get('[data-testid="analysis-complete"]', { timeout: 120000 }).should('be.visible')
})

// Visualization Commands
Cypress.Commands.add('switchVisualizationMode', (mode: '2d' | '3d') => {
  cy.get(`[data-testid="view-mode-${mode}"]`).click()
  if (mode === '3d') {
    cy.waitForMarsGlobe()
  } else {
    cy.get('[data-testid="leaflet-map"]').should('be.visible')
  }
})

Cypress.Commands.add('toggleLayer', (layerName: string) => {
  cy.get('[data-testid="layer-panel"]').click()
  cy.get(`[data-testid="layer-${layerName}-toggle"]`).click()
  cy.get('[data-testid="layer-panel"]').click() // Close panel
})
