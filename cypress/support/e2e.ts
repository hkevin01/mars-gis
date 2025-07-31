// Cypress E2E Support Configuration
// Import commands and custom functions for E2E testing

import '@cypress/code-coverage/support'
import 'cypress-axe'
import './commands'

// Add custom commands type definitions
declare global {
  namespace Cypress {
    interface Chainable {
      /**
       * Login with test user credentials
       */
      login(email?: string, password?: string): Chainable

      /**
       * Logout current user
       */
      logout(): Chainable

      /**
       * Navigate to specific Mars coordinates
       */
      navigateToMarsCoordinates(lat: number, lng: number): Chainable

      /**
       * Wait for Mars 3D globe to load
       */
      waitForMarsGlobe(): Chainable

      /**
       * Check accessibility violations
       */
      checkA11y(): Chainable

      /**
       * Seed test data
       */
      seedTestData(dataType: string): Chainable

      /**
       * Reset application state
       */
      resetAppState(): Chainable

      /**
       * Take performance snapshot
       */
      takePerformanceSnapshot(label: string): Chainable

      /**
       * Mock API response
       */
      mockApiResponse(endpoint: string, response: any): Chainable
    }
  }
}

// Global configuration
