// Cypress E2E Testing Configuration
// Comprehensive test configuration for Mars-GIS platform

import { defineConfig } from 'cypress'

export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',
    supportFile: 'cypress/support/e2e.ts',
    specPattern: 'cypress/e2e/**/*.cy.{js,jsx,ts,tsx}',
    viewportWidth: 1280,
    viewportHeight: 720,
    video: true,
    videosFolder: 'cypress/videos',
    screenshotsFolder: 'cypress/screenshots',
    videoCompression: 32,
    screenshotOnRunFailure: true,

    // Test configuration
    defaultCommandTimeout: 10000,
    pageLoadTimeout: 30000,
    requestTimeout: 10000,
    responseTimeout: 30000,

    // Retry configuration
    retries: {
      runMode: 2,
      openMode: 0,
    },

    // Environment variables
    env: {
      apiUrl: 'http://localhost:8000/api/v1',
      wsUrl: 'ws://localhost:8000',
      coverage: true,
      codeCoverage: {
        url: 'http://localhost:3000/__coverage__',
      },
    },

    // Test isolation
    testIsolation: true,

    setupNodeEvents(on, config) {
      // Code coverage plugin
      require('@cypress/code-coverage/task')(on, config)

      // Accessibility testing plugin
      on('task', {
        log(message) {
          console.log(message)
          return null
        },
        table(message) {
          console.table(message)
          return null
        },
      })

      // Custom tasks for TDD workflow
      on('task', {
        // Reset test database
        resetTestDb() {
          // Implementation for resetting test database
          return null
        },

        // Seed test data
        seedTestData(data) {
          // Implementation for seeding test data
          return null
        },

        // Performance monitoring
        recordPerformance(metrics) {
          // Implementation for recording performance metrics
          return null
        },
      })

      return config
    },
  },

  component: {
    devServer: {
      framework: 'create-react-app',
      bundler: 'webpack',
    },
    supportFile: 'cypress/support/component.ts',
    specPattern: 'src/**/*.cy.{js,jsx,ts,tsx}',
    indexHtmlFile: 'cypress/support/component-index.html',
  },

  // Global configuration
  reporter: 'cypress-multi-reporters',
  reporterOptions: {
    configFile: 'cypress/reporter-config.json',
  },

  // Browser configuration
  chromeWebSecurity: false,
  experimentalStudio: true,
  experimentalWebKitSupport: true,

  // File watching
  watchForFileChanges: true,

  // Timeouts
  execTimeout: 60000,
  taskTimeout: 60000,
}
