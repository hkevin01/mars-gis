// Mock Service Worker server for API mocking in tests
import { rest } from 'msw';
import { setupServer } from 'msw/node';

// Mock API handlers
export const handlers = [
  // Dashboard metrics endpoint
  rest.get('/api/v1/dashboard/metrics', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        activeMissions: 3,
        dataProcessingProgress: 75,
        systemHealth: 'optimal',
        analysisComplete: 12,
        totalDatasets: 156,
        processingQueue: 5
      })
    );
  }),

  // System alerts endpoint
  rest.get('/api/v1/system/alerts', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json([
        {
          id: '1',
          level: 'info',
          message: 'System backup completed successfully',
          timestamp: new Date().toISOString()
        },
        {
          id: '2',
          level: 'warning',
          message: 'High CPU usage detected on processing node 2',
          timestamp: new Date().toISOString()
        }
      ])
    );
  }),

  // Models list endpoint
  rest.get('/api/v1/models', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json([
        { name: 'foundation', description: 'Earth-Mars Transfer Learning Model', status: 'active' },
        { name: 'multimodal', description: 'Multi-Modal Mars Processor', status: 'active' },
        { name: 'comparative', description: 'Comparative Planetary Analyzer', status: 'active' },
        { name: 'optimization', description: 'Landing Site Optimizer', status: 'active' },
        { name: 'self_supervised', description: 'Self-Supervised Learning Model', status: 'active' },
        { name: 'planetary_scale', description: 'Planetary-Scale Embeddings', status: 'active' }
      ])
    );
  }),

  // Model inference endpoint
  rest.post('/api/v1/models/infer', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        predictions: [0.8, 0.2, 0.6, 0.9],
        confidence: 0.85,
        processing_time: 1.2
      })
    );
  }),

  // Terrain analysis endpoint
  rest.post('/api/v1/terrain/analyze', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        analysis_id: 'analysis_123',
        status: 'processing'
      })
    );
  }),

  // Mission planning endpoint
  rest.post('/api/v1/missions/plan', (req, res, ctx) => {
    return res(
      ctx.status(201),
      ctx.json({
        mission_id: 'mission_456',
        optimal_sites: [
          { lat: -14.5684, lon: 175.4726, score: 0.95 },
          { lat: -14.6000, lon: 175.5000, score: 0.87 }
        ],
        risk_assessment: 'low'
      })
    );
  }),

  // Data upload endpoint
  rest.post('/api/v1/data/upload', (req, res, ctx) => {
    return res(
      ctx.status(201),
      ctx.json({
        file_id: 'file_789',
        upload_status: 'success'
      })
    );
  }),

  // Authentication endpoints
  rest.post('/api/v1/auth/login', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        access_token: 'mock_access_token',
        token_type: 'bearer',
        user: {
          id: 1,
          username: 'test_user',
          email: 'test@mars-gis.com',
          role: 'scientist'
        }
      })
    );
  }),

  // Health check endpoint
  rest.get('/health', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      })
    );
  }),

  // Error scenarios for testing
  rest.get('/api/v1/error-test', (req, res, ctx) => {
    return res(
      ctx.status(500),
      ctx.json({
        error: 'Internal server error',
        message: 'Test error for error handling'
      })
    );
  }),

  // Rate limiting test
  rest.get('/api/v1/rate-limit-test', (req, res, ctx) => {
    return res(
      ctx.status(429),
      ctx.json({
        error: 'Too Many Requests',
        message: 'Rate limit exceeded'
      })
    );
  }),

  // Unauthorized access test
  rest.get('/api/v1/unauthorized-test', (req, res, ctx) => {
    return res(
      ctx.status(401),
      ctx.json({
        error: 'Unauthorized',
        message: 'Authentication required'
      })
    );
  })
];

// Create server instance
export const server = setupServer(...handlers);
