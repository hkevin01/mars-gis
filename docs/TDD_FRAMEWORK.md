# Mars-GIS Test-Driven Development (TDD) Framework

## 🎯 Overview

The Mars-GIS platform implements a comprehensive Test-Driven Development (TDD) framework following the **Red-Green-Refactor** cycle. This framework ensures high code quality, maintainability, and reliability through systematic testing practices.

## 🔄 TDD Workflow

### Red Phase 🔴
1. **Write failing tests first**
   - Unit tests that define expected behavior
   - Integration tests for component interactions
   - E2E tests for user workflows
   - Tests should fail initially (proving they work)

### Green Phase 🟢
2. **Write minimal code to make tests pass**
   - Implement just enough functionality
   - Focus on making tests pass, not perfect code
   - Avoid over-engineering at this stage

### Refactor Phase 🔵
3. **Improve code while keeping tests green**
   - Optimize performance
   - Improve readability and maintainability
   - Ensure all tests still pass
   - Apply design patterns where appropriate

## 🛠️ Testing Infrastructure

### Frontend Testing Stack
- **Jest**: Unit testing framework
- **React Testing Library**: Component testing utilities
- **Cypress**: End-to-end testing
- **MSW (Mock Service Worker)**: API mocking
- **@testing-library/jest-dom**: DOM testing utilities
- **Playwright**: Cross-browser testing

### Backend Testing Stack
- **pytest**: Python testing framework
- **testcontainers**: Database testing with containers
- **factory_boy**: Test data generation
- **responses**: HTTP request mocking
- **coverage.py**: Test coverage measurement

### Testing Environment
- **Docker Compose**: Isolated testing environments
- **GitHub Actions**: Continuous Integration
- **Codecov**: Coverage reporting
- **ESLint/Prettier**: Code quality and formatting

## 📁 Project Structure

```
mars-gis/
├── frontend/
│   ├── src/
│   │   ├── __tests__/           # Unit tests
│   │   │   ├── components.test.tsx
│   │   │   ├── hooks.test.ts
│   │   │   └── utils.test.ts
│   │   ├── __mocks__/           # Mock implementations
│   │   │   └── server.ts        # MSW server setup
│   │   └── components/          # Source code with co-located tests
│   ├── cypress/
│   │   ├── e2e/                 # E2E test specifications
│   │   ├── support/             # Cypress commands and configuration
│   │   └── fixtures/            # Test data
│   └── coverage/                # Coverage reports
├── backend/
│   ├── tests/                   # Python tests
│   │   ├── unit/
│   │   ├── integration/
│   │   └── e2e/
│   └── coverage/                # Backend coverage reports
├── scripts/
│   ├── dev.sh                   # Development utilities
│   └── tdd.sh                   # TDD workflow runner
└── docker-compose.test.yml      # Testing environment
```

## 🚀 Quick Start

### 1. Setup TDD Environment
```bash
./scripts/tdd.sh setup
```

### 2. Start Development Environment
```bash
./scripts/tdd.sh dev
```

### 3. Run Complete TDD Cycle
```bash
./scripts/tdd.sh cycle
```

### 4. Run Individual Phases
```bash
# Red phase - write failing tests
./scripts/tdd.sh red

# Green phase - make tests pass
./scripts/tdd.sh green

# Refactor phase - improve code
./scripts/tdd.sh refactor
```

## 🧪 Testing Commands

### Run All Tests
```bash
./scripts/tdd.sh test:all
```

### Specific Test Types
```bash
# Unit tests only
./scripts/tdd.sh test:unit

# Integration tests
./scripts/tdd.sh test:integration

# End-to-end tests
./scripts/tdd.sh test:e2e

# Coverage reporting
./scripts/tdd.sh test:coverage

# Performance tests
./scripts/tdd.sh test:performance

# Accessibility tests
./scripts/tdd.sh test:a11y
```

### Development Testing
```bash
# Watch mode for continuous testing
./scripts/tdd.sh watch

# Validate TDD compliance
./scripts/tdd.sh validate

# Show testing metrics
./scripts/tdd.sh metrics
```

## 📊 Coverage Requirements

- **Minimum Coverage**: 80%
- **Critical Components**: 95%
- **Edge Cases**: All must be tested
- **Integration Points**: 100% coverage required

### Coverage Thresholds
```json
{
  "branches": 80,
  "functions": 80,
  "lines": 80,
  "statements": 80
}
```

## 🔍 Test Categories

### 1. Unit Tests
**Purpose**: Test individual functions and components in isolation
**Location**: `frontend/src/__tests__/`, `backend/tests/unit/`
**Examples**:
- Component rendering
- Function behavior
- State management
- Error handling

### 2. Integration Tests
**Purpose**: Test component interactions and API integrations
**Location**: `frontend/src/__tests__/integration/`, `backend/tests/integration/`
**Examples**:
- API endpoint interactions
- Database operations
- Service layer integration
- Component communication

### 3. End-to-End Tests
**Purpose**: Test complete user workflows
**Location**: `cypress/e2e/`
**Examples**:
- User authentication flow
- Mars data visualization workflow
- Mission planning process
- Data analysis pipeline

### 4. Performance Tests
**Purpose**: Ensure application meets performance requirements
**Examples**:
- Component render times
- API response times
- Memory usage
- Bundle size optimization

### 5. Accessibility Tests
**Purpose**: Ensure WCAG compliance
**Examples**:
- Keyboard navigation
- Screen reader compatibility
- Color contrast
- Focus management

## 🐳 Docker Testing Environment

### Development Environment
```bash
docker-compose --profile development up -d
```

### Testing Environment
```bash
docker-compose -f docker-compose.test.yml up -d
```

### Services
- **frontend-dev**: Development server with hot reload
- **backend-dev**: API server with debug mode
- **frontend-test**: Testing environment for unit tests
- **e2e-tests**: Cypress testing environment
- **test-database**: Isolated PostgreSQL for testing
- **test-redis**: Redis instance for test caching

## ⚡ Continuous Integration

### GitHub Actions Workflow
The CI/CD pipeline runs comprehensive tests on every commit:

1. **Frontend TDD Tests**
   - Unit tests with coverage
   - Component tests
   - Integration tests
   - Accessibility tests
   - Performance tests

2. **Backend TDD Tests**
   - Unit tests with coverage
   - API tests
   - Database integration tests
   - ML model tests
   - Geospatial data tests

3. **End-to-End Tests**
   - Complete user workflows
   - Cross-browser testing
   - Performance validation

4. **Security and Quality**
   - Vulnerability scanning
   - Code quality checks
   - Type checking
   - Linting

## 📈 TDD Best Practices

### 1. Write Tests First
- Always start with a failing test
- Define expected behavior before implementation
- Use descriptive test names

### 2. Keep Tests Simple
- One assertion per test when possible
- Clear arrange-act-assert structure
- Avoid complex test setup

### 3. Test Behavior, Not Implementation
- Focus on what the code should do
- Avoid testing internal implementation details
- Mock external dependencies

### 4. Maintain Test Quality
- Refactor tests along with code
- Remove duplicate test code
- Keep tests readable and maintainable

### 5. Use Appropriate Test Types
- Unit tests for business logic
- Integration tests for interactions
- E2E tests for critical user paths

## 🔧 Configuration Files

### Jest Configuration (`frontend/jest.config.js`)
```javascript
module.exports = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
};
```

### Cypress Configuration (`cypress.config.ts`)
```typescript
export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',
    viewportWidth: 1280,
    viewportHeight: 720,
    video: true,
    screenshotOnRunFailure: true,
  },
});
```

## 🚨 Troubleshooting

### Common Issues

1. **Tests failing in CI but passing locally**
   - Check environment variables
   - Verify Docker configuration
   - Ensure proper test isolation

2. **Slow test execution**
   - Use test parallelization
   - Optimize database operations
   - Mock heavy computations

3. **Flaky E2E tests**
   - Add proper wait conditions
   - Use deterministic test data
   - Implement retry logic

### Debug Commands
```bash
# Check environment health
./scripts/tdd.sh health

# View detailed logs
docker-compose logs frontend-test

# Reset environment
./scripts/tdd.sh reset
```

## 📚 Resources

- [Jest Documentation](https://jestjs.io/)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
- [Cypress Documentation](https://docs.cypress.io/)
- [TDD Best Practices](https://martinfowler.com/bliki/TestDrivenDevelopment.html)
- [Testing Trophy](https://kentcdodds.com/blog/the-testing-trophy-and-testing-classifications)

## 🎯 TDD Success Criteria

✅ **All tests pass in CI/CD pipeline**
✅ **80%+ code coverage maintained**
✅ **No critical accessibility violations**
✅ **Performance budgets met**
✅ **Security vulnerabilities resolved**
✅ **TDD workflow followed consistently**

## 🔄 Continuous Improvement

1. **Regular Retrospectives**: Review testing effectiveness
2. **Metric Tracking**: Monitor coverage and test execution times
3. **Tool Updates**: Keep testing dependencies current
4. **Team Training**: Ensure all developers follow TDD practices

---

**Remember**: The goal of TDD is not just to have tests, but to drive better design and ensure reliable, maintainable code. Follow the Red-Green-Refactor cycle consistently for the best results! 🚀
