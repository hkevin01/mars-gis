# Mars GIS Platform Testing Guide

This document provides comprehensive information about the testing infrastructure and practices for the Mars GIS Platform.

## Test Structure

Our testing framework is organized into several categories:

### Test Categories

- **Unit Tests** (`@pytest.mark.unit`): Test individual functions and classes in isolation
- **Integration Tests** (`@pytest.mark.integration`): Test interactions between components
- **API Tests** (`@pytest.mark.api`): Test REST API endpoints and authentication
- **ML Tests** (`@pytest.mark.ml`): Test machine learning models and training pipelines
- **Geospatial Tests** (`@pytest.mark.geospatial`): Test geospatial analysis and coordinate systems
- **Data Processing Tests** (`@pytest.mark.data_processing`): Test data ingestion and transformation

### Test Files

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_api.py             # API endpoint tests
├── test_ml_models.py       # Machine learning model tests
├── test_geospatial.py      # Geospatial analysis tests
├── test_data_processing.py # Data processing pipeline tests
└── pytest.ini             # Pytest configuration
```

## Running Tests

### Quick Start

```bash
# Run all fast tests (recommended for development)
./run_tests.py --fast

# Run all tests with coverage
./run_tests.py --all --coverage

# Run specific test categories
./run_tests.py --unit          # Unit tests only
./run_tests.py --integration   # Integration tests only
./run_tests.py --api          # API tests only
./run_tests.py --ml           # ML tests only
./run_tests.py --geospatial   # Geospatial tests only
./run_tests.py --data         # Data processing tests only
```

### Advanced Testing Options

```bash
# Run tests in parallel (faster)
./run_tests.py --parallel --fast

# Verbose output with detailed logs
./run_tests.py --verbose --unit

# Stop on first failure
./run_tests.py --fail-fast --api

# Install dependencies and run tests
./run_tests.py --install-deps --all
```

### Direct Pytest Usage

```bash
# Run specific test file
pytest tests/test_api.py -v

# Run specific test class
pytest tests/test_ml_models.py::TestMarsTerrainCNN -v

# Run specific test method
pytest tests/test_api.py::TestMissionAPI::test_create_mission -v

# Run tests with markers
pytest -m "unit and not slow" -v

# Run tests with coverage
pytest --cov=mars_gis --cov-report=html tests/
```

## Test Environment Setup

### Prerequisites

1. **Python 3.8+** - Required for async features and type annotations
2. **PostgreSQL** - For database integration tests
3. **Redis** - For caching tests (optional)
4. **GDAL** - For geospatial functionality

### Environment Variables

Create a `.env` file with test configuration:

```bash
# Test Database
TEST_DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/test_mars_gis
TEST_REDIS_URL=redis://localhost:6379/1

# API Testing
SECRET_KEY=test-secret-key
DEBUG=true
TESTING=true

# ML Testing
TORCH_DEVICE=cpu
BATCH_SIZE=16
```

### Dependencies Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Install development dependencies only
pip install pytest pytest-asyncio pytest-mock pytest-cov factory-boy faker httpx
```

## Writing Tests

### Test Structure Guidelines

1. **Arrange**: Set up test data and mocks
2. **Act**: Execute the function/method being tested
3. **Assert**: Verify the expected outcomes

### Example Test Structure

```python
@pytest.mark.unit
class TestMarsCoordinateSystem:
    """Test cases for Mars coordinate system handling."""
    
    def test_coordinate_validation(self):
        """Test Mars coordinate validation."""
        # Arrange
        valid_coords = [-14.5684, 175.4729]
        invalid_coords = [95.0, 185.0]
        
        # Act
        valid_result = validate_mars_coordinates(valid_coords)
        invalid_result = validate_mars_coordinates(invalid_coords)
        
        # Assert
        assert valid_result is True
        assert invalid_result is False
```

### Using Fixtures

Fixtures are defined in `tests/conftest.py` and provide reusable test data:

```python
def test_mission_creation(mock_mission_data):
    """Test mission creation with mock data."""
    mission = create_mission(mock_mission_data)
    assert mission.name == mock_mission_data['name']
    assert mission.status == 'active'
```

### Mocking External Dependencies

Use pytest-mock for mocking external services:

```python
def test_api_call_with_mock(mocker):
    """Test API call with mocked external service."""
    # Mock external API
    mock_response = mocker.Mock()
    mock_response.json.return_value = {'status': 'success'}
    mocker.patch('requests.get', return_value=mock_response)
    
    # Test the function
    result = fetch_mars_data('some_endpoint')
    assert result['status'] == 'success'
```

### Async Testing

For testing async functions:

```python
@pytest.mark.asyncio
async def test_async_api_endpoint(async_client):
    """Test async API endpoint."""
    response = await async_client.get("/api/v1/missions")
    assert response.status_code == 200
```

## Test Coverage

### Coverage Goals

- **Overall Coverage**: > 80%
- **Critical Components**: > 90%
- **API Endpoints**: 100%
- **ML Models**: > 85%

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=mars_gis --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

### Coverage Configuration

Coverage settings are in `pytest.ini`:

```ini
[tool:pytest]
addopts = 
    --cov=mars_gis
    --cov-branch
    --cov-report=term-missing:skip-covered
    --cov-fail-under=80
```

## Continuous Integration

### GitHub Actions Workflow

Tests are automatically run on:
- Pull requests
- Pushes to main branch
- Scheduled runs (nightly)

### Test Matrix

Tests run across multiple configurations:
- Python versions: 3.8, 3.9, 3.10, 3.11
- Operating systems: Ubuntu, macOS, Windows
- Dependencies: Minimum and latest versions

## Performance Testing

### Load Testing

Use pytest-benchmark for performance tests:

```python
def test_coordinate_transformation_performance(benchmark):
    """Test coordinate transformation performance."""
    coords = generate_test_coordinates(1000)
    
    result = benchmark(transform_coordinates, coords)
    
    # Benchmark automatically measures execution time
    assert len(result) == 1000
```

### Memory Testing

Monitor memory usage during tests:

```python
import psutil

def test_large_dataset_processing():
    """Test processing large datasets without memory leaks."""
    initial_memory = psutil.Process().memory_info().rss
    
    # Process large dataset
    result = process_large_mars_dataset()
    
    final_memory = psutil.Process().memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
```

## Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with pdb on failures
pytest --pdb tests/test_api.py

# Run with detailed output
pytest -vvv --tb=long tests/test_ml_models.py

# Run with logging output
pytest --log-cli-level=DEBUG tests/
```

### Test Data Inspection

Use the `--capture=no` flag to see print statements:

```bash
pytest --capture=no tests/test_geospatial.py::test_coordinate_transformation
```

### IDE Integration

Most IDEs support pytest integration:

- **VS Code**: Python extension with pytest discovery
- **PyCharm**: Built-in pytest runner
- **Vim/Neovim**: vim-test plugin

## Best Practices

### Test Organization

1. **One test class per component** being tested
2. **Descriptive test names** that explain what is being tested
3. **Group related tests** using classes
4. **Use meaningful assertions** with custom error messages

### Mock Usage

1. **Mock external dependencies** (APIs, databases, file systems)
2. **Don't mock the code you're testing**
3. **Use fixtures for common mocks**
4. **Verify mock interactions when relevant**

### Test Data

1. **Use factories** for generating test data
2. **Keep test data minimal** but representative
3. **Use parametrized tests** for multiple scenarios
4. **Clean up test data** after tests

### Async Testing

1. **Mark async tests** with `@pytest.mark.asyncio`
2. **Use async fixtures** for async setup
3. **Test both success and failure cases**
4. **Handle timeouts appropriately**

## Troubleshooting

### Common Issues

1. **Import Errors**: Check PYTHONPATH and virtual environment
2. **Database Errors**: Ensure test database is created and accessible
3. **Permission Errors**: Check file permissions for test files
4. **Memory Errors**: Use smaller test datasets or increase available memory

### Test Environment Issues

```bash
# Reset test environment
rm -rf .pytest_cache htmlcov
python -m pytest --cache-clear

# Check test discovery
pytest --collect-only tests/

# Verify fixtures
pytest --fixtures tests/conftest.py
```

### Getting Help

1. Check the test output for detailed error messages
2. Use `--tb=long` for full tracebacks
3. Add print statements for debugging
4. Use pytest's `--pdb` flag to drop into debugger on failures

## Contributing

When adding new features:

1. **Write tests first** (TDD approach)
2. **Ensure good coverage** of new code
3. **Update test documentation** if needed
4. **Run all tests** before submitting PR

### Test Review Checklist

- [ ] Tests cover both positive and negative cases
- [ ] Edge cases are tested
- [ ] Mocks are used appropriately
- [ ] Test names are descriptive
- [ ] Tests are independent and can run in any order
- [ ] Performance impact is considered
- [ ] Documentation is updated if needed
