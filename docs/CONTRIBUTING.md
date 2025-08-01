# Contributing to MARS-GIS

**Welcome to the MARS-GIS project!** We're excited that you're interested in contributing to advancing Mars exploration through geospatial intelligence. This guide will help you get started with contributing to our mission.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Environment](#development-environment)
4. [Contributing Workflow](#contributing-workflow)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Issue Reporting](#issue-reporting)
9. [Feature Requests](#feature-requests)
10. [Review Process](#review-process)
11. [Release Process](#release-process)
12. [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to making participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Examples of behavior that contributes to creating a positive environment include:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Examples of unacceptable behavior include:**

- The use of sexualized language or imagery
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at conduct@mars-gis.org. All complaints will be reviewed and investigated promptly and fairly.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.9+** installed
- **Git** for version control
- **Docker** (optional, for containerized development)
- **PostgreSQL 13+** with PostGIS extension
- **Redis 6+** for caching
- **Node.js 16+** (for frontend development)

### Quick Setup

1. **Fork the repository** on GitHub
2. **Clone your fork locally:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/mars-gis.git
   cd mars-gis
   ```

3. **Set up the development environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

   # Set up pre-commit hooks
   pre-commit install
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

5. **Initialize the database:**
   ```bash
   # Start PostgreSQL and Redis
   docker-compose up -d postgres redis

   # Run migrations
   alembic upgrade head

   # Load sample data (optional)
   python -m mars_gis.cli load-sample-data
   ```

6. **Run the application:**
   ```bash
   # Backend
   python -m mars_gis.main

   # Frontend (in separate terminal)
   cd frontend
   npm install
   npm start
   ```

## Development Environment

### Recommended Tools

- **IDE**: PyCharm, VS Code, or vim with appropriate plugins
- **Git Client**: Command line or GitHub Desktop
- **Database Client**: pgAdmin, DBeaver, or psql
- **API Testing**: Postman, Insomnia, or curl
- **Browser**: Chrome/Firefox with developer tools

### Environment Configuration

**Development Settings (.env):**
```bash
# Application
MARS_GIS_ENV=development
MARS_GIS_DEBUG=true
MARS_GIS_SECRET_KEY=dev-secret-key-change-in-production

# Database
DATABASE_URL=postgresql://mars_gis_user:password@localhost:5432/mars_gis_dev

# Redis
REDIS_URL=redis://localhost:6379/0

# External APIs
NASA_API_KEY=your_nasa_api_key
ESA_API_KEY=your_esa_api_key

# ML Models
ML_MODELS_PATH=./models
ML_CACHE_SIZE=1000

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=detailed
```

### Docker Development

**For containerized development:**
```bash
# Start all services
docker-compose up -d

# Follow logs
docker-compose logs -f mars-gis

# Run commands in container
docker-compose exec mars-gis python -m pytest

# Stop services
docker-compose down
```

## Contributing Workflow

### 1. Choose an Issue

- Browse [open issues](https://github.com/your-org/mars-gis/issues)
- Look for issues labeled `good first issue` for beginners
- Comment on the issue to indicate you're working on it
- Ask questions if the requirements are unclear

### 2. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/add-mission-planner
# or
git checkout -b bugfix/fix-coordinate-validation
# or
git checkout -b docs/update-api-documentation
```

### 3. Make Changes

- **Write code** following our [coding standards](#coding-standards)
- **Add tests** for new functionality
- **Update documentation** if needed
- **Test your changes** locally

### 4. Commit Changes

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: add mission path optimization algorithm

- Implement A* pathfinding for rover missions
- Add terrain cost calculation
- Include obstacle avoidance
- Add comprehensive tests

Closes #123"
```

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/add-mission-planner

# Create pull request on GitHub
# - Provide clear title and description
# - Link to related issues
# - Add screenshots for UI changes
# - Request review from maintainers
```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(api): add mission risk assessment endpoint
fix(ui): resolve coordinate input validation
docs(readme): update installation instructions
test(models): add unit tests for terrain classifier
```

## Coding Standards

### Python Code Style

We follow **PEP 8** with some modifications:

```python
# Line length: 88 characters (Black formatter)
# Use Black for formatting
# Use isort for import sorting
# Use flake8 for linting
# Use mypy for type checking
```

**Example Python Code:**
```python
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from pydantic import BaseModel, Field
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class MissionRequest(BaseModel):
    """Request model for creating a new mission."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    mission_type: str = Field(..., regex="^(rover|orbital|surface)$")
    target_coordinates: List[float] = Field(..., min_items=2, max_items=2)
    constraints: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "name": "Olympus Mons Survey",
                "description": "Geological survey of Olympus Mons region",
                "mission_type": "rover",
                "target_coordinates": [-14.5684, 175.4729],
                "constraints": {"max_slope": 15.0}
            }
        }


class MissionService:
    """Service for managing Mars missions."""

    def __init__(self, db_session, logger=None):
        self.db_session = db_session
        self.logger = logger or logging.getLogger(__name__)

    async def create_mission(self, request: MissionRequest) -> Dict[str, Any]:
        """Create a new mission with validation and risk assessment.

        Args:
            request: Mission creation request

        Returns:
            Created mission data with ID and risk assessment

        Raises:
            HTTPException: If validation fails or coordinates are invalid
        """
        try:
            # Validate coordinates
            if not self._validate_mars_coordinates(request.target_coordinates):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid Mars coordinates"
                )

            # Create mission
            mission = await self._create_mission_record(request)

            # Calculate risk assessment
            risk_assessment = await self._calculate_risk_assessment(mission)

            self.logger.info(f"Created mission {mission.id}: {mission.name}")

            return {
                "id": mission.id,
                "name": mission.name,
                "status": mission.status,
                "risk_assessment": risk_assessment,
                "created_at": mission.created_at.isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to create mission: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )

    def _validate_mars_coordinates(self, coordinates: List[float]) -> bool:
        """Validate Mars coordinates are within valid ranges."""
        lat, lon = coordinates
        return -90 <= lat <= 90 and -180 <= lon <= 180
```

### TypeScript/JavaScript Code Style

We use **ESLint** and **Prettier** for JavaScript/TypeScript:

```typescript
// Use TypeScript for type safety
// Use ESLint with Airbnb config
// Use Prettier for formatting
// Use Husky for pre-commit hooks
```

**Example TypeScript Code:**
```typescript
import React, { useState, useEffect, useCallback } from 'react';
import { Mission, MissionStatus, Coordinates } from '../types/Mission';
import { apiClient } from '../services/apiClient';
import { useNotification } from '../hooks/useNotification';

interface MissionListProps {
  filters?: {
    status?: MissionStatus;
    type?: string;
  };
  onMissionSelect?: (mission: Mission) => void;
}

const MissionList: React.FC<MissionListProps> = ({
  filters,
  onMissionSelect
}) => {
  const [missions, setMissions] = useState<Mission[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { showNotification } = useNotification();

  const fetchMissions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await apiClient.getMissions(filters);

      if (response.success) {
        setMissions(response.data.missions);
      } else {
        throw new Error(response.error || 'Failed to fetch missions');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      showNotification('Error fetching missions', 'error');
    } finally {
      setLoading(false);
    }
  }, [filters, showNotification]);

  useEffect(() => {
    fetchMissions();
  }, [fetchMissions]);

  const handleMissionClick = useCallback((mission: Mission) => {
    onMissionSelect?.(mission);
  }, [onMissionSelect]);

  if (loading) {
    return <div className="loading-spinner">Loading missions...</div>;
  }

  if (error) {
    return (
      <div className="error-message">
        <p>Error: {error}</p>
        <button onClick={fetchMissions} type="button">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="mission-list">
      {missions.length === 0 ? (
        <p className="no-missions">No missions found</p>
      ) : (
        <ul>
          {missions.map((mission) => (
            <li key={mission.id} className="mission-item">
              <button
                onClick={() => handleMissionClick(mission)}
                type="button"
                className="mission-button"
              >
                <h3>{mission.name}</h3>
                <p>Status: {mission.status}</p>
                <p>Type: {mission.type}</p>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default MissionList;
```

### Code Quality Tools

**Pre-commit Configuration (.pre-commit-config.yaml):**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings, flake8-import-order]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-redis]
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── performance/   # Performance tests
├── security/      # Security tests
└── fixtures/      # Test data and fixtures
```

### Testing Requirements

1. **Unit Tests**: All new code must have unit tests with 95%+ coverage
2. **Integration Tests**: API endpoints and service integrations
3. **E2E Tests**: Critical user workflows
4. **Performance Tests**: For ML models and API endpoints
5. **Security Tests**: Authentication, authorization, input validation

### Test Example

```python
import pytest
from httpx import AsyncClient
from unittest.mock import Mock, patch

from mars_gis.main import app
from mars_gis.models.mission import Mission, MissionStatus


class TestMissionAPI:
    """Test cases for mission management API."""

    @pytest.fixture
    async def client(self):
        """Create test client."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.fixture
    def sample_mission_data(self):
        """Sample mission data for testing."""
        return {
            "name": "Test Mission",
            "description": "A test mission",
            "mission_type": "rover",
            "target_coordinates": [-14.5684, 175.4729]
        }

    async def test_create_mission_success(self, client, sample_mission_data):
        """Test successful mission creation."""
        response = await client.post("/api/v1/missions", json=sample_mission_data)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == sample_mission_data["name"]
        assert "id" in data["data"]

    async def test_create_mission_invalid_coordinates(self, client, sample_mission_data):
        """Test mission creation with invalid coordinates."""
        sample_mission_data["target_coordinates"] = [95.0, 185.0]  # Invalid

        response = await client.post("/api/v1/missions", json=sample_mission_data)

        assert response.status_code == 422
        assert "coordinate" in response.json()["detail"][0]["msg"].lower()

    @patch('mars_gis.services.mission_service.MissionService.calculate_risk')
    async def test_mission_risk_calculation(self, mock_risk_calc, client, sample_mission_data):
        """Test mission risk assessment calculation."""
        mock_risk_calc.return_value = {"overall_risk": "low", "factors": {}}

        response = await client.post("/api/v1/missions", json=sample_mission_data)

        assert response.status_code == 201
        data = response.json()
        assert "risk_assessment" in data["data"]
        mock_risk_calc.assert_called_once()
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m e2e

# Run with coverage
pytest --cov=mars_gis --cov-report=html

# Run specific test file
pytest tests/test_mission_api.py

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

## Documentation

### Documentation Standards

1. **Code Documentation**: All public APIs must have docstrings
2. **API Documentation**: OpenAPI/Swagger specs for all endpoints
3. **User Documentation**: Clear guides and examples
4. **Developer Documentation**: Architecture and setup guides

### Docstring Format

We use **Google-style docstrings**:

```python
def calculate_trajectory(
    start_coords: Coordinates,
    target_coords: Coordinates,
    constraints: Dict[str, Any]
) -> TrajectoryPlan:
    """Calculate optimal trajectory between two points on Mars.

    This function computes the most efficient path for a rover to travel
    between two coordinates while respecting terrain constraints and
    mission parameters.

    Args:
        start_coords: Starting coordinates (lat, lon) in decimal degrees
        target_coords: Target coordinates (lat, lon) in decimal degrees
        constraints: Mission constraints including:
            - max_slope: Maximum allowable slope in degrees
            - avoid_zones: List of coordinates to avoid
            - preferred_terrain: Preferred terrain types

    Returns:
        TrajectoryPlan containing:
            - waypoints: List of intermediate coordinates
            - distance: Total distance in meters
            - estimated_time: Estimated travel time in seconds
            - risk_assessment: Overall risk score (0-1)

    Raises:
        ValueError: If coordinates are invalid or outside Mars bounds
        CalculationError: If no valid path can be found

    Example:
        >>> start = Coordinates(lat=-14.5, lon=175.4)
        >>> target = Coordinates(lat=-14.6, lon=175.5)
        >>> constraints = {"max_slope": 15.0}
        >>> plan = calculate_trajectory(start, target, constraints)
        >>> print(f"Distance: {plan.distance}m")
    """
```

### API Documentation

All API endpoints are documented using **OpenAPI 3.0**:

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="MARS-GIS API",
    description="Mars Geospatial Intelligence System REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.post(
    "/api/v1/missions",
    response_model=MissionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new mission",
    description="Create a new Mars exploration mission with risk assessment",
    responses={
        201: {"description": "Mission created successfully"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    },
    tags=["missions"]
)
async def create_mission(
    mission: MissionRequest = Body(
        ...,
        example={
            "name": "Olympus Mons Survey",
            "description": "Geological survey mission",
            "mission_type": "rover",
            "target_coordinates": [-14.5684, 175.4729]
        }
    )
) -> MissionResponse:
    """Create a new mission with automatic risk assessment."""
    # Implementation here
```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Clear title** describing the issue
2. **Steps to reproduce** the bug
3. **Expected behavior** vs actual behavior
4. **Environment information**:
   - MARS-GIS version
   - Python version
   - Operating system
   - Browser (for frontend issues)
5. **Error messages** and stack traces
6. **Screenshots** if applicable

**Bug Report Template:**
```markdown
## Bug Description
Brief description of the bug.

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- MARS-GIS version: 1.0.0
- Python version: 3.9.7
- OS: Ubuntu 20.04
- Browser: Chrome 96.0

## Error Messages
```
Paste error messages here
```

## Additional Context
Any additional information that might be helpful.
```

### Security Issues

**Do not** report security vulnerabilities in public issues. Instead:

1. Email security@mars-gis.org
2. Include detailed description
3. Provide steps to reproduce
4. Allow time for investigation before disclosure

## Feature Requests

### Feature Request Process

1. **Search existing issues** to avoid duplicates
2. **Create detailed proposal** with use cases
3. **Discuss with community** in issue comments
4. **Wait for maintainer approval** before implementing

**Feature Request Template:**
```markdown
## Feature Summary
Brief description of the proposed feature.

## Problem Statement
What problem does this feature solve?

## Proposed Solution
Detailed description of how the feature should work.

## Use Cases
- Use case 1: Description
- Use case 2: Description

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Implementation Notes
Technical notes or considerations.

## Alternatives Considered
Other solutions that were considered.
```

## Review Process

### Pull Request Review

1. **Automated Checks**: All PRs must pass:
   - Unit tests
   - Integration tests
   - Code quality checks
   - Security scans

2. **Manual Review**: Maintainers will review:
   - Code quality and style
   - Test coverage
   - Documentation
   - Performance implications
   - Security considerations

3. **Approval Process**:
   - At least one maintainer approval required
   - All review comments must be addressed
   - All checks must pass

### Review Checklist

**For Reviewers:**
- [ ] Code follows project standards
- [ ] Tests are comprehensive and pass
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance is acceptable
- [ ] API changes are backward compatible

**For Contributors:**
- [ ] Self-review completed
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] PR description is clear and complete

## Release Process

### Version Numbering

We follow **Semantic Versioning (SemVer)**:

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Schedule

- **Major releases**: Every 6 months
- **Minor releases**: Monthly
- **Patch releases**: As needed for critical bugs

### Release Steps

1. **Feature freeze** 1 week before release
2. **Release candidate** testing
3. **Final testing** and bug fixes
4. **Release notes** preparation
5. **Version tagging** and deployment

## Community

### Communication Channels

- **GitHub Discussions**: General questions and community discussions
- **Slack**: Real-time chat and coordination (#mars-gis channel)
- **Twitter**: @MarsGIS for announcements
- **Monthly Calls**: Community video calls (first Thursday of each month)

### Recognition

We recognize contributors through:

- **Contributors file** listing all contributors
- **Release notes** highlighting contributions
- **GitHub badges** for significant contributions
- **Swag program** for active contributors

### Mentorship

New contributors can:

- Join the **mentorship program**
- Pair with experienced contributors
- Attend **office hours** (Fridays 2-4 PM UTC)
- Ask questions in the **#newcomers** Slack channel

---

**Thank you for contributing to MARS-GIS!** Your work helps advance our understanding of Mars and contributes to future human exploration of the Red Planet. 🚀

For questions about contributing, contact us at:
- Email: contributors@mars-gis.org
- Slack: #contributors channel
- GitHub: @mars-gis/maintainers
