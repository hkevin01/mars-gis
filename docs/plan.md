Based on your resume and the job requirements, I'll create a comprehensive MARS-GIS project that showcases your expertise in geospatial data processing, AI/ML, and mission-critical systems. This project will demonstrate skills directly relevant to the Expert GIS AI Software Engineer position.

Let me create a complete project structure with all the requested components:

## Project Analysis & Concept

The MARS-GIS project will be a comprehensive geospatial analysis and visualization platform for Mars exploration data, integrating NASA's Mars datasets, USGS planetary data, and advanced AI/ML techniques for terrain analysis, landing site selection, and mission planning support.

## Complete Project Structure

```
MARS-GIS/
├── .copilot/
├── .github/
├── .vscode/
├── assets/
├── data/
├── docs/
├── scripts/
├── src/
├── tests/
├── venv/
├── .editorconfig
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

Let me create all the necessary files:

## .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
.venv/
.env/

# IDEs
.vscode/settings_local.json
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.h5
*.hdf5
*.nc
*.tif
*.tiff
data/raw/
data/processed/temp/
*.pickle
*.pkl

# Logs
logs/
*.log

# Cache
.cache/
.pytest_cache/

# Coverage
htmlcov/
.coverage
.coverage.*
coverage.xml

# MyPy
.mypy_cache/
.dmypy.json
dmypy.json

# AI/ML
models/checkpoints/
*.pth
*.onnx
wandb/

# GIS specific
*.shp.xml
*.qpj
*.cpg
```

## .vscode/settings.json

```json
{
  "chat.tools.autoApprove": true,
  "chat.agent.maxRequests": 100,
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=88"],
  "python.sortImports.args": ["--profile", "black"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.autoImportCompletions": true,
  "files.associations": {
    "*.py": "python"
  },
  "python.naming.functionNameConvention": "snake_case",
  "python.naming.variableNameConvention": "snake_case",
  "python.naming.constantNameConvention": "UPPER_CASE",
  "python.naming.classNameConvention": "PascalCase",
  "python.naming.moduleNameConvention": "snake_case",
  "java.compile.nullAnalysis.mode": "automatic",
  "java.format.settings.url": "https://raw.githubusercontent.com/google/styleguide/gh-pages/eclipse-java-google-style.xml",
  "java.codeGeneration.hashCodeEquals.useJava7Objects": true,
  "java.codeGeneration.useBlocks": true,
  "C_Cpp.clang_format_style": "Google",
  "C_Cpp.formatting": "clangFormat",
  "files.exclude": {
    "**/.git": true,
    "**/.svn": true,
    "**/.hg": true,
    "**/CVS": true,
    "**/.DS_Store": true,
    "**/Thumbs.db": true,
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/venv": true,
    "**/env": true
  },
  "search.exclude": {
    "**/node_modules": true,
    "**/bower_components": true,
    "**/*.code-search": true,
    "**/venv": true,
    "**/env": true,
    "**/__pycache__": true
  }
}
```

## .vscode/launch.json

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true,
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    },
    {
      "name": "Python: Mars GIS Main",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/src/mars_gis/main.py",
      "console": "integratedTerminal",
      "justMyCode": true,
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    }
  ]
}
```

## .vscode/tasks.json

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "setup-env",
      "type": "shell",
      "command": "python",
      "args": ["-m", "venv", "venv"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "install-deps",
      "type": "shell",
      "command": "${workspaceFolder}/venv/bin/pip",
      "args": ["install", "-r", "requirements.txt"],
      "group": "build",
      "dependsOn": "setup-env"
    },
    {
      "label": "run-tests",
      "type": "shell",
      "command": "${workspaceFolder}/venv/bin/python",
      "args": ["-m", "pytest", "tests/"],
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "format-code",
      "type": "shell",
      "command": "${workspaceFolder}/venv/bin/black",
      "args": ["src/", "tests/"],
      "group": "build"
    }
  ]
}
```

## .github/workflows/ci.yml

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install GDAL dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y gdal-bin libgdal-dev
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Test with pytest
      run: |
        pip install pytest pytest-cov
        pytest tests/ --cov=src/ --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

## .github/ISSUE_TEMPLATE/bug_report.md

```markdown
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment (please complete the following information):**
 - OS: [e.g. iOS]
 - Python Version: [e.g. 3.9]
 - MARS-GIS Version: [e.g. 1.0.0]

**Additional context**
Add any other context about the problem here.
```

## .github/PULL_REQUEST_TEMPLATE.md

```markdown
## Description
Brief description of the changes in this PR.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally with my changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
```

## .copilot/settings.yml

```yaml
# GitHub Copilot configuration for MARS-GIS project
project_context:
  description: "Mars GIS analysis and visualization platform"
  primary_languages: 
    - python
    - javascript
    - sql
  frameworks:
    - geopandas
    - pytorch
    - fastapi
    - react
  
suggestions:
  enable_geospatial_completions: true
  enable_ml_completions: true
  enable_nasa_api_completions: true
  
custom_prompts:
  - name: "mars_data_processing"
    description: "Generate code for processing Mars geological data"
    template: "Process Mars {data_type} data using GeoPandas and create visualization"
  
  - name: "ai_terrain_analysis"
    description: "Generate ML models for terrain classification"
    template: "Create PyTorch model for Mars terrain classification with {input_features}"

ignore_patterns:
  - "*.log"
  - "data/raw/*"
  - "venv/*"
```

## docs/project_plan.md

```markdown
# MARS-GIS Project Plan

## Project Overview
MARS-GIS is a comprehensive geospatial analysis and visualization platform designed for Mars exploration data. The project integrates NASA's Mars datasets, USGS planetary data, and advanced AI/ML techniques for terrain analysis, landing site selection, and mission planning support.

## Project Goals
- Create a unified platform for Mars geospatial data analysis
- Implement AI-driven terrain classification and hazard detection
- Support mission planning with landing site optimization
- Provide real-time data visualization and interactive mapping
- Enable collaborative research through data sharing capabilities

---

## Phase 1: Foundation & Data Infrastructure
**Timeline: Weeks 1-4**

### Data Architecture & Management
- [ ] **NASA Mars Data Integration**
  - Set up automated data pipelines for Mars Reconnaissance Orbiter (MRO) data
  - Integrate Mars Global Surveyor elevation models (MOLA)
  - Connect to NASA Planetary Data System (PDS) APIs
  - Solutions: Use NASA's REST APIs, implement caching with Redis, create ETL pipelines with Apache Airflow

- [ ] **USGS Planetary Data Integration**
  - Connect to USGS Astrogeology Science Center databases
  - Implement Mars geological mapping data access
  - Set up mineral composition datasets integration
  - Solutions: Use USGS web services, implement OGC WMS/WFS protocols, create data validation schemas

- [ ] **Geospatial Database Setup**
  - Design PostGIS database schema for Mars coordinate systems
  - Implement spatial indexing for large raster datasets
  - Create data versioning and lineage tracking
  - Solutions: PostgreSQL with PostGIS extension, implement R-tree spatial indexing, use Apache Kafka for change streams

- [ ] **Real-time Data Streaming**
  - Set up data ingestion pipelines for live satellite feeds
  - Implement change detection algorithms for surface monitoring
  - Create data quality assessment frameworks
  - Solutions: Apache Kafka + Kafka Connect, implement computer vision change detection, use Great Expectations for data validation

- [ ] **Cloud Storage Architecture**
  - Design scalable storage for multi-terabyte Mars datasets
  - Implement data compression and archival strategies
  - Set up disaster recovery and backup systems
  - Solutions: AWS S3 with intelligent tiering, implement HDF5/NetCDF compression, use AWS Glacier for archival

---

## Phase 2: AI/ML Core Development
**Timeline: Weeks 5-8**

### Machine Learning Infrastructure
- [ ] **Terrain Classification Models**
  - Develop CNN models for Mars surface feature identification
  - Implement transfer learning from Earth geological data
  - Create ensemble models for improved accuracy
  - Solutions: PyTorch with torchvision, use ResNet/EfficientNet architectures, implement weighted ensemble voting

- [ ] **Landing Site Safety Assessment**
  - Build ML models for hazard detection (rocks, slopes, dust storms)
  - Implement multi-criteria decision analysis algorithms
  - Create uncertainty quantification for safety predictions
  - Solutions: Use computer vision object detection (YOLO/R-CNN), implement fuzzy logic systems, use Bayesian neural networks

- [ ] **Atmospheric Analysis Models**
  - Develop time-series models for weather prediction
  - Implement dust storm tracking and prediction
  - Create atmospheric composition analysis tools
  - Solutions: Use LSTM/Transformer models, implement particle tracking algorithms, use spectroscopic analysis libraries

- [ ] **MLOps Pipeline Implementation**
  - Set up model versioning and experiment tracking
  - Implement automated model training and validation
  - Create model deployment and monitoring systems
  - Solutions: Use MLflow for experiment tracking, implement Kubeflow pipelines, use Prometheus for model monitoring

- [ ] **GPU Computing Optimization**
  - Optimize CUDA kernels for large-scale data processing
  - Implement distributed training across multiple GPUs
  - Create memory-efficient algorithms for limited resources
  - Solutions: Use CuPy for GPU acceleration, implement data parallel training with PyTorch DDP, use gradient checkpointing

---

## Phase 3: Geospatial Analysis Engine
**Timeline: Weeks 9-12**

### Advanced Spatial Analytics
- [ ] **3D Terrain Reconstruction**
  - Implement stereo photogrammetry algorithms
  - Create digital elevation model (DEM) generation
  - Develop mesh simplification for real-time rendering
  - Solutions: Use OpenCV stereo algorithms, implement Delaunay triangulation, use level-of-detail (LOD) techniques

- [ ] **Geological Feature Extraction**
  - Develop algorithms for crater detection and analysis
  - Implement mineral mapping from spectroscopic data
  - Create geological unit boundary delineation
  - Solutions: Use Hough transform for crater detection, implement spectral unmixing algorithms, use watershed segmentation

- [ ] **Mission Path Planning**
  - Implement optimal route planning for rovers
  - Create obstacle avoidance algorithms
  - Develop energy-efficient path optimization
  - Solutions: Use A* and Dijkstra's algorithms, implement RRT (Rapidly-exploring Random Trees), use genetic algorithms for optimization

- [ ] **Spatial Statistics & Modeling**
  - Implement geostatistical analysis tools
  - Create spatial autocorrelation analysis
  - Develop predictive spatial models
  - Solutions: Use PyKrige for kriging interpolation, implement Moran's I statistics, use spatial regression models

- [ ] **Multi-scale Analysis Framework**
  - Develop pyramid data structures for multi-resolution analysis
  - Implement scale-invariant feature detection
  - Create automated scale selection algorithms
  - Solutions: Use image pyramids, implement SIFT/SURF feature detection, use wavelet transforms

---

## Phase 4: Visualization & User Interface
**Timeline: Weeks 13-16**

### Interactive Mapping Platform
- [ ] **3D Globe Visualization**
  - Implement WebGL-based Mars globe rendering
  - Create real-time layer switching and transparency controls
  - Develop smooth zoom and pan interactions
  - Solutions: Use Three.js or Cesium.js, implement tile-based rendering, use WebGL shaders for performance

- [ ] **Data Layer Management**
  - Create dynamic layer loading and caching system
  - Implement temporal data visualization (time sliders)
  - Develop custom symbology and styling tools
  - Solutions: Use OpenLayers or Leaflet, implement tile caching with Redis, create custom WebGL renderers

- [ ] **Real-time Dashboard**
  - Build mission monitoring dashboard with live metrics
  - Implement alert systems for critical events
  - Create customizable widget layouts
  - Solutions: Use React with D3.js, implement WebSocket connections, use React Grid Layout for dashboards

- [ ] **Collaborative Features**
  - Implement multi-user annotation and markup tools
  - Create shared workspace functionality
  - Develop version control for collaborative analysis
  - Solutions: Use Socket.io for real-time collaboration, implement operational transforms, use Git-like versioning

- [ ] **Mobile-Responsive Interface**
  - Create touch-optimized controls for tablets
  - Implement offline capabilities for field use
  - Develop simplified mobile workflows
  - Solutions: Use Progressive Web App (PWA) architecture, implement service workers for offline caching, use responsive CSS frameworks

---

## Phase 5: Integration & Deployment
**Timeline: Weeks 17-20**

### Production Systems
- [ ] **API Development**
  - Create RESTful APIs for data access and analysis
  - Implement GraphQL for flexible data querying
  - Develop authentication and authorization systems
  - Solutions: Use FastAPI with Pydantic validation, implement OAuth2 with JWT tokens, use rate limiting with Redis

- [ ] **Performance Optimization**
  - Implement caching strategies for frequently accessed data
  - Optimize database queries and spatial indexes
  - Create load balancing for high-traffic scenarios
  - Solutions: Use Redis for application caching, implement connection pooling, use NGINX for load balancing

- [ ] **Security Implementation**
  - Implement secure data transmission (HTTPS/TLS)
  - Create audit logging for all system activities
  - Develop intrusion detection and prevention
  - Solutions: Use Let's Encrypt SSL certificates, implement structured logging with ELK stack, use fail2ban for security

- [ ] **Monitoring & Observability**
  - Set up application performance monitoring
  - Implement distributed tracing for microservices
  - Create automated alerting for system issues
  - Solutions: Use Prometheus and Grafana for metrics, implement Jaeger for tracing, use PagerDuty for alerting

- [ ] **Deployment Automation**
  - Create containerized deployment with Docker
  - Implement CI/CD pipelines with automated testing
  - Set up blue-green deployment strategies
  - Solutions: Use Docker and Kubernetes, implement GitHub Actions workflows, use ArgoCD for GitOps deployment

---

## Success Metrics
- **Performance**: Sub-second response times for spatial queries
- **Accuracy**: >95% accuracy in terrain classification models
- **Scalability**: Support for 100+ concurrent users
- **Data Volume**: Handle 10TB+ of Mars geological data
- **Availability**: 99.9% uptime for critical mission support

## Risk Mitigation
- **Data Quality**: Implement comprehensive validation and error handling
- **Performance**: Use caching and optimization strategies throughout
- **Security**: Follow OWASP guidelines and conduct regular security audits
- **Scalability**: Design with microservices architecture for horizontal scaling
```

## README.md

```markdown
# MARS-GIS: Mars Geospatial Intelligence System

[![CI/CD](https://github.com/yourusername/mars-gis/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/yourusername/mars-gis/actions)
[![codecov](https://codecov.io/gh/yourusername/mars-gis/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/mars-gis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive geospatial analysis and visualization platform for Mars exploration data, integrating NASA's Mars datasets, USGS planetary data, and advanced AI/ML techniques for terrain analysis, landing site selection, and mission planning support.

## 🚀 Features

- **Multi-source Data Integration**: NASA MRO, MGS, USGS Astrogeology data
- **AI-Powered Analysis**: Deep learning models for terrain classification and hazard detection
- **3D Visualization**: Interactive Mars globe with real-time data layers
- **Mission Planning Tools**: Landing site optimization and path planning algorithms
- **Real-time Monitoring**: Live satellite feed integration with change detection
- **Collaborative Platform**: Multi-user annotation and shared workspace capabilities

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, FastAPI, PostgreSQL/PostGIS
- **AI/ML**: PyTorch, scikit-learn, CUDA acceleration
- **Geospatial**: GeoPandas, GDAL, PostGIS, OGC standards
- **Frontend**: React, Three.js/Cesium.js, D3.js
- **Infrastructure**: Docker, Kubernetes, Redis, Apache Kafka
- **Cloud**: AWS S3, EC2, RDS

## 📋 Prerequisites

- Python 3.8 or higher
- GDAL development libraries
- PostgreSQL with PostGIS extension
- CUDA-capable GPU (recommended for ML workflows)
- 16GB+ RAM (for large dataset processing)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mars-gis.git
cd mars-gis
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Initialize Database
```bash
python scripts/setup_database.py
```

### 6. Download Sample Data
```bash
python scripts/download_sample_data.py
```

### 7. Run the Application
```bash
python src/mars_gis/main.py
```

Visit `http://localhost:8000` to access the web interface.

## 🏗️ Project Structure

```
MARS-GIS/
├── src/mars_gis/          # Main application code
│   ├── api/               # FastAPI endpoints
│   ├── core/              # Core business logic
│   ├── data/              # Data processing modules
│   ├── ml/                # Machine learning models
│   └── visualization/     # Visualization components
├── tests/                 # Test suite
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── data/                  # Data storage
└── assets/                # Static assets
```

## 📊 Key Capabilities

### AI-Powered Terrain Analysis
- **Surface Classification**: CNN models trained on Mars geological features
- **Hazard Detection**: Real-time identification of landing hazards
- **Temporal Analysis**: Change detection across multiple time periods

### Mission Planning Support
- **Landing Site Selection**: Multi-criteria optimization algorithms
- **Route Planning**: Energy-efficient path planning for rovers
- **Risk Assessment**: Probabilistic safety analysis for mission planning

### Data Integration
- **NASA APIs**: Automated ingestion from Planetary Data System
- **USGS Integration**: Geological mapping and mineral composition data
- **Real-time Feeds**: Live satellite data with change detection

## 🔧 Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
isort src/ tests/
```

### Type Checking
```bash
mypy src/
```

### Building Documentation
```bash
cd docs/
make html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](.github/CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- NASA Planetary Data System for Mars datasets
- USGS Astrogeology Science Center for geological data
- European Space Agency for additional Mars observations
- Open-source geospatial community for foundational tools

## 📞 Support

For questions and support:
- 📧 Email: kevin.hildebrand@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/mars-gis/issues)
- 📖 Documentation: [Full Documentation](https://mars-gis.readthedocs.io/)

---

**Built with ❤️ for Mars exploration and scientific discovery**
```

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools-scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "mars-gis"
dynamic = ["version"]
description = "Mars Geospatial Intelligence System"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Kevin Hildebrand", email = "kevin.hildebrand@gmail.com"}
]
keywords = ["mars", "gis", "geospatial", "nasa", "ai", "machine-learning"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: GIS",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.8"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.0.0",
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "geoalchemy2>=0.14.0",
    "geopandas>=0.14.0",
    "rasterio>=1.3.0",
    "gdal>=3.4.0",
    "pyproj>=3.4.0",
    "shapely>=2.0.0",
    "fiona>=1.8.0",
    "pandas>=2.0.0",
    "numpy>=1.21.0",
    "scipy>=1.9.0",
    "scikit-learn>=1.3.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "xarray>=2023.1.0",
    "netcdf4>=1.6.0",
    "h5py>=3.8.0",
    "redis>=4.5.0",
    "celery>=5.3.0",
    "requests>=2.28.0",
    "aiohttp>=3.8.0",
    "pillow>=10.0.0",
    "matplotlib>=3.6.0",
    "plotly>=5.17.0",
    "folium>=0.15.0",
    "streamlit>=1.28.0",
    "python-multipart>=0.0.6",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.9.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.6.0",
    "pre-commit>=3.4.0",
]
docs = [
    "sphinx>=7.1.0",
    "sphinx-rtd-theme>=1.3.0",
    "myst-parser>=2.0.0",
]
ml = [
    "wandb>=0.15.0",
    "mlflow>=2.7.0",
    "optuna>=3.4.0",
    "tensorboard>=2.14.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/mars-gis"
Documentation = "https://mars-gis.readthedocs.io/"
Repository = "https://github.com/yourusername/mars-gis.git"
"Bug Tracker" = "https://github.com/yourusername/mars-gis/issues"

[project.scripts]
mars-gis = "mars_gis.cli:main"

[tool.setuptools_scm]
write_to = "src/mars_gis/_version.py"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["mars_gis"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "geopandas.*",
    "rasterio.*",
    "gdal.*",
    "osr.*",
    "ogr.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src/mars_gis",
    "--cov-report=html",
    "--cov-report=term-missing",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

## requirements.txt

```txt
# Core Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
geoalchemy2>=0.14.0

# Geospatial
geopandas>=0.14.0
rasterio>=1.3.0
GDAL>=3.4.0
pyproj>=3.4.0
shapely>=2.0.0
fiona>=1.8.0

# Data Science
pandas>=2.0.0
numpy>=1.21.0
scipy>=1.9.0
scikit-learn>=1.3.0
xarray>=2023.1.0
netcdf4>=1.6.0
h5py>=3.8.0

# Machine Learning
torch>=2.0.0
torchvision>=0.15.0

# Caching & Queues
redis>=4.5.0
celery>=5.3.0

# HTTP & APIs
requests>=2.28.0
aiohttp>=3.8.0

# Visualization
pillow>=10.0.0
matplotlib>=3.6.0
plotly>=5.17.0
folium>=0.15.0

# Web Interface
streamlit>=1.28.0

# Security
python-multipart>=0.0.6
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4

# Development
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.0
black>=23.9.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.6.0
pre-commit>=3.4.0
```

## Source Code Structure

Let me create the basic source code structure:

## src/mars_gis/__init__.py

```python
"""MARS-GIS: Mars Geospatial Intelligence System."""

__version__ = "0.1.0"
__author__ = "Kevin Hildebrand"
__email__ = "kevin.hildebrand@gmail.com"

from mars_gis.core.config import settings

__all__ = ["settings"]
```

## src/mars_gis/main.py

```python
"""Main application entry point for MARS-GIS."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mars_gis.api.routes import api_router
from mars_gis.core.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MARS-GIS API",
        description="Mars Geospatial Intelligence System API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "mars_gis.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.ENVIRONMENT == "development" else False,
    )
```

## scripts/setup_environment.py

```python
#!/usr/bin/env python3
"""Setup script for MARS-GIS development environment."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, cwd: Path = None) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            command.split(),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}: {e.stderr}")
        return False


def main():
    """Set up the development environment."""
    project_root = Path(__file__).parent.parent
    
    print("Setting up MARS-GIS development environment...")
    
    # Create virtual environment if it doesn't exist
    venv_path = project_root / "venv"
    if not venv_path.exists():
        print("Creating virtual environment...")
        if not run_command(f"{sys.executable} -m venv venv", project_root):
            sys.exit(1)
    
    # Determine pip path
    if os.name == 'nt':  # Windows
        pip_path = venv_path / "Scripts" / "pip"
    else:  # Unix-like
        pip_path = venv_path / "bin" / "pip"
    
    # Upgrade pip
    if not run_command(f"{pip_path} install --upgrade pip", project_root):
        sys.exit(1)
    
    # Install dependencies
    if not run_command(f"{pip_path} install -r requirements.txt", project_root):
        sys.exit(1)
    
    # Install package in development mode
    if not run_command(f"{pip_path} install -e .", project_root):
        sys.exit(1)
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "logs",
        "assets/maps",
        "assets/images",
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("\n🚀 Environment setup complete!")
    print("To activate the virtual environment:")
    if os.name == 'nt':
        print("  venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")


if __name__ == "__main__":
    main()
```

This comprehensive MARS-GIS project structure demonstrates:

1. **Professional Development Standards**: Modern Python packaging, CI/CD, comprehensive testing
2. **Geospatial Expertise**: Integration with NASA/USGS data, advanced GIS capabilities
3. **AI/ML Integration**: PyTorch models, MLOps practices, GPU optimization
4. **Mission-Critical Focus**: Robust architecture, monitoring, security considerations
5. **Collaborative Features**: GitHub workflows, documentation, contribution guidelines

The project directly aligns with the job requirements by showcasing skills in:
- Advanced GIS software engineering
- AI/ML integration for geospatial analysis
- Large dataset processing and management
- Mission-critical system development
- DevOps and modern development practices

This structure provides a solid foundation for demonstrating your expertise while creating a genuinely valuable tool for Mars exploration research.