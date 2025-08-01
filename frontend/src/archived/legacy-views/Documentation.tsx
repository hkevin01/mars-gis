import {
    Api as ApiIcon,
    Article as ArticleIcon,
    BugReport as BugReportIcon,
    Code as CodeIcon,
    Download as DownloadIcon,
    Feedback as FeedbackIcon,
    MenuBook as GuideIcon,
    Help as HelpIcon,
    Home as HomeIcon,
    NavigateNext as NavigateNextIcon,
    Print as PrintIcon,
    Search as SearchIcon,
    Share as ShareIcon,
    ThumbUp as ThumbUpIcon,
    Quiz as TutorialIcon
} from '@mui/icons-material';
import {
    Box,
    Breadcrumbs,
    Button,
    Card,
    CardContent,
    CardHeader,
    Chip,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Divider,
    IconButton,
    InputAdornment,
    Link,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Rating,
    TextField,
    Typography
} from '@mui/material';
import React, { useState } from 'react';

interface DocSection {
  id: string;
  title: string;
  category: 'guide' | 'api' | 'tutorial' | 'faq' | 'troubleshooting';
  icon: React.ReactNode;
  content: string;
  tags: string[];
  lastUpdated: string;
  rating: number;
  helpful: number;
}

interface DocCategory {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  sections: DocSection[];
}

const documentation: DocCategory[] = [
  {
    id: 'getting-started',
    name: 'Getting Started',
    icon: <HomeIcon />,
    description: 'Learn the basics of using MARS-GIS platform',
    sections: [
      {
        id: 'overview',
        title: 'Platform Overview',
        category: 'guide',
        icon: <GuideIcon />,
        content: `# MARS-GIS Platform Overview

The MARS-GIS platform is a comprehensive geospatial analysis and mission planning system designed specifically for Mars exploration. It combines advanced AI/ML capabilities with intuitive visualization tools to support scientific research and mission operations.

## Key Features

### 🗺️ Interactive Mapping
- High-resolution Mars surface imagery
- Multiple data layers (geological, atmospheric, terrain)
- Real-time mission tracking and visualization

### 🤖 AI-Powered Analysis
- Automated terrain classification
- Hazard detection and assessment
- Mineral composition analysis
- Weather pattern prediction

### 🚀 Mission Planning
- Route optimization algorithms
- Resource requirement calculations
- Risk assessment and mitigation
- Timeline management and scheduling

### 📊 Data Analysis
- Comprehensive data visualization
- Statistical analysis tools
- Export capabilities for research
- Integration with scientific workflows

## System Architecture

The platform consists of several integrated components:

1. **Frontend Interface**: React-based web application with Material-UI
2. **Backend Services**: Python FastAPI with PostgreSQL database
3. **AI/ML Engine**: PyTorch-based machine learning models
4. **Geospatial Engine**: Advanced GIS processing capabilities
5. **Visualization System**: 3D rendering with Three.js and Cesium`,
        tags: ['overview', 'features', 'architecture'],
        lastUpdated: '2024-01-20',
        rating: 4.8,
        helpful: 87,
      },
      {
        id: 'quick-start',
        title: 'Quick Start Guide',
        category: 'tutorial',
        icon: <TutorialIcon />,
        content: `# Quick Start Guide

Get up and running with MARS-GIS in just a few minutes.

## Step 1: Navigation

Use the left sidebar to navigate between different modules:
- **Dashboard**: Overview of missions and system status
- **3D Viewer**: Interactive Mars globe with terrain data
- **Interactive Map**: 2D mapping interface with layers
- **Mission Planner**: Create and manage exploration missions
- **Analysis Tools**: Data analysis and visualization

## Step 2: Your First Mission

1. Click on "Mission Planner" in the sidebar
2. Click the "+" button to create a new mission
3. Fill in mission details:
   - Name: "My First Mars Mission"
   - Description: "Exploration of interesting terrain"
   - Assign to available rover
4. Add tasks using the task creation wizard
5. Review and launch your mission

## Step 3: Monitoring Progress

1. Return to the Dashboard to see mission status
2. Use the 3D Viewer to visualize rover location
3. Check the Interactive Map for detailed positioning
4. View analysis results in Data Analysis section

## Step 4: Data Analysis

1. Navigate to "Data Analysis"
2. Select time range and filters
3. Explore different data types:
   - Atmospheric conditions
   - Geological samples
   - Terrain classifications
4. Export results for further analysis`,
        tags: ['tutorial', 'beginner', 'guide'],
        lastUpdated: '2024-01-19',
        rating: 4.9,
        helpful: 124,
      },
    ],
  },
  {
    id: 'user-guides',
    name: 'User Guides',
    icon: <GuideIcon />,
    description: 'Detailed guides for each platform feature',
    sections: [
      {
        id: 'dashboard-guide',
        title: 'Mission Dashboard',
        category: 'guide',
        icon: <ArticleIcon />,
        content: `# Mission Dashboard Guide

The Mission Dashboard provides a comprehensive overview of all active operations, system health, and key metrics.

## Dashboard Components

### Mission Control Panel
- **Active Missions**: Current rover and orbiter operations
- **System Status**: Real-time health monitoring
- **Environmental Conditions**: Current Mars weather
- **Resource Levels**: Battery, communication, storage

### Key Metrics Cards
- **Distance Traveled**: Total rover movement
- **Samples Collected**: Scientific specimen count
- **System Health**: Overall operational status
- **Mission Efficiency**: Completion rate metrics

### Tabbed Data Views

#### Weather Tab
- Temperature trends over time
- Atmospheric pressure variations
- Wind speed and direction
- Dust storm alerts and forecasts

#### Mission Progress Tab
- Daily distance traveled
- Sample collection progress
- Battery consumption patterns
- Task completion status

#### Terrain Analysis Tab
- Surface composition analysis
- Recent geological discoveries
- Hazard identification results
- Mineral distribution maps

#### Active Assets Tab
- Real-time asset locations
- Communication status
- Battery levels and charging
- Last contact timestamps

## Customization Options

### Time Range Selection
- Last 24 hours
- Last 7 days
- Last 30 days
- Custom date ranges

### Alert Configuration
- Critical system alerts
- Weather warnings
- Mission milestone notifications
- Equipment maintenance reminders`,
        tags: ['dashboard', 'overview', 'monitoring'],
        lastUpdated: '2024-01-18',
        rating: 4.6,
        helpful: 76,
      },
      {
        id: 'mission-planning',
        title: 'Mission Planning Guide',
        category: 'guide',
        icon: <ArticleIcon />,
        content: `# Mission Planning Guide

Create, manage, and execute Mars exploration missions with precision and efficiency.

## Mission Creation Workflow

### 1. Mission Definition
- **Name**: Descriptive mission identifier
- **Objective**: Primary scientific or operational goals
- **Duration**: Expected timeline for completion
- **Asset Assignment**: Rover, lander, or orbiter selection

### 2. Task Planning
- **Navigation Tasks**: Route planning and waypoint definition
- **Sampling Tasks**: Specimen collection locations and methods
- **Analysis Tasks**: On-site scientific measurements
- **Communication Tasks**: Data transmission windows
- **Maintenance Tasks**: Equipment checks and repairs

### 3. Resource Planning
- **Power Requirements**: Battery consumption estimates
- **Communication Windows**: Earth-Mars communication times
- **Environmental Constraints**: Temperature and weather limits
- **Equipment Needs**: Specialized instruments and tools

### 4. Risk Assessment
- **Terrain Hazards**: Steep slopes, rocks, soft soil
- **Environmental Risks**: Dust storms, extreme temperatures
- **Equipment Failures**: Backup plans and contingencies
- **Communication Blackouts**: Independent operation protocols

## Task Types and Configuration

### Navigation Tasks
- Start and end coordinates
- Intermediate waypoints
- Maximum speed settings
- Obstacle avoidance parameters

### Sampling Tasks
- Target coordinates
- Sample types (soil, rock, atmospheric)
- Collection methods
- Storage requirements

### Analysis Tasks
- Instrument selection
- Measurement parameters
- Data collection duration
- Quality control checks

## Mission Execution

### Pre-Launch Checklist
- [ ] All tasks defined and validated
- [ ] Resource requirements met
- [ ] Risk mitigation plans in place
- [ ] Communication schedule established
- [ ] Emergency procedures reviewed

### Launch and Monitoring
- Real-time progress tracking
- Task completion verification
- Resource consumption monitoring
- Environmental condition updates

### Mission Adjustments
- Dynamic replanning capabilities
- Emergency stop procedures
- Task priority modifications
- Route optimization updates`,
        tags: ['mission', 'planning', 'workflow'],
        lastUpdated: '2024-01-17',
        rating: 4.7,
        helpful: 92,
      },
    ],
  },
  {
    id: 'api-reference',
    name: 'API Reference',
    icon: <ApiIcon />,
    description: 'Technical documentation for developers',
    sections: [
      {
        id: 'rest-api',
        title: 'REST API Documentation',
        category: 'api',
        icon: <CodeIcon />,
        content: `# REST API Documentation

## Base URL
\`\`\`
https://api.mars-gis.com/v1
\`\`\`

## Authentication
All API requests require authentication using Bearer tokens:

\`\`\`http
Authorization: Bearer <your_api_token>
Content-Type: application/json
\`\`\`

## Missions Endpoint

### GET /missions
List all missions with optional filtering.

#### Parameters
- \`status\` (string): Filter by mission status
- \`asset\` (string): Filter by assigned asset
- \`limit\` (integer): Maximum results (default: 50)
- \`offset\` (integer): Pagination offset

#### Response
\`\`\`json
{
  "missions": [
    {
      "id": "mission-123",
      "name": "Olympia Survey",
      "status": "active",
      "asset": "rover-alpha",
      "created_at": "2024-01-15T10:00:00Z",
      "progress": 65
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
\`\`\`

### POST /missions
Create a new mission.

#### Request Body
\`\`\`json
{
  "name": "New Mission",
  "description": "Mission description",
  "asset": "rover-alpha",
  "start_date": "2024-01-20T08:00:00Z",
  "tasks": [
    {
      "type": "navigation",
      "name": "Move to target",
      "coordinates": [-14.5684, 175.4729]
    }
  ]
}
\`\`\`

### GET /missions/{id}
Get detailed mission information.

### PUT /missions/{id}
Update mission details.

### DELETE /missions/{id}
Cancel and delete a mission.

## Analysis Endpoint

### GET /analysis/results
List analysis results.

### POST /analysis/terrain
Start terrain analysis.

#### Request Body
\`\`\`json
{
  "region": "Olympia Undae",
  "analysis_type": "slope_stability",
  "coordinates": {
    "lat": -14.5684,
    "lon": 175.4729,
    "radius": 1000
  }
}
\`\`\`

## Data Endpoint

### GET /data/atmospheric
Get atmospheric data.

### GET /data/geological
Get geological sample data.

### GET /data/terrain
Get terrain classification data.`,
        tags: ['api', 'rest', 'endpoints'],
        lastUpdated: '2024-01-16',
        rating: 4.5,
        helpful: 45,
      },
    ],
  },
  {
    id: 'troubleshooting',
    name: 'Troubleshooting',
    icon: <BugReportIcon />,
    description: 'Common issues and solutions',
    sections: [
      {
        id: 'common-issues',
        title: 'Common Issues',
        category: 'troubleshooting',
        icon: <BugReportIcon />,
        content: `# Common Issues and Solutions

## Connection Problems

### Cannot Connect to Server
**Problem**: Error message "Unable to connect to MARS-GIS server"

**Solutions**:
1. Check your internet connection
2. Verify the server URL in settings
3. Check if firewall is blocking the connection
4. Contact system administrator if on corporate network

### Slow Loading Times
**Problem**: Application loads slowly or times out

**Solutions**:
1. Clear browser cache and cookies
2. Check internet connection speed
3. Try different browser or incognito mode
4. Reduce map quality in settings
5. Close other browser tabs to free memory

## Data Issues

### Missing Map Data
**Problem**: Blank areas or missing tiles in maps

**Solutions**:
1. Check internet connection
2. Refresh the page (Ctrl+F5)
3. Clear map cache in settings
4. Try different map layer
5. Report persistent issues to support

### Analysis Not Starting
**Problem**: Analysis jobs remain in "queued" status

**Solutions**:
1. Check if maximum concurrent analyses limit reached
2. Verify sufficient system resources
3. Wait for other analyses to complete
4. Cancel and restart the analysis
5. Check analysis parameters for validity

## Performance Issues

### High Memory Usage
**Problem**: Browser becomes slow or unresponsive

**Solutions**:
1. Close unused browser tabs
2. Reduce map cache size in settings
3. Lower render quality settings
4. Restart browser
5. Check system memory availability

### 3D Viewer Performance
**Problem**: Slow or jerky 3D visualization

**Solutions**:
1. Reduce render quality to "Low"
2. Disable unnecessary layers
3. Update graphics drivers
4. Check WebGL support in browser
5. Try different browser

## Mission Planning Issues

### Cannot Create Mission
**Problem**: Mission creation fails with validation errors

**Solutions**:
1. Check all required fields are filled
2. Verify coordinates are within valid range
3. Ensure asset is available and online
4. Check mission dates are in future
5. Validate task dependencies

### Mission Not Starting
**Problem**: Mission remains in "planned" status

**Solutions**:
1. Verify asset is available and charged
2. Check environmental conditions
3. Ensure all pre-launch checks completed
4. Review mission parameters
5. Contact mission control if persistent

## Account and Access Issues

### Login Problems
**Problem**: Cannot access account or frequent logouts

**Solutions**:
1. Verify username and password
2. Check if account is active
3. Clear browser cookies
4. Try password reset
5. Contact administrator

### Permission Errors
**Problem**: "Access denied" or "Insufficient permissions"

**Solutions**:
1. Check user role and permissions
2. Request access from administrator
3. Verify account is active
4. Try logging out and back in
5. Clear browser cache

## Getting Help

If these solutions don't resolve your issue:

1. **Check System Status**: Visit status.mars-gis.com
2. **Search Documentation**: Use search in help system
3. **Contact Support**: support@mars-gis.com
4. **Report Bugs**: Use feedback form in application
5. **Community Forum**: forum.mars-gis.com`,
        tags: ['troubleshooting', 'problems', 'solutions'],
        lastUpdated: '2024-01-15',
        rating: 4.4,
        helpful: 203,
      },
    ],
  },
];

export const Documentation: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>('getting-started');
  const [selectedSection, setSelectedSection] = useState<string>('overview');
  const [searchQuery, setSearchQuery] = useState('');
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackRating, setFeedbackRating] = useState<number>(0);

  const currentCategory = documentation.find(cat => cat.id === selectedCategory);
  const currentSection = currentCategory?.sections.find(sec => sec.id === selectedSection);

  const filteredSections = currentCategory?.sections.filter(section =>
    section.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    section.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
    section.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
  ) || [];

  const handleSectionSelect = (categoryId: string, sectionId: string) => {
    setSelectedCategory(categoryId);
    setSelectedSection(sectionId);
  };

  return (
    <Box sx={{ p: 3, height: '100%', display: 'flex', gap: 2 }}>
      {/* Sidebar Navigation */}
      <Box sx={{ width: 300, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {/* Search */}
        <Card>
          <CardContent sx={{ p: 2 }}>
            <TextField
              fullWidth
              placeholder="Search documentation..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
            />
          </CardContent>
        </Card>

        {/* Category Navigation */}
        <Card sx={{ flex: 1 }}>
          <CardHeader title="Documentation" />
          <CardContent sx={{ p: 0, height: 'calc(100% - 64px)', overflow: 'auto' }}>
            <List>
              {documentation.map((category) => (
                <React.Fragment key={category.id}>
                  <ListItem>
                    <ListItemButton
                      selected={selectedCategory === category.id}
                      onClick={() => {
                        setSelectedCategory(category.id);
                        setSelectedSection(category.sections[0]?.id || '');
                      }}
                    >
                      <ListItemIcon>{category.icon}</ListItemIcon>
                      <ListItemText
                        primary={category.name}
                        secondary={category.description}
                      />
                    </ListItemButton>
                  </ListItem>
                  
                  {selectedCategory === category.id && (
                    <Box sx={{ pl: 4 }}>
                      {filteredSections.map((section) => (
                        <ListItem key={section.id} dense>
                          <ListItemButton
                            selected={selectedSection === section.id}
                            onClick={() => setSelectedSection(section.id)}
                          >
                            <ListItemIcon sx={{ minWidth: 32 }}>
                              {section.icon}
                            </ListItemIcon>
                            <ListItemText
                              primary={section.title}
                              primaryTypographyProps={{ variant: 'body2' }}
                            />
                            <Chip
                              label={section.category}
                              size="small"
                              variant="outlined"
                              sx={{ ml: 1 }}
                            />
                          </ListItemButton>
                        </ListItem>
                      ))}
                    </Box>
                  )}
                  
                  <Divider />
                </React.Fragment>
              ))}
            </List>
          </CardContent>
        </Card>
      </Box>

      {/* Main Content */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {/* Header */}
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Breadcrumbs separator={<NavigateNextIcon fontSize="small" />}>
                <Link
                  component="button"
                  variant="body2"
                  onClick={() => console.log('Home')}
                  sx={{ textDecoration: 'none' }}
                >
                  Documentation
                </Link>
                <Link
                  component="button"
                  variant="body2"
                  onClick={() => console.log(currentCategory?.name)}
                  sx={{ textDecoration: 'none' }}
                >
                  {currentCategory?.name}
                </Link>
                <Typography variant="body2" color="text.primary">
                  {currentSection?.title}
                </Typography>
              </Breadcrumbs>
              
              <Box sx={{ display: 'flex', gap: 1 }}>
                <IconButton size="small">
                  <PrintIcon />
                </IconButton>
                <IconButton size="small">
                  <ShareIcon />
                </IconButton>
                <IconButton size="small">
                  <DownloadIcon />
                </IconButton>
              </Box>
            </Box>

            {currentSection && (
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                <Chip
                  label={currentSection.category.toUpperCase()}
                  color="primary"
                  variant="outlined"
                />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Rating value={currentSection.rating} readOnly size="small" />
                  <Typography variant="caption" color="text.secondary">
                    ({currentSection.rating})
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  Updated: {currentSection.lastUpdated}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {currentSection.helpful} found this helpful
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>

        {/* Content */}
        <Card sx={{ flex: 1 }}>
          <CardContent sx={{ height: '100%', overflow: 'auto' }}>
            {currentSection ? (
              <Box>
                <Typography
                  variant="body1"
                  component="div"
                  sx={{
                    whiteSpace: 'pre-wrap',
                    '& h1': { fontSize: '2rem', fontWeight: 700, mb: 2, mt: 3 },
                    '& h2': { fontSize: '1.5rem', fontWeight: 600, mb: 2, mt: 2 },
                    '& h3': { fontSize: '1.25rem', fontWeight: 600, mb: 1, mt: 2 },
                    '& p': { mb: 2 },
                    '& ul': { mb: 2, pl: 3 },
                    '& ol': { mb: 2, pl: 3 },
                    '& code': {
                      backgroundColor: 'rgba(0, 0, 0, 0.1)',
                      padding: '2px 4px',
                      borderRadius: 1,
                      fontFamily: 'monospace',
                    },
                    '& pre': {
                      backgroundColor: 'rgba(0, 0, 0, 0.1)',
                      padding: 2,
                      borderRadius: 1,
                      overflow: 'auto',
                      fontFamily: 'monospace',
                      mb: 2,
                    },
                  }}
                >
                  {currentSection.content}
                </Typography>

                <Divider sx={{ my: 3 }} />

                {/* Feedback Section */}
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Was this helpful?
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <Button
                      variant="outlined"
                      startIcon={<ThumbUpIcon />}
                      onClick={() => setFeedbackOpen(true)}
                    >
                      Yes
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<FeedbackIcon />}
                      onClick={() => setFeedbackOpen(true)}
                    >
                      Provide Feedback
                    </Button>
                  </Box>
                </Box>
              </Box>
            ) : (
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  textAlign: 'center',
                }}
              >
                <HelpIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  Welcome to MARS-GIS Documentation
                </Typography>
                <Typography variant="body1" color="text.disabled">
                  Select a topic from the sidebar to get started, or use the search to find specific information.
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>
      </Box>

      {/* Feedback Dialog */}
      <Dialog open={feedbackOpen} onClose={() => setFeedbackOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Provide Feedback</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <Typography variant="body2">
              How would you rate this documentation?
            </Typography>
            <Rating
              value={feedbackRating}
              onChange={(_, value) => setFeedbackRating(value || 0)}
              size="large"
            />
            <TextField
              multiline
              rows={4}
              placeholder="Tell us how we can improve this documentation..."
              fullWidth
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFeedbackOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={() => setFeedbackOpen(false)}>
            Submit Feedback
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};
