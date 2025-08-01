import {
    Assessment as AssessmentIcon,
    CameraAlt as CameraIcon,
    Download as DownloadIcon,
    Layers as LayersIcon,
    Refresh as RefreshIcon,
    Settings as SettingsIcon,
    Visibility as VisibilityIcon,
    ZoomIn as ZoomInIcon
} from '@mui/icons-material';
import {
    Avatar,
    Box,
    Button,
    Card,
    CardContent,
    CardHeader,
    Chip,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    FormControl,
    FormControlLabel,
    FormLabel,
    Grid,
    IconButton,
    LinearProgress,
    List,
    ListItem,
    Paper,
    Radio,
    RadioGroup,
    Slider,
    TextField,
    Tooltip,
    Typography
} from '@mui/material';
import React, { useState } from 'react';

interface TerrainFeature {
  id: string;
  name: string;
  type: 'crater' | 'ridge' | 'valley' | 'plain' | 'dune' | 'volcano';
  coordinates: [number, number];
  elevation: number;
  size: number; // in km²
  difficulty: 'low' | 'medium' | 'high' | 'extreme';
  hazards: string[];
  mineralContent: string[];
  description: string;
}

interface TerrainLayer {
  id: string;
  name: string;
  type: 'elevation' | 'slope' | 'roughness' | 'thermal' | 'composition';
  visible: boolean;
  opacity: number;
  colorScale: string;
}

interface TerrainAnalysisResult {
  id: string;
  region: string;
  analysisType: string;
  status: 'completed' | 'processing' | 'queued';
  progress: number;
  confidence: number;
  findings: string;
  recommendedActions: string[];
  startTime: Date;
  completionTime?: Date;
}

const terrainFeatures: TerrainFeature[] = [
  {
    id: 'valles-marineris',
    name: 'Valles Marineris',
    type: 'valley',
    coordinates: [-14.0, -60.0],
    elevation: -7000,
    size: 4000,
    difficulty: 'extreme',
    hazards: ['steep_slopes', 'rockfall', 'communication_shadow'],
    mineralContent: ['sulfates', 'phyllosilicates', 'hematite'],
    description: 'Massive canyon system, deepest point on Mars',
  },
  {
    id: 'olympus-mons',
    name: 'Olympus Mons',
    type: 'volcano',
    coordinates: [18.65, -133.8],
    elevation: 21900,
    size: 300000,
    difficulty: 'extreme',
    hazards: ['extreme_elevation', 'steep_cliffs', 'thermal_anomalies'],
    mineralContent: ['basalt', 'olivine', 'pyroxene'],
    description: 'Largest volcano in the solar system',
  },
  {
    id: 'gale-crater',
    name: 'Gale Crater',
    type: 'crater',
    coordinates: [-5.4, 137.8],
    elevation: -4500,
    size: 15400,
    difficulty: 'medium',
    hazards: ['dust_accumulation', 'wind_patterns'],
    mineralContent: ['clay_minerals', 'sulfates', 'magnetite'],
    description: 'Impact crater with evidence of ancient water activity',
  },
  {
    id: 'hellas-basin',
    name: 'Hellas Basin',
    type: 'crater',
    coordinates: [-42.7, 70.0],
    elevation: -8200,
    size: 2300000,
    difficulty: 'high',
    hazards: ['atmospheric_pressure_variations', 'dust_storms'],
    mineralContent: ['impact_glass', 'shocked_minerals', 'basalt'],
    description: 'Largest impact crater on Mars',
  },
];

const terrainLayers: TerrainLayer[] = [
  {
    id: 'elevation',
    name: 'Digital Elevation Model',
    type: 'elevation',
    visible: true,
    opacity: 0.8,
    colorScale: 'terrain',
  },
  {
    id: 'slope',
    name: 'Slope Analysis',
    type: 'slope',
    visible: false,
    opacity: 0.7,
    colorScale: 'viridis',
  },
  {
    id: 'roughness',
    name: 'Surface Roughness',
    type: 'roughness',
    visible: false,
    opacity: 0.6,
    colorScale: 'plasma',
  },
  {
    id: 'thermal',
    name: 'Thermal Inertia',
    type: 'thermal',
    visible: false,
    opacity: 0.5,
    colorScale: 'coolwarm',
  },
];

const analysisResults: TerrainAnalysisResult[] = [
  {
    id: 'analysis-1',
    region: 'Olympia Undae',
    analysisType: 'Slope Stability',
    status: 'completed',
    progress: 100,
    confidence: 94,
    findings: 'Stable terrain with slopes <15°. Suitable for rover operations.',
    recommendedActions: ['Proceed with planned mission', 'Monitor for seasonal changes'],
    startTime: new Date('2024-01-15T10:00:00'),
    completionTime: new Date('2024-01-15T14:30:00'),
  },
  {
    id: 'analysis-2',
    region: 'Acidalia Planitia',
    analysisType: 'Trafficability Assessment',
    status: 'processing',
    progress: 68,
    confidence: 87,
    findings: 'Preliminary results show good mobility conditions',
    recommendedActions: ['Continue analysis', 'Prepare contingency routes'],
    startTime: new Date('2024-01-16T09:00:00'),
  },
  {
    id: 'analysis-3',
    region: 'Chryse Planitia',
    analysisType: 'Landing Site Evaluation',
    status: 'queued',
    progress: 0,
    confidence: 0,
    findings: 'Analysis pending resource availability',
    recommendedActions: ['Wait for processing slot'],
    startTime: new Date('2024-01-17T08:00:00'),
  },
];

const getDifficultyColor = (difficulty: string) => {
  switch (difficulty) {
    case 'low': return 'success';
    case 'medium': return 'info';
    case 'high': return 'warning';
    case 'extreme': return 'error';
    default: return 'default';
  }
};

const getTerrainTypeIcon = (type: string) => {
  switch (type) {
    case 'crater': return '🌑';
    case 'ridge': return '⛰️';
    case 'valley': return '🏞️';
    case 'plain': return '🏔️';
    case 'dune': return '🏜️';
    case 'volcano': return '🌋';
    default: return '🗻';
  }
};

export const TerrainAnalysis: React.FC = () => {
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
  const [layers, setLayers] = useState<TerrainLayer[]>(terrainLayers);
  const [analysisDialogOpen, setAnalysisDialogOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [elevationRange, setElevationRange] = useState<number[]>([-8000, 22000]);
  const [slopeThreshold, setSlopeThreshold] = useState(30);
  const [selectedAnalysisType, setSelectedAnalysisType] = useState('slope');

  const selectedFeatureData = terrainFeatures.find(f => f.id === selectedFeature);

  const handleLayerToggle = (layerId: string) => {
    setLayers(prev => prev.map(layer =>
      layer.id === layerId
        ? { ...layer, visible: !layer.visible }
        : layer
    ));
  };

  const handleLayerOpacity = (layerId: string, opacity: number) => {
    setLayers(prev => prev.map(layer =>
      layer.id === layerId
        ? { ...layer, opacity: opacity / 100 }
        : layer
    ));
  };

  const startNewAnalysis = () => {
    console.log('Starting new terrain analysis:', selectedAnalysisType);
    setAnalysisDialogOpen(false);
  };

  return (
    <Box sx={{ p: 3, height: '100%', display: 'flex', gap: 2 }}>
      {/* Main Analysis View */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {/* Header */}
        <Card>
          <CardHeader
            title="Mars Terrain Analysis"
            subheader="Advanced geomorphological analysis and terrain classification"
            action={
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Tooltip title="New Analysis">
                  <Button
                    variant="contained"
                    startIcon={<AssessmentIcon />}
                    onClick={() => setAnalysisDialogOpen(true)}
                  >
                    Analyze
                  </Button>
                </Tooltip>
                <Tooltip title="Settings">
                  <IconButton onClick={() => setSettingsOpen(true)}>
                    <SettingsIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Refresh">
                  <IconButton>
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Export">
                  <IconButton>
                    <DownloadIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
        </Card>

        {/* Terrain Visualization */}
        <Card sx={{ flex: 1 }}>
          <CardHeader
            title="3D Terrain Model"
            action={
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Chip
                  icon={<LayersIcon />}
                  label={`${layers.filter(l => l.visible).length} layers active`}
                  variant="outlined"
                />
                <IconButton size="small">
                  <ZoomInIcon />
                </IconButton>
                <IconButton size="small">
                  <CameraIcon />
                </IconButton>
              </Box>
            }
          />
          <CardContent sx={{ height: 'calc(100% - 64px)', p: 0, position: 'relative' }}>
            {/* Mock 3D Terrain View */}
            <Box
              sx={{
                width: '100%',
                height: '100%',
                backgroundColor: '#2c1810',
                backgroundImage: `
                  radial-gradient(circle at 30% 20%, rgba(139, 69, 19, 0.4) 0%, transparent 50%),
                  radial-gradient(circle at 70% 60%, rgba(205, 97, 51, 0.3) 0%, transparent 50%),
                  radial-gradient(circle at 20% 80%, rgba(160, 82, 45, 0.2) 0%, transparent 50%)
                `,
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              {/* Terrain Features Overlay */}
              {terrainFeatures.map((feature, index) => (
                <Box
                  key={feature.id}
                  sx={{
                    position: 'absolute',
                    left: `${20 + index * 20}%`,
                    top: `${15 + index * 15}%`,
                    cursor: 'pointer',
                    transform: 'translate(-50%, -50%)',
                  }}
                  onClick={() => setSelectedFeature(feature.id)}
                >
                  <Paper
                    sx={{
                      p: 1,
                      backgroundColor: selectedFeature === feature.id ? 'primary.main' : 'rgba(255, 255, 255, 0.9)',
                      color: selectedFeature === feature.id ? 'white' : 'text.primary',
                      minWidth: 120,
                      textAlign: 'center',
                      border: '2px solid',
                      borderColor: selectedFeature === feature.id ? 'primary.main' : 'transparent',
                    }}
                  >
                    <Typography variant="caption" display="block">
                      {getTerrainTypeIcon(feature.type)} {feature.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {feature.elevation > 0 ? '+' : ''}{feature.elevation}m
                    </Typography>
                  </Paper>
                </Box>
              ))}

              {/* Layer Controls Overlay */}
              <Paper
                sx={{
                  position: 'absolute',
                  top: 16,
                  right: 16,
                  p: 2,
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  minWidth: 200,
                }}
              >
                <Typography variant="subtitle2" gutterBottom>
                  Active Layers
                </Typography>
                {layers.filter(l => l.visible).map((layer) => (
                  <Box key={layer.id} sx={{ mb: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="caption">{layer.name}</Typography>
                      <Typography variant="caption">{Math.round(layer.opacity * 100)}%</Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={layer.opacity * 100}
                      sx={{ height: 4, borderRadius: 2 }}
                    />
                  </Box>
                ))}
              </Paper>

              {/* Analysis Progress Overlay */}
              {analysisResults.some(a => a.status === 'processing') && (
                <Paper
                  sx={{
                    position: 'absolute',
                    bottom: 16,
                    left: 16,
                    p: 2,
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    minWidth: 250,
                  }}
                >
                  <Typography variant="subtitle2" gutterBottom>
                    Analysis in Progress
                  </Typography>
                  {analysisResults
                    .filter(a => a.status === 'processing')
                    .map((analysis) => (
                      <Box key={analysis.id} sx={{ mb: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                          <Typography variant="caption">{analysis.analysisType}</Typography>
                          <Typography variant="caption">{analysis.progress}%</Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={analysis.progress}
                          sx={{ height: 4, borderRadius: 2 }}
                        />
                      </Box>
                    ))}
                </Paper>
              )}
            </Box>
          </CardContent>
        </Card>
      </Box>

      {/* Right Panel */}
      <Box sx={{ width: 350, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {/* Feature Details */}
        {selectedFeatureData && (
          <Card>
            <CardHeader
              title={selectedFeatureData.name}
              subheader={`${selectedFeatureData.type.charAt(0).toUpperCase() + selectedFeatureData.type.slice(1)} Feature`}
              avatar={
                <Avatar sx={{ bgcolor: 'primary.main' }}>
                  {getTerrainTypeIcon(selectedFeatureData.type)}
                </Avatar>
              }
            />
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {selectedFeatureData.description}
                  </Typography>
                </Box>

                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">
                      Elevation
                    </Typography>
                    <Typography variant="body2" fontWeight={600}>
                      {selectedFeatureData.elevation > 0 ? '+' : ''}{selectedFeatureData.elevation.toLocaleString()}m
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">
                      Area
                    </Typography>
                    <Typography variant="body2" fontWeight={600}>
                      {selectedFeatureData.size.toLocaleString()} km²
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="caption" color="text.secondary">
                      Coordinates
                    </Typography>
                    <Typography variant="body2" fontWeight={600}>
                      {selectedFeatureData.coordinates[0].toFixed(2)}°, {selectedFeatureData.coordinates[1].toFixed(2)}°
                    </Typography>
                  </Grid>
                </Grid>

                <Box>
                  <Typography variant="caption" color="text.secondary" gutterBottom>
                    Traversal Difficulty
                  </Typography>
                  <Chip
                    label={selectedFeatureData.difficulty.toUpperCase()}
                    color={getDifficultyColor(selectedFeatureData.difficulty) as any}
                    size="small"
                  />
                </Box>

                {selectedFeatureData.hazards.length > 0 && (
                  <Box>
                    <Typography variant="caption" color="text.secondary" gutterBottom>
                      Known Hazards
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selectedFeatureData.hazards.map((hazard) => (
                        <Chip
                          key={hazard}
                          label={hazard.replace('_', ' ')}
                          size="small"
                          variant="outlined"
                          color="warning"
                        />
                      ))}
                    </Box>
                  </Box>
                )}

                {selectedFeatureData.mineralContent.length > 0 && (
                  <Box>
                    <Typography variant="caption" color="text.secondary" gutterBottom>
                      Mineral Content
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selectedFeatureData.mineralContent.map((mineral) => (
                        <Chip
                          key={mineral}
                          label={mineral.replace('_', ' ')}
                          size="small"
                          variant="outlined"
                          color="info"
                        />
                      ))}
                    </Box>
                  </Box>
                )}

                <Button
                  variant="outlined"
                  fullWidth
                  startIcon={<AssessmentIcon />}
                  onClick={() => setAnalysisDialogOpen(true)}
                >
                  Analyze This Feature
                </Button>
              </Box>
            </CardContent>
          </Card>
        )}

        {/* Layer Controls */}
        <Card>
          <CardHeader
            title="Analysis Layers"
            titleTypographyProps={{ variant: 'h6' }}
          />
          <CardContent>
            <List dense>
              {layers.map((layer) => (
                <ListItem key={layer.id} sx={{ px: 0 }}>
                  <Box sx={{ width: '100%' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={layer.visible}
                            onChange={() => handleLayerToggle(layer.id)}
                          />
                        }
                        label={layer.name}
                        sx={{ flex: 1 }}
                      />
                      <IconButton size="small">
                        <VisibilityIcon />
                      </IconButton>
                    </Box>
                    {layer.visible && (
                      <Box sx={{ px: 3 }}>
                        <Typography variant="caption" gutterBottom>
                          Opacity: {Math.round(layer.opacity * 100)}%
                        </Typography>
                        <Slider
                          value={layer.opacity * 100}
                          onChange={(_, value) => handleLayerOpacity(layer.id, value as number)}
                          size="small"
                          sx={{ mt: 1 }}
                        />
                      </Box>
                    )}
                  </Box>
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>

        {/* Analysis Results */}
        <Card sx={{ flex: 1 }}>
          <CardHeader
            title="Analysis Results"
            titleTypographyProps={{ variant: 'h6' }}
          />
          <CardContent sx={{ height: 'calc(100% - 64px)', overflow: 'auto' }}>
            <List>
              {analysisResults.map((result) => (
                <ListItem key={result.id} sx={{ px: 0, mb: 2 }}>
                  <Paper sx={{ width: '100%', p: 2 }} variant="outlined">
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="subtitle2">{result.analysisType}</Typography>
                      <Chip
                        label={result.status.toUpperCase()}
                        size="small"
                        color={
                          result.status === 'completed' ? 'success' :
                          result.status === 'processing' ? 'info' : 'default'
                        }
                      />
                    </Box>
                    <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                      Region: {result.region}
                    </Typography>
                    
                    {result.status === 'processing' && (
                      <Box sx={{ mb: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                          <Typography variant="caption">Progress</Typography>
                          <Typography variant="caption">{result.progress}%</Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={result.progress}
                          sx={{ height: 4, borderRadius: 2 }}
                        />
                      </Box>
                    )}
                    
                    {result.findings && (
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        {result.findings}
                      </Typography>
                    )}
                    
                    {result.status === 'completed' && result.confidence > 0 && (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <Typography variant="caption">Confidence:</Typography>
                        <LinearProgress
                          variant="determinate"
                          value={result.confidence}
                          sx={{ flex: 1, height: 4 }}
                          color={result.confidence > 80 ? 'success' : 'warning'}
                        />
                        <Typography variant="caption">{result.confidence}%</Typography>
                      </Box>
                    )}
                    
                    <Typography variant="caption" color="text.secondary">
                      Started: {result.startTime.toLocaleString()}
                    </Typography>
                  </Paper>
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      </Box>

      {/* Analysis Dialog */}
      <Dialog open={analysisDialogOpen} onClose={() => setAnalysisDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Start New Terrain Analysis</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <FormControl>
              <FormLabel>Analysis Type</FormLabel>
              <RadioGroup
                value={selectedAnalysisType}
                onChange={(e) => setSelectedAnalysisType(e.target.value)}
              >
                <FormControlLabel value="slope" control={<Radio />} label="Slope Stability Analysis" />
                <FormControlLabel value="trafficability" control={<Radio />} label="Rover Trafficability" />
                <FormControlLabel value="landing" control={<Radio />} label="Landing Site Evaluation" />
                <FormControlLabel value="hazard" control={<Radio />} label="Hazard Assessment" />
                <FormControlLabel value="composition" control={<Radio />} label="Surface Composition" />
              </RadioGroup>
            </FormControl>
            
            <TextField
              label="Analysis Region"
              defaultValue={selectedFeatureData?.name || 'Custom Region'}
              fullWidth
            />
            
            <TextField
              label="Priority"
              select
              defaultValue="medium"
              SelectProps={{ native: true }}
              fullWidth
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="critical">Critical</option>
            </TextField>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAnalysisDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={startNewAnalysis}>
            Start Analysis
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};
