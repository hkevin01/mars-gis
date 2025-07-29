import {
    CameraAlt as CameraIcon,
    Fullscreen as FullscreenIcon,
    Info as InfoIcon,
    Layers as LayersIcon,
    Navigation as NavigationIcon,
    Refresh as RefreshIcon,
    Satellite as SatelliteIcon,
    Save as SaveIcon,
    Share as ShareIcon,
    Terrain as TerrainIcon
} from '@mui/icons-material';
import {
    Box,
    Button,
    Card,
    CardContent,
    CardHeader,
    Chip,
    Divider,
    FormControl,
    FormControlLabel,
    FormLabel,
    IconButton,
    Paper,
    Radio,
    RadioGroup,
    Slider,
    SpeedDial,
    SpeedDialAction,
    SpeedDialIcon,
    Switch,
    Tooltip,
    Typography,
} from '@mui/material';
import React, { useEffect, useRef, useState } from 'react';

interface ViewerControls {
  elevation: number;
  azimuth: number;
  zoom: number;
  layer: 'terrain' | 'satellite' | 'hybrid';
  showGrid: boolean;
  showPaths: boolean;
  showMarkers: boolean;
  quality: 'low' | 'medium' | 'high';
}

interface MarsMission {
  id: string;
  name: string;
  position: [number, number, number];
  status: 'active' | 'inactive';
  type: 'rover' | 'lander' | 'orbiter';
}

const mockMissions: MarsMission[] = [
  {
    id: 'rover-alpha',
    name: 'Rover Alpha',
    position: [-14.5684, 175.4729, 0],
    status: 'active',
    type: 'rover',
  },
  {
    id: 'rover-beta',
    name: 'Rover Beta',
    position: [22.4821, 49.9735, 0],
    status: 'active',
    type: 'rover',
  },
  {
    id: 'lander-gamma',
    name: 'Lander Gamma',
    position: [4.5895, 137.4417, 0],
    status: 'inactive',
    type: 'lander',
  },
];

export const Mars3DViewer: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedMission, setSelectedMission] = useState<string | null>(null);
  const [speedDialOpen, setSpeedDialOpen] = useState(false);
  
  const [controls, setControls] = useState<ViewerControls>({
    elevation: 45,
    azimuth: 0,
    zoom: 1.0,
    layer: 'hybrid',
    showGrid: true,
    showPaths: true,
    showMarkers: true,
    quality: 'medium',
  });

  useEffect(() => {
    // Initialize Three.js scene
    initializeViewer();
    
    // Simulate loading
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    // Update viewer when controls change
    updateViewer();
  }, [controls]);

  const initializeViewer = () => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    
    if (!canvas || !container) return;

    // Initialize Three.js scene here
    // This would include:
    // - Scene setup
    // - Mars sphere geometry
    // - Texture loading
    // - Lighting setup
    // - Camera controls
    // - Mission markers
    
    console.log('Initializing Mars 3D Viewer...');
    
    // Mock canvas context for demonstration
    const ctx = canvas.getContext('2d');
    if (ctx) {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      
      // Draw a mock Mars surface
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const radius = Math.min(centerX, centerY) * 0.8;
      
      // Create radial gradient for Mars appearance
      const gradient = ctx.createRadialGradient(
        centerX - radius * 0.3,
        centerY - radius * 0.3,
        0,
        centerX,
        centerY,
        radius
      );
      gradient.addColorStop(0, '#ff8c42');
      gradient.addColorStop(0.5, '#cd6133');
      gradient.addColorStop(1, '#8b4513');
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
      ctx.fill();
      
      // Add some surface features
      ctx.fillStyle = 'rgba(139, 69, 19, 0.3)';
      for (let i = 0; i < 20; i++) {
        const x = centerX + (Math.random() - 0.5) * radius * 1.5;
        const y = centerY + (Math.random() - 0.5) * radius * 1.5;
        const r = Math.random() * 20 + 5;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fill();
      }
      
      // Add mission markers
      mockMissions.forEach((mission, index) => {
        const angle = (index * 120) * (Math.PI / 180);
        const x = centerX + Math.cos(angle) * radius * 0.7;
        const y = centerY + Math.sin(angle) * radius * 0.7;
        
        // Marker
        ctx.fillStyle = mission.status === 'active' ? '#4caf50' : '#f44336';
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        // Label
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(mission.name, x, y - 15);
      });
    }
  };

  const updateViewer = () => {
    // Update the 3D viewer based on control changes
    console.log('Updating viewer with controls:', controls);
    
    // Re-render the scene with new parameters
    initializeViewer();
  };

  const handleControlChange = <K extends keyof ViewerControls>(
    key: K,
    value: ViewerControls[K]
  ) => {
    setControls(prev => ({ ...prev, [key]: value }));
  };

  const handleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const handleCapture = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const link = document.createElement('a');
      link.download = `mars-view-${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
    }
  };

  const speedDialActions = [
    { icon: <CameraIcon />, name: 'Capture', action: handleCapture },
    { icon: <SaveIcon />, name: 'Save View', action: () => console.log('Save view') },
    { icon: <ShareIcon />, name: 'Share', action: () => console.log('Share view') },
    { icon: <InfoIcon />, name: 'Info', action: () => console.log('Show info') },
  ];

  return (
    <Box sx={{ p: 3, height: '100%', display: 'flex', gap: 2 }}>
      {/* Main 3D Viewer */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <Card sx={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          <CardHeader
            title="Mars 3D Globe Viewer"
            action={
              <Box>
                <Tooltip title="Refresh View">
                  <IconButton onClick={() => initializeViewer()}>
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}>
                  <IconButton onClick={handleFullscreen}>
                    <FullscreenIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
          <CardContent sx={{ p: 0, height: 'calc(100% - 64px)', position: 'relative' }}>
            <Box
              ref={containerRef}
              sx={{
                width: '100%',
                height: '100%',
                position: 'relative',
                backgroundColor: '#0a0a0a',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <canvas
                ref={canvasRef}
                style={{
                  width: '100%',
                  height: '100%',
                  cursor: 'grab',
                }}
              />
              
              {isLoading && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    color: 'white',
                    textAlign: 'center',
                  }}
                >
                  <Typography variant="h6" gutterBottom>
                    Loading Mars Surface Data...
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Rendering high-resolution terrain
                  </Typography>
                </Box>
              )}

              {/* Mission Info Overlay */}
              {selectedMission && (
                <Paper
                  sx={{
                    position: 'absolute',
                    top: 20,
                    left: 20,
                    p: 2,
                    maxWidth: 300,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    color: 'white',
                  }}
                >
                  <Typography variant="h6" gutterBottom>
                    Mission Details
                  </Typography>
                  <Typography variant="body2">
                    Selected: {mockMissions.find(m => m.id === selectedMission)?.name}
                  </Typography>
                </Paper>
              )}

              {/* Coordinate Display */}
              <Paper
                sx={{
                  position: 'absolute',
                  bottom: 20,
                  right: 20,
                  p: 1,
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  color: 'white',
                }}
              >
                <Typography variant="caption">
                  Lat: {controls.elevation.toFixed(2)}° | Lon: {controls.azimuth.toFixed(2)}°
                </Typography>
              </Paper>
            </Box>

            {/* Speed Dial */}
            <SpeedDial
              ariaLabel="Viewer Actions"
              sx={{ position: 'absolute', bottom: 16, right: 16 }}
              icon={<SpeedDialIcon />}
              open={speedDialOpen}
              onClose={() => setSpeedDialOpen(false)}
              onOpen={() => setSpeedDialOpen(true)}
            >
              {speedDialActions.map((action) => (
                <SpeedDialAction
                  key={action.name}
                  icon={action.icon}
                  tooltipTitle={action.name}
                  onClick={action.action}
                />
              ))}
            </SpeedDial>
          </CardContent>
        </Card>
      </Box>

      {/* Control Panel */}
      <Box sx={{ width: 300, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {/* View Controls */}
        <Card>
          <CardHeader
            title="View Controls"
            titleTypographyProps={{ variant: 'h6' }}
          />
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <Box>
                <Typography gutterBottom>Elevation</Typography>
                <Slider
                  value={controls.elevation}
                  onChange={(_, value) => handleControlChange('elevation', value as number)}
                  min={-90}
                  max={90}
                  step={1}
                  marks={[
                    { value: -90, label: '-90°' },
                    { value: 0, label: '0°' },
                    { value: 90, label: '90°' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Box>

              <Box>
                <Typography gutterBottom>Azimuth</Typography>
                <Slider
                  value={controls.azimuth}
                  onChange={(_, value) => handleControlChange('azimuth', value as number)}
                  min={0}
                  max={360}
                  step={1}
                  marks={[
                    { value: 0, label: '0°' },
                    { value: 90, label: '90°' },
                    { value: 180, label: '180°' },
                    { value: 270, label: '270°' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Box>

              <Box>
                <Typography gutterBottom>Zoom Level</Typography>
                <Slider
                  value={controls.zoom}
                  onChange={(_, value) => handleControlChange('zoom', value as number)}
                  min={0.1}
                  max={5.0}
                  step={0.1}
                  marks={[
                    { value: 0.1, label: '0.1x' },
                    { value: 1, label: '1x' },
                    { value: 5, label: '5x' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Box>
            </Box>
          </CardContent>
        </Card>

        {/* Layer Controls */}
        <Card>
          <CardHeader
            title="Layer Options"
            titleTypographyProps={{ variant: 'h6' }}
          />
          <CardContent>
            <FormControl component="fieldset">
              <FormLabel component="legend">Base Layer</FormLabel>
              <RadioGroup
                value={controls.layer}
                onChange={(e) => handleControlChange('layer', e.target.value as ViewerControls['layer'])}
              >
                <FormControlLabel
                  value="terrain"
                  control={<Radio />}
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <TerrainIcon fontSize="small" />
                      Terrain Only
                    </Box>
                  }
                />
                <FormControlLabel
                  value="satellite"
                  control={<Radio />}
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <SatelliteIcon fontSize="small" />
                      Satellite Imagery
                    </Box>
                  }
                />
                <FormControlLabel
                  value="hybrid"
                  control={<Radio />}
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LayersIcon fontSize="small" />
                      Hybrid View
                    </Box>
                  }
                />
              </RadioGroup>
            </FormControl>

            <Divider sx={{ my: 2 }} />

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2">Show Grid</Typography>
                <Switch
                  checked={controls.showGrid}
                  onChange={(e) => handleControlChange('showGrid', e.target.checked)}
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2">Show Mission Paths</Typography>
                <Switch
                  checked={controls.showPaths}
                  onChange={(e) => handleControlChange('showPaths', e.target.checked)}
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2">Show Markers</Typography>
                <Switch
                  checked={controls.showMarkers}
                  onChange={(e) => handleControlChange('showMarkers', e.target.checked)}
                />
              </Box>
            </Box>
          </CardContent>
        </Card>

        {/* Mission List */}
        <Card>
          <CardHeader
            title="Active Missions"
            titleTypographyProps={{ variant: 'h6' }}
          />
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {mockMissions.map((mission) => (
                <Button
                  key={mission.id}
                  variant={selectedMission === mission.id ? 'contained' : 'outlined'}
                  onClick={() => setSelectedMission(
                    selectedMission === mission.id ? null : mission.id
                  )}
                  sx={{ justifyContent: 'space-between' }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <NavigationIcon fontSize="small" />
                    {mission.name}
                  </Box>
                  <Chip
                    label={mission.status}
                    size="small"
                    color={mission.status === 'active' ? 'success' : 'default'}
                  />
                </Button>
              ))}
            </Box>
          </CardContent>
        </Card>

        {/* Performance Settings */}
        <Card>
          <CardHeader
            title="Performance"
            titleTypographyProps={{ variant: 'h6' }}
          />
          <CardContent>
            <FormControl component="fieldset">
              <FormLabel component="legend">Render Quality</FormLabel>
              <RadioGroup
                value={controls.quality}
                onChange={(e) => handleControlChange('quality', e.target.value as ViewerControls['quality'])}
              >
                <FormControlLabel value="low" control={<Radio />} label="Low (Fast)" />
                <FormControlLabel value="medium" control={<Radio />} label="Medium" />
                <FormControlLabel value="high" control={<Radio />} label="High (Detailed)" />
              </RadioGroup>
            </FormControl>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};
