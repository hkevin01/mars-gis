import {
    ExpandMore as ExpandMoreIcon,
    Fullscreen as FullscreenIcon,
    Layers as LayersIcon,
    LocationOn as LocationIcon,
    MyLocation as MyLocationIcon,
    Search as SearchIcon,
    ZoomIn as ZoomInIcon,
    ZoomOut as ZoomOutIcon
} from '@mui/icons-material';
import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
    Badge,
    Box,
    Button,
    ButtonGroup,
    Card,
    CardContent,
    CardHeader,
    Checkbox,
    Chip,
    Divider,
    Drawer,
    IconButton,
    InputAdornment,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Paper,
    TextField,
    Tooltip,
    Typography
} from '@mui/material';
import React, { useEffect, useRef, useState } from 'react';

interface MapLayer {
  id: string;
  name: string;
  type: 'base' | 'overlay';
  visible: boolean;
  opacity: number;
  category: string;
}

interface MapMarker {
  id: string;
  name: string;
  position: [number, number];
  type: 'mission' | 'landmark' | 'hazard' | 'sample';
  status?: 'active' | 'inactive' | 'completed';
  description: string;
}

interface MapRoute {
  id: string;
  name: string;
  points: [number, number][];
  type: 'planned' | 'completed' | 'active';
  distance: number;
  duration: number;
}

const mapLayers: MapLayer[] = [
  {
    id: 'satellite',
    name: 'Satellite Imagery',
    type: 'base',
    visible: true,
    opacity: 1.0,
    category: 'Base Maps',
  },
  {
    id: 'terrain',
    name: 'Terrain Elevation',
    type: 'base',
    visible: false,
    opacity: 1.0,
    category: 'Base Maps',
  },
  {
    id: 'geological',
    name: 'Geological Features',
    type: 'overlay',
    visible: true,
    opacity: 0.7,
    category: 'Science Data',
  },
  {
    id: 'weather',
    name: 'Weather Patterns',
    type: 'overlay',
    visible: false,
    opacity: 0.5,
    category: 'Environmental',
  },
  {
    id: 'paths',
    name: 'Mission Paths',
    type: 'overlay',
    visible: true,
    opacity: 0.8,
    category: 'Missions',
  },
  {
    id: 'hazards',
    name: 'Hazard Zones',
    type: 'overlay',
    visible: true,
    opacity: 0.6,
    category: 'Safety',
  },
];

const mapMarkers: MapMarker[] = [
  {
    id: 'rover-alpha',
    name: 'Rover Alpha',
    position: [-14.5684, 175.4729],
    type: 'mission',
    status: 'active',
    description: 'Primary exploration rover currently investigating mineral deposits',
  },
  {
    id: 'sample-site-1',
    name: 'Sample Site Alpha-7',
    position: [-14.5720, 175.4800],
    type: 'sample',
    status: 'completed',
    description: 'Rock sample collected, high iron content confirmed',
  },
  {
    id: 'olympus-mons',
    name: 'Olympus Mons',
    position: [18.65, -133.8],
    type: 'landmark',
    description: 'Largest volcano in the solar system, 21.9 km high',
  },
  {
    id: 'dust-storm',
    name: 'Dust Storm Zone',
    position: [10.0, -150.0],
    type: 'hazard',
    description: 'Active dust storm, avoid mission planning in this area',
  },
];

const mapRoutes: MapRoute[] = [
  {
    id: 'route-1',
    name: 'Alpha Exploration Path',
    points: [
      [-14.5684, 175.4729],
      [-14.5720, 175.4800],
      [-14.5750, 175.4850],
    ],
    type: 'completed',
    distance: 2.1,
    duration: 3.5,
  },
  {
    id: 'route-2',
    name: 'Planned Survey Route',
    points: [
      [-14.5750, 175.4850],
      [-14.5800, 175.4900],
      [-14.5850, 175.4950],
    ],
    type: 'planned',
    distance: 1.8,
    duration: 2.2,
  },
];

export const InteractiveMap: React.FC = () => {
  const mapRef = useRef<HTMLDivElement>(null);
  const [layersOpen, setLayersOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedMarker, setSelectedMarker] = useState<string | null>(null);
  const [layers, setLayers] = useState<MapLayer[]>(mapLayers);
  const [zoomLevel, setZoomLevel] = useState(5);
  const [center, setCenter] = useState<[number, number]>([0, 0]);

  useEffect(() => {
    initializeMap();
  }, []);

  useEffect(() => {
    updateMapLayers();
  }, [layers]);

  const initializeMap = () => {
    const mapContainer = mapRef.current;
    if (!mapContainer) return;

    // Initialize Leaflet map here
    // This would include:
    // - Map instance creation
    // - Base layer setup
    // - Marker rendering
    // - Route visualization
    // - Interactive controls

    console.log('Initializing interactive Mars map...');

    // Mock map rendering
    mapContainer.style.backgroundColor = '#8b4513';
    mapContainer.style.backgroundImage = `
      radial-gradient(circle at 20% 30%, rgba(205, 97, 51, 0.3) 0%, transparent 50%),
      radial-gradient(circle at 80% 70%, rgba(139, 69, 19, 0.3) 0%, transparent 50%),
      radial-gradient(circle at 40% 80%, rgba(160, 82, 45, 0.2) 0%, transparent 50%)
    `;

    // Add some visual elements to represent the map
    const existingElements = mapContainer.querySelectorAll('.map-element');
    existingElements.forEach(el => el.remove());

    // Add markers
    mapMarkers.forEach((marker, index) => {
      const markerEl = document.createElement('div');
      markerEl.className = 'map-element marker';
      markerEl.style.position = 'absolute';
      markerEl.style.left = `${30 + index * 15}%`;
      markerEl.style.top = `${20 + index * 10}%`;
      markerEl.style.width = '12px';
      markerEl.style.height = '12px';
      markerEl.style.borderRadius = '50%';
      markerEl.style.backgroundColor = getMarkerColor(marker.type, marker.status);
      markerEl.style.border = '2px solid white';
      markerEl.style.cursor = 'pointer';
      markerEl.style.zIndex = '10';
      markerEl.onclick = () => setSelectedMarker(marker.id);
      mapContainer.appendChild(markerEl);
    });

    // Add routes
    mapRoutes.forEach((route, index) => {
      const routeEl = document.createElement('div');
      routeEl.className = 'map-element route';
      routeEl.style.position = 'absolute';
      routeEl.style.left = `${25 + index * 20}%`;
      routeEl.style.top = `${30 + index * 15}%`;
      routeEl.style.width = '100px';
      routeEl.style.height = '2px';
      routeEl.style.backgroundColor = getRouteColor(route.type);
      routeEl.style.transform = `rotate(${index * 30}deg)`;
      routeEl.style.transformOrigin = 'left center';
      mapContainer.appendChild(routeEl);
    });
  };

  const updateMapLayers = () => {
    console.log('Updating map layers:', layers.filter(l => l.visible));
    // Re-render map with current layer settings
    initializeMap();
  };

  const getMarkerColor = (type: string, status?: string) => {
    if (status === 'active') return '#4caf50';
    if (status === 'inactive') return '#f44336';
    if (status === 'completed') return '#2196f3';
    
    switch (type) {
      case 'mission': return '#ff9800';
      case 'landmark': return '#9c27b0';
      case 'hazard': return '#f44336';
      case 'sample': return '#4caf50';
      default: return '#757575';
    }
  };

  const getRouteColor = (type: string) => {
    switch (type) {
      case 'completed': return '#4caf50';
      case 'active': return '#ff9800';
      case 'planned': return '#2196f3';
      default: return '#757575';
    }
  };

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

  const handleZoom = (direction: 'in' | 'out') => {
    setZoomLevel(prev => {
      if (direction === 'in') return Math.min(prev + 1, 18);
      return Math.max(prev - 1, 1);
    });
  };

  const selectedMarkerData = mapMarkers.find(m => m.id === selectedMarker);

  // Group layers by category
  const layersByCategory = layers.reduce((acc, layer) => {
    if (!acc[layer.category]) acc[layer.category] = [];
    acc[layer.category].push(layer);
    return acc;
  }, {} as Record<string, MapLayer[]>);

  return (
    <Box sx={{ height: '100%', display: 'flex', position: 'relative' }}>
      {/* Main Map Container */}
      <Box sx={{ flex: 1, position: 'relative' }}>
        <Card sx={{ height: '100%', borderRadius: 0 }}>
          <CardHeader
            title="Interactive Mars Map"
            action={
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  size="small"
                  placeholder="Search locations..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <SearchIcon fontSize="small" />
                      </InputAdornment>
                    ),
                  }}
                  sx={{ width: 200 }}
                />
                <Tooltip title="Toggle Layers">
                  <IconButton onClick={() => setLayersOpen(true)}>
                    <Badge badgeContent={layers.filter(l => l.visible).length} color="primary">
                      <LayersIcon />
                    </Badge>
                  </IconButton>
                </Tooltip>
                <Tooltip title="Fullscreen">
                  <IconButton>
                    <FullscreenIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
          <CardContent sx={{ p: 0, height: 'calc(100% - 64px)', position: 'relative' }}>
            {/* Map Canvas */}
            <Box
              ref={mapRef}
              sx={{
                width: '100%',
                height: '100%',
                position: 'relative',
                cursor: 'grab',
                '&:active': { cursor: 'grabbing' },
              }}
            />

            {/* Map Controls */}
            <Box
              sx={{
                position: 'absolute',
                top: 16,
                right: 16,
                display: 'flex',
                flexDirection: 'column',
                gap: 1,
              }}
            >
              <ButtonGroup orientation="vertical" variant="contained" size="small">
                <Button onClick={() => handleZoom('in')}>
                  <ZoomInIcon />
                </Button>
                <Button onClick={() => handleZoom('out')}>
                  <ZoomOutIcon />
                </Button>
              </ButtonGroup>
              <Button variant="contained" size="small" startIcon={<MyLocationIcon />}>
                Center
              </Button>
            </Box>

            {/* Coordinates Display */}
            <Paper
              sx={{
                position: 'absolute',
                bottom: 16,
                left: 16,
                p: 1,
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                color: 'white',
              }}
            >
              <Typography variant="caption">
                Zoom: {zoomLevel} | Center: {center[0].toFixed(4)}°, {center[1].toFixed(4)}°
              </Typography>
            </Paper>

            {/* Scale Bar */}
            <Box
              sx={{
                position: 'absolute',
                bottom: 16,
                right: 16,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <Paper sx={{ p: 1, backgroundColor: 'rgba(0, 0, 0, 0.7)' }}>
                <Typography variant="caption" color="white">
                  0────10────20 km
                </Typography>
              </Paper>
            </Box>

            {/* Selected Marker Info */}
            {selectedMarkerData && (
              <Paper
                sx={{
                  position: 'absolute',
                  top: 16,
                  left: 16,
                  p: 2,
                  maxWidth: 300,
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <LocationIcon color="primary" />
                  <Typography variant="h6">{selectedMarkerData.name}</Typography>
                  {selectedMarkerData.status && (
                    <Chip
                      label={selectedMarkerData.status}
                      size="small"
                      color={
                        selectedMarkerData.status === 'active' ? 'success' :
                        selectedMarkerData.status === 'completed' ? 'info' : 'default'
                      }
                    />
                  )}
                </Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {selectedMarkerData.description}
                </Typography>
                <Typography variant="caption" color="text.disabled">
                  Coordinates: {selectedMarkerData.position[0].toFixed(4)}°, {selectedMarkerData.position[1].toFixed(4)}°
                </Typography>
                <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
                  <Button size="small" variant="outlined">
                    Details
                  </Button>
                  <Button size="small" variant="outlined" onClick={() => setSelectedMarker(null)}>
                    Close
                  </Button>
                </Box>
              </Paper>
            )}
          </CardContent>
        </Card>
      </Box>

      {/* Layers Panel */}
      <Drawer
        anchor="right"
        open={layersOpen}
        onClose={() => setLayersOpen(false)}
        sx={{
          '& .MuiDrawer-paper': {
            width: 320,
            p: 2,
          },
        }}
      >
        <Typography variant="h6" gutterBottom>
          Map Layers
        </Typography>

        {Object.entries(layersByCategory).map(([category, categoryLayers]) => (
          <Accordion key={category} defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2">{category}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <List dense>
                {categoryLayers.map((layer) => (
                  <ListItem key={layer.id} disablePadding>
                    <ListItemButton onClick={() => handleLayerToggle(layer.id)}>
                      <ListItemIcon>
                        <Checkbox
                          checked={layer.visible}
                          onChange={() => handleLayerToggle(layer.id)}
                        />
                      </ListItemIcon>
                      <ListItemText
                        primary={layer.name}
                        secondary={layer.type === 'base' ? 'Base Layer' : 'Overlay'}
                      />
                    </ListItemButton>
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>
        ))}

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle2" gutterBottom>
          Legend
        </Typography>
        <List dense>
          <ListItem>
            <ListItemIcon>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: '#4caf50',
                  border: '2px solid white',
                }}
              />
            </ListItemIcon>
            <ListItemText primary="Active Mission" />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: '#2196f3',
                  border: '2px solid white',
                }}
              />
            </ListItemIcon>
            <ListItemText primary="Completed Sample" />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: '#9c27b0',
                  border: '2px solid white',
                }}
              />
            </ListItemIcon>
            <ListItemText primary="Landmark" />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: '#f44336',
                  border: '2px solid white',
                }}
              />
            </ListItemIcon>
            <ListItemText primary="Hazard Zone" />
          </ListItem>
        </List>
      </Drawer>
    </Box>
  );
};
