import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  LinearProgress,
  Chip,
  Button,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Alert,
  Tabs,
  Tab,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Schedule as ScheduleIcon,
  Place as LocationIcon,
  Speed as SpeedIcon,
  Thermostat as ThermostatIcon,
  Air as WindIcon,
  Visibility as VisibilityIcon,
  BatteryFull as BatteryIcon,
  SignalCellularAlt as SignalIcon,
} from '@mui/icons-material';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`dashboard-tabpanel-${index}`}
      aria-labelledby={`dashboard-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

// Mock data for various dashboard metrics
const weatherData = [
  { time: '00:00', temperature: -78, pressure: 0.636, wind: 12 },
  { time: '04:00', temperature: -82, pressure: 0.634, wind: 8 },
  { time: '08:00', temperature: -75, pressure: 0.638, wind: 15 },
  { time: '12:00', temperature: -45, pressure: 0.642, wind: 22 },
  { time: '16:00', temperature: -38, pressure: 0.645, wind: 28 },
  { time: '20:00', temperature: -65, pressure: 0.641, wind: 18 },
];

const missionData = [
  { sol: 1230, distance: 125.4, samples: 3, battery: 85 },
  { sol: 1231, distance: 87.2, samples: 2, battery: 82 },
  { sol: 1232, distance: 203.1, samples: 5, battery: 78 },
  { sol: 1233, distance: 156.8, samples: 4, battery: 75 },
  { sol: 1234, distance: 98.5, samples: 2, battery: 73 },
];

const terrainDistribution = [
  { name: 'Rocky Plains', value: 35, color: '#8b5a3a' },
  { name: 'Sand Dunes', value: 28, color: '#d4a574' },
  { name: 'Crater Walls', value: 18, color: '#666666' },
  { name: 'Volcanic', value: 12, color: '#8b0000' },
  { name: 'Ice Deposits', value: 7, color: '#87ceeb' },
];

const activeMissions = [
  {
    id: 'rover-alpha',
    name: 'Rover Alpha',
    status: 'active',
    location: 'Olympia Undae',
    battery: 73,
    lastContact: '2 mins ago',
    avatar: 'RA',
  },
  {
    id: 'rover-beta',
    name: 'Rover Beta',
    status: 'charging',
    location: 'Acidalia Planitia',
    battery: 45,
    lastContact: '15 mins ago',
    avatar: 'RB',
  },
  {
    id: 'orbit-gamma',
    name: 'Orbiter Gamma',
    status: 'active',
    location: 'Orbital Survey',
    battery: 92,
    lastContact: '1 min ago',
    avatar: 'OG',
  },
  {
    id: 'lander-delta',
    name: 'Lander Delta',
    status: 'maintenance',
    location: 'Chryse Planitia',
    battery: 0,
    lastContact: '2 hours ago',
    avatar: 'LD',
  },
];

const getStatusColor = (status: string) => {
  switch (status) {
    case 'active': return 'success';
    case 'charging': return 'warning';
    case 'maintenance': return 'error';
    default: return 'default';
  }
};

export const Dashboard: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [currentSol, setCurrentSol] = useState(1234);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Box sx={{ p: 3, backgroundColor: 'background.default', minHeight: '100%' }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" fontWeight={700} color="primary">
            Mission Control Dashboard
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="h6" color="text.secondary">
              Sol {currentSol} • {currentTime.toLocaleTimeString()}
            </Typography>
            <IconButton color="primary">
              <RefreshIcon />
            </IconButton>
          </Box>
        </Box>
        
        {/* Critical Alerts */}
        <Alert severity="warning" sx={{ mb: 2 }}>
          <strong>Weather Alert:</strong> Dust storm approaching Olympia Undae region. 
          Rover Alpha operations may be affected. Estimated arrival: 6 hours.
        </Alert>
      </Box>

      {/* Main Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Active Missions
                  </Typography>
                  <Typography variant="h4" color="primary">
                    3
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingUpIcon color="success" fontSize="small" />
                    <Typography variant="body2" color="success.main" sx={{ ml: 0.5 }}>
                      +1 this sol
                    </Typography>
                  </Box>
                </Box>
                <Avatar sx={{ bgcolor: 'primary.main', width: 56, height: 56 }}>
                  <ScheduleIcon />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Distance Traveled
                  </Typography>
                  <Typography variant="h4" color="primary">
                    2.3 km
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingUpIcon color="success" fontSize="small" />
                    <Typography variant="body2" color="success.main" sx={{ ml: 0.5 }}>
                      +98.5m today
                    </Typography>
                  </Box>
                </Box>
                <Avatar sx={{ bgcolor: 'success.main', width: 56, height: 56 }}>
                  <SpeedIcon />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Samples Collected
                  </Typography>
                  <Typography variant="h4" color="primary">
                    47
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingUpIcon color="success" fontSize="small" />
                    <Typography variant="body2" color="success.main" sx={{ ml: 0.5 }}>
                      +2 today
                    </Typography>
                  </Box>
                </Box>
                <Avatar sx={{ bgcolor: 'warning.main', width: 56, height: 56 }}>
                  <LocationIcon />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    System Health
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    94%
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingDownIcon color="error" fontSize="small" />
                    <Typography variant="body2" color="error.main" sx={{ ml: 0.5 }}>
                      -2% this sol
                    </Typography>
                  </Box>
                </Box>
                <Avatar sx={{ bgcolor: 'info.main', width: 56, height: 56 }}>
                  <CheckCircleIcon />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabbed Content */}
      <Card>
        <CardHeader
          title="Mission Operations"
          action={
            <Tabs value={tabValue} onChange={handleTabChange}>
              <Tab label="Weather" />
              <Tab label="Missions" />
              <Tab label="Terrain" />
              <Tab label="Assets" />
            </Tabs>
          }
        />
        <CardContent>
          <TabPanel value={tabValue} index={0}>
            {/* Weather Tab */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Typography variant="h6" gutterBottom>
                  Environmental Conditions (Last 24 Hours)
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={weatherData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis yAxisId="temp" orientation="left" />
                    <YAxis yAxisId="pressure" orientation="right" />
                    <Tooltip />
                    <Line
                      yAxisId="temp"
                      type="monotone"
                      dataKey="temperature"
                      stroke="#ff6b35"
                      strokeWidth={2}
                      name="Temperature (°C)"
                    />
                    <Line
                      yAxisId="pressure"
                      type="monotone"
                      dataKey="pressure"
                      stroke="#2196f3"
                      strokeWidth={2}
                      name="Pressure (kPa)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom>
                  Current Conditions
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <ThermostatIcon color="primary" />
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Temperature
                      </Typography>
                      <Typography variant="h6">-38°C</Typography>
                    </Box>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <WindIcon color="primary" />
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Wind Speed
                      </Typography>
                      <Typography variant="h6">28 m/s</Typography>
                    </Box>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <VisibilityIcon color="primary" />
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Visibility
                      </Typography>
                      <Typography variant="h6">12.5 km</Typography>
                    </Box>
                  </Box>
                </Box>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {/* Mission Progress Tab */}
            <Typography variant="h6" gutterBottom>
              Mission Progress (Last 5 Sols)
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={missionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="sol" />
                <YAxis />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="distance"
                  stackId="1"
                  stroke="#ff6b35"
                  fill="#ff6b35"
                  fillOpacity={0.6}
                  name="Distance (m)"
                />
                <Area
                  type="monotone"
                  dataKey="samples"
                  stackId="2"
                  stroke="#2196f3"
                  fill="#2196f3"
                  fillOpacity={0.6}
                  name="Samples"
                />
              </AreaChart>
            </ResponsiveContainer>
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            {/* Terrain Analysis Tab */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Terrain Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={terrainDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {terrainDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Recent Discoveries
                </Typography>
                <List>
                  <ListItem>
                    <ListItemAvatar>
                      <Avatar sx={{ bgcolor: 'success.main' }}>
                        <CheckCircleIcon />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary="Mineral Deposit Identified"
                      secondary="High iron content detected in sector 7-G"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemAvatar>
                      <Avatar sx={{ bgcolor: 'info.main' }}>
                        <LocationIcon />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary="Geological Formation"
                      secondary="Unusual rock stratification in crater rim"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemAvatar>
                      <Avatar sx={{ bgcolor: 'warning.main' }}>
                        <WarningIcon />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary="Subsurface Anomaly"
                      secondary="Radar indicates possible void at 2m depth"
                    />
                  </ListItem>
                </List>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            {/* Mission Assets Tab */}
            <Typography variant="h6" gutterBottom>
              Active Mission Assets
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Asset</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Location</TableCell>
                    <TableCell>Battery</TableCell>
                    <TableCell>Last Contact</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {activeMissions.map((mission) => (
                    <TableRow key={mission.id}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                          <Avatar sx={{ width: 32, height: 32 }}>
                            {mission.avatar}
                          </Avatar>
                          <Typography variant="body2" fontWeight={600}>
                            {mission.name}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={mission.status.toUpperCase()}
                          size="small"
                          color={getStatusColor(mission.status) as any}
                        />
                      </TableCell>
                      <TableCell>{mission.location}</TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={mission.battery}
                            sx={{ width: 60, height: 6 }}
                          />
                          <Typography variant="body2">{mission.battery}%</Typography>
                        </Box>
                      </TableCell>
                      <TableCell>{mission.lastContact}</TableCell>
                      <TableCell>
                        <Button size="small" variant="outlined">
                          Details
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>
        </CardContent>
      </Card>
    </Box>
  );
};
