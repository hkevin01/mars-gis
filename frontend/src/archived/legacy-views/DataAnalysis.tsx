import {
    Analytics as AnalyticsIcon,
    Assessment as AssessmentIcon,
    CheckCircle as CheckCircleIcon,
    Download as DownloadIcon,
    FilterList as FilterIcon,
    Info as InfoIcon,
    Refresh as RefreshIcon,
    Science as ScienceIcon,
    TrendingUp as TrendingUpIcon,
    Warning as WarningIcon
} from '@mui/icons-material';
import {
    Alert,
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
    Grid,
    IconButton,
    InputLabel,
    LinearProgress,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    MenuItem,
    Paper,
    Select,
    Tab,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Tabs,
    TextField,
    Tooltip,
    Typography
} from '@mui/material';
import React, { useState } from 'react';
import {
    Area,
    AreaChart,
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    Line,
    LineChart,
    Pie,
    PieChart,
    Tooltip as RechartsTooltip,
    ResponsiveContainer,
    XAxis,
    YAxis
} from 'recharts';

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
      id={`analysis-tabpanel-${index}`}
      aria-labelledby={`analysis-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

// Mock data for various analyses
const atmosphericData = [
  { sol: 1230, pressure: 0.636, temperature: -78, humidity: 0.02, windSpeed: 12 },
  { sol: 1231, pressure: 0.634, temperature: -82, humidity: 0.01, windSpeed: 8 },
  { sol: 1232, pressure: 0.638, temperature: -75, humidity: 0.03, windSpeed: 15 },
  { sol: 1233, pressure: 0.642, temperature: -45, humidity: 0.04, windSpeed: 22 },
  { sol: 1234, pressure: 0.645, temperature: -38, humidity: 0.05, windSpeed: 28 },
];

const geologicalData = [
  { sample: 'Sample A-1', silicate: 45, iron: 18, magnesium: 12, calcium: 8, aluminum: 9, other: 8 },
  { sample: 'Sample A-2', silicate: 42, iron: 22, magnesium: 11, calcium: 7, aluminum: 10, other: 8 },
  { sample: 'Sample A-3', silicate: 48, iron: 16, magnesium: 13, calcium: 9, aluminum: 8, other: 6 },
  { sample: 'Sample A-4', silicate: 44, iron: 20, magnesium: 12, calcium: 8, aluminum: 9, other: 7 },
];

const terrainAnalysis = [
  { region: 'Olympia Undae', rocky: 35, sandy: 45, volcanic: 15, icy: 5 },
  { region: 'Acidalia Planitia', rocky: 60, sandy: 25, volcanic: 10, icy: 5 },
  { region: 'Chryse Planitia', rocky: 40, sandy: 35, volcanic: 20, icy: 5 },
  { region: 'Utopia Planitia', rocky: 30, sandy: 40, volcanic: 25, icy: 5 },
];

const missionMetrics = [
  {
    id: 'distance',
    name: 'Total Distance Traveled',
    value: '12.7 km',
    trend: '+2.3%',
    status: 'good',
    description: 'Cumulative distance across all active missions',
  },
  {
    id: 'samples',
    name: 'Samples Collected',
    value: '47',
    trend: '+8.7%',
    status: 'excellent',
    description: 'Total geological and atmospheric samples',
  },
  {
    id: 'uptime',
    name: 'System Uptime',
    value: '94.2%',
    trend: '-1.2%',
    status: 'warning',
    description: 'Average uptime across all mission assets',
  },
  {
    id: 'efficiency',
    name: 'Mission Efficiency',
    value: '87%',
    trend: '+5.1%',
    status: 'good',
    description: 'Ratio of completed vs planned objectives',
  },
];

const analysisResults = [
  {
    id: 'mineral-comp',
    title: 'Mineral Composition Analysis',
    status: 'completed',
    confidence: 94,
    findings: 'High iron content detected in 78% of samples',
    samples: 23,
    startDate: '2024-01-15',
    completedDate: '2024-01-18',
  },
  {
    id: 'terrain-class',
    title: 'Terrain Classification',
    status: 'processing',
    confidence: 76,
    findings: 'ML model identifies 5 distinct terrain types',
    samples: 156,
    startDate: '2024-01-16',
    completedDate: null,
  },
  {
    id: 'weather-pred',
    title: 'Weather Pattern Prediction',
    status: 'completed',
    confidence: 89,
    findings: 'Dust storm probability increased by 23%',
    samples: 87,
    startDate: '2024-01-12',
    completedDate: '2024-01-19',
  },
];

const COLORS = ['#ff6b35', '#2196f3', '#4caf50', '#ff9800', '#9c27b0', '#f44336'];

export const DataAnalysis: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [selectedTimeRange, setSelectedTimeRange] = useState('7d');
  const [filterDialogOpen, setFilterDialogOpen] = useState(false);
  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'success';
      case 'good': return 'info';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getAnalysisStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'info';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box sx={{ p: 3, backgroundColor: 'background.default', minHeight: '100%' }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" fontWeight={700} color="primary">
            Data Analysis Center
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Time Range</InputLabel>
              <Select
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value)}
                label="Time Range"
              >
                <MenuItem value="24h">Last 24 Hours</MenuItem>
                <MenuItem value="7d">Last 7 Days</MenuItem>
                <MenuItem value="30d">Last 30 Days</MenuItem>
                <MenuItem value="90d">Last 90 Days</MenuItem>
              </Select>
            </FormControl>
            <Tooltip title="Filter Data">
              <IconButton onClick={() => setFilterDialogOpen(true)}>
                <FilterIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Refresh Data">
              <IconButton color="primary">
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            <Button variant="contained" startIcon={<DownloadIcon />}>
              Export Report
            </Button>
          </Box>
        </Box>
      </Box>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {missionMetrics.map((metric) => (
          <Grid item xs={12} sm={6} md={3} key={metric.id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                  <Typography color="textSecondary" variant="body2">
                    {metric.name}
                  </Typography>
                  <Chip
                    label={metric.trend}
                    size="small"
                    color={metric.trend.startsWith('+') ? 'success' : 'error'}
                    variant="outlined"
                  />
                </Box>
                <Typography variant="h4" color="primary" gutterBottom>
                  {metric.value}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {metric.description}
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <LinearProgress
                    variant="determinate"
                    value={75}
                    color={getStatusColor(metric.status) as any}
                    sx={{ height: 4, borderRadius: 2 }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Main Analysis Tabs */}
      <Card>
        <CardHeader
          title="Analysis Dashboard"
          action={
            <Tabs value={tabValue} onChange={handleTabChange}>
              <Tab label="Atmospheric" icon={<AnalyticsIcon />} />
              <Tab label="Geological" icon={<ScienceIcon />} />
              <Tab label="Terrain" icon={<AssessmentIcon />} />
              <Tab label="AI Results" icon={<TrendingUpIcon />} />
            </Tabs>
          }
        />
        <CardContent>
          <TabPanel value={tabValue} index={0}>
            {/* Atmospheric Analysis */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Typography variant="h6" gutterBottom>
                  Environmental Conditions Trend
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={atmosphericData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="sol" />
                    <YAxis yAxisId="temp" orientation="left" />
                    <YAxis yAxisId="pressure" orientation="right" />
                    <RechartsTooltip />
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
                    <Line
                      yAxisId="temp"
                      type="monotone"
                      dataKey="windSpeed"
                      stroke="#4caf50"
                      strokeWidth={2}
                      name="Wind Speed (m/s)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom>
                  Current Conditions
                </Typography>
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <InfoIcon color="info" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Temperature"
                      secondary="-38°C (Optimal range)"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircleIcon color="success" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Atmospheric Pressure"
                      secondary="0.645 kPa (Stable)"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <WarningIcon color="warning" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Wind Speed"
                      secondary="28 m/s (High)"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <InfoIcon color="info" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Humidity"
                      secondary="0.05% (Very Low)"
                    />
                  </ListItem>
                </List>
                
                <Alert severity="warning" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    <strong>Weather Advisory:</strong> Increased wind activity detected. 
                    Consider postponing sensitive operations.
                  </Typography>
                </Alert>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {/* Geological Analysis */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Sample Composition Analysis
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={geologicalData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="sample" />
                    <YAxis />
                    <RechartsTooltip />
                    <Bar dataKey="silicate" stackId="a" fill="#ff6b35" name="Silicate" />
                    <Bar dataKey="iron" stackId="a" fill="#2196f3" name="Iron" />
                    <Bar dataKey="magnesium" stackId="a" fill="#4caf50" name="Magnesium" />
                    <Bar dataKey="calcium" stackId="a" fill="#ff9800" name="Calcium" />
                    <Bar dataKey="aluminum" stackId="a" fill="#9c27b0" name="Aluminum" />
                    <Bar dataKey="other" stackId="a" fill="#757575" name="Other" />
                  </BarChart>
                </ResponsiveContainer>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Mineral Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Silicate', value: 45, color: '#ff6b35' },
                        { name: 'Iron', value: 19, color: '#2196f3' },
                        { name: 'Magnesium', value: 12, color: '#4caf50' },
                        { name: 'Calcium', value: 8, color: '#ff9800' },
                        { name: 'Aluminum', value: 9, color: '#9c27b0' },
                        { name: 'Other', value: 7, color: '#757575' },
                      ]}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {geologicalData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Analysis Summary
                </Typography>
                <TableContainer component={Paper}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Sample ID</TableCell>
                        <TableCell>Primary Mineral</TableCell>
                        <TableCell>Iron Content</TableCell>
                        <TableCell>Collection Date</TableCell>
                        <TableCell>Analysis Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {geologicalData.map((sample, index) => (
                        <TableRow key={sample.sample}>
                          <TableCell>{sample.sample}</TableCell>
                          <TableCell>Silicate ({sample.silicate}%)</TableCell>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <LinearProgress
                                variant="determinate"
                                value={sample.iron}
                                sx={{ width: 60, height: 6 }}
                              />
                              {sample.iron}%
                            </Box>
                          </TableCell>
                          <TableCell>Sol {1230 + index}</TableCell>
                          <TableCell>
                            <Chip
                              label="Complete"
                              size="small"
                              color="success"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            {/* Terrain Analysis */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Typography variant="h6" gutterBottom>
                  Regional Terrain Composition
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={terrainAnalysis}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="region" />
                    <YAxis />
                    <RechartsTooltip />
                    <Area
                      type="monotone"
                      dataKey="rocky"
                      stackId="1"
                      stroke="#8b4513"
                      fill="#8b4513"
                      name="Rocky"
                    />
                    <Area
                      type="monotone"
                      dataKey="sandy"
                      stackId="1"
                      stroke="#d2691e"
                      fill="#d2691e"
                      name="Sandy"
                    />
                    <Area
                      type="monotone"
                      dataKey="volcanic"
                      stackId="1"
                      stroke="#dc143c"
                      fill="#dc143c"
                      name="Volcanic"
                    />
                    <Area
                      type="monotone"
                      dataKey="icy"
                      stackId="1"
                      stroke="#87ceeb"
                      fill="#87ceeb"
                      name="Icy"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom>
                  Terrain Classifications
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {[
                    { name: 'Rocky Plains', percentage: 42, color: '#8b4513' },
                    { name: 'Sand Dunes', percentage: 36, color: '#d2691e' },
                    { name: 'Volcanic Fields', percentage: 17, color: '#dc143c' },
                    { name: 'Ice Deposits', percentage: 5, color: '#87ceeb' },
                  ].map((terrain) => (
                    <Box key={terrain.name}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                        <Typography variant="body2">{terrain.name}</Typography>
                        <Typography variant="body2" fontWeight={600}>
                          {terrain.percentage}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={terrain.percentage}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: terrain.color,
                          },
                        }}
                      />
                    </Box>
                  ))}
                </Box>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            {/* AI Analysis Results */}
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Machine Learning Analysis Results
                </Typography>
                <Grid container spacing={2}>
                  {analysisResults.map((analysis) => (
                    <Grid item xs={12} md={4} key={analysis.id}>
                      <Card 
                        sx={{ 
                          cursor: 'pointer',
                          '&:hover': { elevation: 4 },
                          border: selectedAnalysis === analysis.id ? '2px solid' : '1px solid',
                          borderColor: selectedAnalysis === analysis.id ? 'primary.main' : 'divider',
                        }}
                        onClick={() => setSelectedAnalysis(analysis.id)}
                      >
                        <CardContent>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography variant="h6" fontSize="1rem">
                              {analysis.title}
                            </Typography>
                            <Chip
                              label={analysis.status.toUpperCase()}
                              size="small"
                              color={getAnalysisStatusColor(analysis.status) as any}
                            />
                          </Box>
                          
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            {analysis.findings}
                          </Typography>
                          
                          <Box sx={{ mt: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                              <Typography variant="caption">Confidence</Typography>
                              <Typography variant="caption" fontWeight={600}>
                                {analysis.confidence}%
                              </Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={analysis.confidence}
                              color={analysis.confidence > 80 ? 'success' : analysis.confidence > 60 ? 'info' : 'warning'}
                              sx={{ height: 6, borderRadius: 3 }}
                            />
                          </Box>
                          
                          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="caption" color="text.secondary">
                              {analysis.samples} samples
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              Started: {analysis.startDate}
                            </Typography>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Grid>
            </Grid>
          </TabPanel>
        </CardContent>
      </Card>

      {/* Filter Dialog */}
      <Dialog open={filterDialogOpen} onClose={() => setFilterDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Data Filters</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <FormControl fullWidth>
              <InputLabel>Mission Asset</InputLabel>
              <Select defaultValue="all" label="Mission Asset">
                <MenuItem value="all">All Assets</MenuItem>
                <MenuItem value="rover-alpha">Rover Alpha</MenuItem>
                <MenuItem value="rover-beta">Rover Beta</MenuItem>
                <MenuItem value="lander-gamma">Lander Gamma</MenuItem>
              </Select>
            </FormControl>
            
            <FormControl fullWidth>
              <InputLabel>Data Type</InputLabel>
              <Select defaultValue="all" label="Data Type">
                <MenuItem value="all">All Data Types</MenuItem>
                <MenuItem value="atmospheric">Atmospheric</MenuItem>
                <MenuItem value="geological">Geological</MenuItem>
                <MenuItem value="terrain">Terrain</MenuItem>
              </Select>
            </FormControl>
            
            <TextField
              label="Minimum Confidence"
              type="number"
              defaultValue={70}
              inputProps={{ min: 0, max: 100 }}
              fullWidth
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFilterDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={() => setFilterDialogOpen(false)}>
            Apply Filters
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};
