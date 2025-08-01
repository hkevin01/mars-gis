import {
    CheckCircle as CheckCircleIcon,
    Cloud as CloudIcon,
    Download as DownloadIcon,
    Network as NetworkIcon,
    Notifications as NotificationsIcon,
    Palette as PaletteIcon,
    Restore as RestoreIcon,
    Save as SaveIcon,
    Settings as SettingsIcon,
    Storage as StorageIcon,
    Computer as SystemIcon,
    Upload as UploadIcon,
    Warning as WarningIcon
} from '@mui/icons-material';
import {
    Alert,
    Box,
    Button,
    Card,
    CardContent,
    CardHeader,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Divider,
    FormControl,
    FormControlLabel,
    Grid,
    InputLabel,
    List,
    ListItem,
    ListItemIcon,
    ListItemSecondaryAction,
    ListItemText,
    MenuItem,
    Select,
    Slider,
    Switch,
    Tab,
    Tabs,
    Typography
} from '@mui/material';
import React, { useState } from 'react';

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
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

interface UserSettings {
  theme: 'light' | 'dark' | 'mars';
  notifications: {
    email: boolean;
    push: boolean;
    desktop: boolean;
    critical: boolean;
  };
  dashboard: {
    refreshInterval: number;
    defaultView: string;
    showWelcome: boolean;
  };
  map: {
    defaultLayer: string;
    quality: 'low' | 'medium' | 'high';
    cacheSize: number;
  };
  analysis: {
    autoSave: boolean;
    maxConcurrent: number;
    confidenceThreshold: number;
  };
}

const defaultSettings: UserSettings = {
  theme: 'mars',
  notifications: {
    email: true,
    push: true,
    desktop: false,
    critical: true,
  },
  dashboard: {
    refreshInterval: 30,
    defaultView: 'dashboard',
    showWelcome: true,
  },
  map: {
    defaultLayer: 'hybrid',
    quality: 'medium',
    cacheSize: 512,
  },
  analysis: {
    autoSave: true,
    maxConcurrent: 3,
    confidenceThreshold: 70,
  },
};

export const Settings: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [settings, setSettings] = useState<UserSettings>(defaultSettings);
  const [resetDialogOpen, setResetDialogOpen] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleSettingChange = <K extends keyof UserSettings>(
    category: K,
    key: keyof UserSettings[K],
    value: any
  ) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: value,
      },
    }));
    setHasChanges(true);
  };

  const handleSaveSettings = () => {
    // Save settings to backend
    console.log('Saving settings:', settings);
    setHasChanges(false);
  };

  const handleResetSettings = () => {
    setSettings(defaultSettings);
    setHasChanges(true);
    setResetDialogOpen(false);
  };

  const handleExportSettings = () => {
    const dataStr = JSON.stringify(settings, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = `mars-gis-settings-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <Box sx={{ p: 3, height: '100%' }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" fontWeight={700} color="primary">
            System Settings
          </Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              variant="outlined"
              startIcon={<DownloadIcon />}
              onClick={handleExportSettings}
            >
              Export
            </Button>
            <Button
              variant="outlined"
              startIcon={<RestoreIcon />}
              onClick={() => setResetDialogOpen(true)}
            >
              Reset
            </Button>
            <Button
              variant="contained"
              startIcon={<SaveIcon />}
              onClick={handleSaveSettings}
              disabled={!hasChanges}
            >
              Save Changes
            </Button>
          </Box>
        </Box>

        {hasChanges && (
          <Alert severity="info" sx={{ mb: 2 }}>
            You have unsaved changes. Click "Save Changes" to apply them.
          </Alert>
        )}
      </Box>

      {/* Settings Tabs */}
      <Card>
        <CardHeader
          title="Configuration"
          action={
            <Tabs value={tabValue} onChange={handleTabChange}>
              <Tab label="General" icon={<SettingsIcon />} />
              <Tab label="Appearance" icon={<PaletteIcon />} />
              <Tab label="Notifications" icon={<NotificationsIcon />} />
              <Tab label="Data & Storage" icon={<StorageIcon />} />
              <Tab label="System" icon={<SystemIcon />} />
            </Tabs>
          }
        />
        <CardContent>
          <TabPanel value={tabValue} index={0}>
            {/* General Settings */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardHeader title="Dashboard Settings" />
                  <CardContent>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      <FormControl fullWidth>
                        <InputLabel>Default View</InputLabel>
                        <Select
                          value={settings.dashboard.defaultView}
                          onChange={(e) => handleSettingChange('dashboard', 'defaultView', e.target.value)}
                          label="Default View"
                        >
                          <MenuItem value="dashboard">Mission Dashboard</MenuItem>
                          <MenuItem value="mars-3d">3D Mars Viewer</MenuItem>
                          <MenuItem value="interactive-map">Interactive Map</MenuItem>
                          <MenuItem value="mission-planner">Mission Planner</MenuItem>
                        </Select>
                      </FormControl>

                      <Box>
                        <Typography gutterBottom>Refresh Interval (seconds)</Typography>
                        <Slider
                          value={settings.dashboard.refreshInterval}
                          onChange={(_, value) => handleSettingChange('dashboard', 'refreshInterval', value)}
                          min={5}
                          max={300}
                          step={5}
                          marks={[
                            { value: 5, label: '5s' },
                            { value: 30, label: '30s' },
                            { value: 60, label: '1m' },
                            { value: 300, label: '5m' },
                          ]}
                          valueLabelDisplay="auto"
                        />
                      </Box>

                      <FormControlLabel
                        control={
                          <Switch
                            checked={settings.dashboard.showWelcome}
                            onChange={(e) => handleSettingChange('dashboard', 'showWelcome', e.target.checked)}
                          />
                        }
                        label="Show welcome message"
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardHeader title="Map Settings" />
                  <CardContent>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      <FormControl fullWidth>
                        <InputLabel>Default Layer</InputLabel>
                        <Select
                          value={settings.map.defaultLayer}
                          onChange={(e) => handleSettingChange('map', 'defaultLayer', e.target.value)}
                          label="Default Layer"
                        >
                          <MenuItem value="satellite">Satellite Imagery</MenuItem>
                          <MenuItem value="terrain">Terrain Only</MenuItem>
                          <MenuItem value="hybrid">Hybrid View</MenuItem>
                        </Select>
                      </FormControl>

                      <FormControl fullWidth>
                        <InputLabel>Render Quality</InputLabel>
                        <Select
                          value={settings.map.quality}
                          onChange={(e) => handleSettingChange('map', 'quality', e.target.value)}
                          label="Render Quality"
                        >
                          <MenuItem value="low">Low (Fast)</MenuItem>
                          <MenuItem value="medium">Medium</MenuItem>
                          <MenuItem value="high">High (Detailed)</MenuItem>
                        </Select>
                      </FormControl>

                      <Box>
                        <Typography gutterBottom>Cache Size (MB)</Typography>
                        <Slider
                          value={settings.map.cacheSize}
                          onChange={(_, value) => handleSettingChange('map', 'cacheSize', value)}
                          min={128}
                          max={2048}
                          step={128}
                          marks={[
                            { value: 128, label: '128MB' },
                            { value: 512, label: '512MB' },
                            { value: 1024, label: '1GB' },
                            { value: 2048, label: '2GB' },
                          ]}
                          valueLabelDisplay="auto"
                        />
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card variant="outlined">
                  <CardHeader title="Analysis Settings" />
                  <CardContent>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={4}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={settings.analysis.autoSave}
                              onChange={(e) => handleSettingChange('analysis', 'autoSave', e.target.checked)}
                            />
                          }
                          label="Auto-save analysis results"
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <FormControl fullWidth>
                          <InputLabel>Max Concurrent Analyses</InputLabel>
                          <Select
                            value={settings.analysis.maxConcurrent}
                            onChange={(e) => handleSettingChange('analysis', 'maxConcurrent', e.target.value)}
                            label="Max Concurrent Analyses"
                          >
                            <MenuItem value={1}>1</MenuItem>
                            <MenuItem value={2}>2</MenuItem>
                            <MenuItem value={3}>3</MenuItem>
                            <MenuItem value={5}>5</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Box>
                          <Typography gutterBottom>Confidence Threshold (%)</Typography>
                          <Slider
                            value={settings.analysis.confidenceThreshold}
                            onChange={(_, value) => handleSettingChange('analysis', 'confidenceThreshold', value)}
                            min={50}
                            max={95}
                            step={5}
                            marks={[
                              { value: 50, label: '50%' },
                              { value: 70, label: '70%' },
                              { value: 90, label: '90%' },
                            ]}
                            valueLabelDisplay="auto"
                          />
                        </Box>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {/* Appearance Settings */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardHeader title="Theme Settings" />
                  <CardContent>
                    <FormControl fullWidth>
                      <InputLabel>Color Theme</InputLabel>
                      <Select
                        value={settings.theme}
                        onChange={(e) => handleSettingChange('theme', '', e.target.value)}
                        label="Color Theme"
                      >
                        <MenuItem value="light">Light</MenuItem>
                        <MenuItem value="dark">Dark</MenuItem>
                        <MenuItem value="mars">Mars (Default)</MenuItem>
                      </Select>
                    </FormControl>
                    
                    <Box sx={{ mt: 3 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Theme Preview
                      </Typography>
                      <Box
                        sx={{
                          p: 2,
                          border: '1px solid',
                          borderColor: 'divider',
                          borderRadius: 1,
                          backgroundColor: settings.theme === 'mars' ? '#1a1a1a' : 
                                         settings.theme === 'dark' ? '#121212' : '#ffffff',
                          color: settings.theme === 'light' ? '#000000' : '#ffffff',
                        }}
                      >
                        <Typography variant="body2">
                          Sample text in {settings.theme} theme
                        </Typography>
                        <Button
                          variant="contained"
                          size="small"
                          sx={{ mt: 1 }}
                          style={{
                            backgroundColor: settings.theme === 'mars' ? '#ff6b35' : undefined,
                          }}
                        >
                          Sample Button
                        </Button>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardHeader title="Display Settings" />
                  <CardContent>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      <Alert severity="info">
                        <Typography variant="body2">
                          Additional display settings will be available in future updates.
                        </Typography>
                      </Alert>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            {/* Notification Settings */}
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card variant="outlined">
                  <CardHeader title="Notification Preferences" />
                  <CardContent>
                    <List>
                      <ListItem>
                        <ListItemIcon>
                          <NotificationsIcon />
                        </ListItemIcon>
                        <ListItemText
                          primary="Email Notifications"
                          secondary="Receive notifications via email"
                        />
                        <ListItemSecondaryAction>
                          <Switch
                            checked={settings.notifications.email}
                            onChange={(e) => handleSettingChange('notifications', 'email', e.target.checked)}
                          />
                        </ListItemSecondaryAction>
                      </ListItem>
                      <Divider variant="inset" component="li" />
                      
                      <ListItem>
                        <ListItemIcon>
                          <NotificationsIcon />
                        </ListItemIcon>
                        <ListItemText
                          primary="Push Notifications"
                          secondary="Browser push notifications"
                        />
                        <ListItemSecondaryAction>
                          <Switch
                            checked={settings.notifications.push}
                            onChange={(e) => handleSettingChange('notifications', 'push', e.target.checked)}
                          />
                        </ListItemSecondaryAction>
                      </ListItem>
                      <Divider variant="inset" component="li" />
                      
                      <ListItem>
                        <ListItemIcon>
                          <NotificationsIcon />
                        </ListItemIcon>
                        <ListItemText
                          primary="Desktop Notifications"
                          secondary="System desktop notifications"
                        />
                        <ListItemSecondaryAction>
                          <Switch
                            checked={settings.notifications.desktop}
                            onChange={(e) => handleSettingChange('notifications', 'desktop', e.target.checked)}
                          />
                        </ListItemSecondaryAction>
                      </ListItem>
                      <Divider variant="inset" component="li" />
                      
                      <ListItem>
                        <ListItemIcon>
                          <WarningIcon color="error" />
                        </ListItemIcon>
                        <ListItemText
                          primary="Critical Alerts"
                          secondary="Emergency and critical system alerts (recommended)"
                        />
                        <ListItemSecondaryAction>
                          <Switch
                            checked={settings.notifications.critical}
                            onChange={(e) => handleSettingChange('notifications', 'critical', e.target.checked)}
                          />
                        </ListItemSecondaryAction>
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            {/* Data & Storage Settings */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardHeader title="Data Management" />
                  <CardContent>
                    <List>
                      <ListItem>
                        <ListItemIcon>
                          <StorageIcon />
                        </ListItemIcon>
                        <ListItemText
                          primary="Cache Data"
                          secondary="Local cache: 1.2 GB"
                        />
                        <ListItemSecondaryAction>
                          <Button size="small" variant="outlined">
                            Clear
                          </Button>
                        </ListItemSecondaryAction>
                      </ListItem>
                      <Divider variant="inset" component="li" />
                      
                      <ListItem>
                        <ListItemIcon>
                          <CloudIcon />
                        </ListItemIcon>
                        <ListItemText
                          primary="Analysis Results"
                          secondary="Stored results: 847 MB"
                        />
                        <ListItemSecondaryAction>
                          <Button size="small" variant="outlined">
                            Archive
                          </Button>
                        </ListItemSecondaryAction>
                      </ListItem>
                      <Divider variant="inset" component="li" />
                      
                      <ListItem>
                        <ListItemIcon>
                          <NetworkIcon />
                        </ListItemIcon>
                        <ListItemText
                          primary="Mission Data"
                          secondary="Active missions: 234 MB"
                        />
                        <ListItemSecondaryAction>
                          <Button size="small" variant="outlined">
                            Export
                          </Button>
                        </ListItemSecondaryAction>
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardHeader title="Backup & Sync" />
                  <CardContent>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      <Alert severity="success" icon={<CheckCircleIcon />}>
                        <Typography variant="body2">
                          Last backup: 2 hours ago
                        </Typography>
                      </Alert>
                      
                      <Button variant="outlined" fullWidth startIcon={<DownloadIcon />}>
                        Create Backup
                      </Button>
                      
                      <Button variant="outlined" fullWidth startIcon={<UploadIcon />}>
                        Restore from Backup
                      </Button>
                      
                      <Divider />
                      
                      <FormControlLabel
                        control={<Switch defaultChecked />}
                        label="Auto-backup daily"
                      />
                      
                      <FormControlLabel
                        control={<Switch defaultChecked />}
                        label="Sync with cloud storage"
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={4}>
            {/* System Settings */}
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card variant="outlined">
                  <CardHeader title="System Information" />
                  <CardContent>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="h6" gutterBottom>
                          Application
                        </Typography>
                        <List dense>
                          <ListItem>
                            <ListItemText primary="Version" secondary="1.0.0-beta" />
                          </ListItem>
                          <ListItem>
                            <ListItemText primary="Build Date" secondary="2024-01-20" />
                          </ListItem>
                          <ListItem>
                            <ListItemText primary="Environment" secondary="Production" />
                          </ListItem>
                        </List>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Typography variant="h6" gutterBottom>
                          Performance
                        </Typography>
                        <List dense>
                          <ListItem>
                            <ListItemText primary="Memory Usage" secondary="456 MB / 2 GB" />
                          </ListItem>
                          <ListItem>
                            <ListItemText primary="CPU Usage" secondary="12%" />
                          </ListItem>
                          <ListItem>
                            <ListItemText primary="Network Status" secondary="Connected" />
                          </ListItem>
                        </List>
                      </Grid>
                    </Grid>
                    
                    <Divider sx={{ my: 2 }} />
                    
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button variant="outlined" startIcon={<DownloadIcon />}>
                        Download Logs
                      </Button>
                      <Button variant="outlined" startIcon={<RefreshIcon />}>
                        Check for Updates
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>
        </CardContent>
      </Card>

      {/* Reset Dialog */}
      <Dialog open={resetDialogOpen} onClose={() => setResetDialogOpen(false)}>
        <DialogTitle>Reset Settings</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to reset all settings to their default values? 
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleResetSettings} color="error" variant="contained">
            Reset All Settings
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};
