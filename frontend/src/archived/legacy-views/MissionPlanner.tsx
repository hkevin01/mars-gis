import {
    Add as AddIcon,
    BatteryFull as BatteryIcon,
    CheckCircle as CheckCircleIcon,
    Edit as EditIcon,
    FlightTakeoff as LaunchIcon,
    Pause as PauseIcon,
    RadioButtonUnchecked as PendingIcon,
    Route as RouteIcon,
    Schedule as ScheduleIcon,
    Science as ScienceIcon,
    PlayArrow as StartIcon,
    Stop as StopIcon,
    Thermostat as ThermostatIcon,
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
    Fab,
    FormControl,
    Grid,
    IconButton,
    InputLabel,
    LinearProgress,
    List,
    ListItem,
    ListItemIcon,
    ListItemSecondaryAction,
    ListItemText,
    MenuItem,
    Select,
    Step,
    StepContent,
    StepLabel,
    Stepper,
    TextField,
    Timeline,
    TimelineConnector,
    TimelineContent,
    TimelineDot,
    TimelineItem,
    TimelineOppositeContent,
    TimelineSeparator,
    Tooltip,
    Typography
} from '@mui/material';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import React, { useState } from 'react';

interface MissionTask {
  id: string;
  name: string;
  type: 'navigation' | 'sampling' | 'analysis' | 'communication' | 'maintenance';
  status: 'pending' | 'active' | 'completed' | 'failed';
  priority: 'low' | 'medium' | 'high' | 'critical';
  estimatedDuration: number; // in hours
  dependencies: string[];
  requirements: {
    battery: number;
    weather: string[];
    temperature: { min: number; max: number };
  };
  location?: [number, number];
  description: string;
}

interface Mission {
  id: string;
  name: string;
  description: string;
  status: 'draft' | 'planned' | 'active' | 'completed' | 'cancelled';
  startDate: Date;
  endDate: Date;
  assignedAsset: string;
  tasks: MissionTask[];
  progress: number;
  risks: string[];
}

const mockMissions: Mission[] = [
  {
    id: 'mission-1',
    name: 'Olympia Undae Survey',
    description: 'Comprehensive geological survey of the northern dune fields',
    status: 'active',
    startDate: new Date('2024-01-15'),
    endDate: new Date('2024-01-22'),
    assignedAsset: 'Rover Alpha',
    tasks: [
      {
        id: 'task-1',
        name: 'Navigate to Survey Area',
        type: 'navigation',
        status: 'completed',
        priority: 'high',
        estimatedDuration: 4,
        dependencies: [],
        requirements: {
          battery: 80,
          weather: ['clear', 'light_dust'],
          temperature: { min: -90, max: 10 },
        },
        location: [-14.5684, 175.4729],
        description: 'Move to initial survey coordinates',
      },
      {
        id: 'task-2',
        name: 'Collect Soil Samples',
        type: 'sampling',
        status: 'active',
        priority: 'critical',
        estimatedDuration: 6,
        dependencies: ['task-1'],
        requirements: {
          battery: 70,
          weather: ['clear'],
          temperature: { min: -80, max: 20 },
        },
        location: [-14.5720, 175.4800],
        description: 'Extract soil samples from 3 different locations',
      },
      {
        id: 'task-3',
        name: 'Spectroscopic Analysis',
        type: 'analysis',
        status: 'pending',
        priority: 'medium',
        estimatedDuration: 3,
        dependencies: ['task-2'],
        requirements: {
          battery: 60,
          weather: ['clear', 'light_dust'],
          temperature: { min: -70, max: 30 },
        },
        description: 'Perform on-site mineral composition analysis',
      },
    ],
    progress: 65,
    risks: ['dust_storm_approaching', 'battery_degradation'],
  },
  {
    id: 'mission-2',
    name: 'Crater Rim Exploration',
    description: 'Investigation of impact crater geological features',
    status: 'planned',
    startDate: new Date('2024-01-25'),
    endDate: new Date('2024-02-01'),
    assignedAsset: 'Rover Beta',
    tasks: [],
    progress: 0,
    risks: ['steep_terrain', 'communication_blackout'],
  },
];

const missionSteps = [
  'Mission Planning',
  'Risk Assessment',
  'Resource Allocation',
  'Timeline Creation',
  'Approval & Launch',
];

const taskTypeIcons = {
  navigation: <RouteIcon />,
  sampling: <ScienceIcon />,
  analysis: <ScienceIcon />,
  communication: <LaunchIcon />,
  maintenance: <BatteryIcon />,
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed': return 'success';
    case 'active': return 'primary';
    case 'failed': return 'error';
    case 'pending': return 'default';
    default: return 'default';
  }
};

const getPriorityColor = (priority: string) => {
  switch (priority) {
    case 'critical': return 'error';
    case 'high': return 'warning';
    case 'medium': return 'info';
    case 'low': return 'default';
    default: return 'default';
  }
};

export const MissionPlanner: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [missions, setMissions] = useState<Mission[]>(mockMissions);
  const [selectedMission, setSelectedMission] = useState<string | null>(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newMission, setNewMission] = useState<Partial<Mission>>({
    name: '',
    description: '',
    startDate: new Date(),
    endDate: new Date(),
    assignedAsset: '',
    tasks: [],
  });

  const selectedMissionData = missions.find(m => m.id === selectedMission);

  const handleStepClick = (step: number) => {
    setActiveStep(step);
  };

  const handleCreateMission = () => {
    if (newMission.name && newMission.description) {
      const mission: Mission = {
        id: `mission-${Date.now()}`,
        name: newMission.name,
        description: newMission.description,
        status: 'draft',
        startDate: newMission.startDate || new Date(),
        endDate: newMission.endDate || new Date(),
        assignedAsset: newMission.assignedAsset || '',
        tasks: [],
        progress: 0,
        risks: [],
      };
      
      setMissions(prev => [...prev, mission]);
      setCreateDialogOpen(false);
      setNewMission({
        name: '',
        description: '',
        startDate: new Date(),
        endDate: new Date(),
        assignedAsset: '',
        tasks: [],
      });
    }
  };

  const getTaskStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon color="success" />;
      case 'active':
        return <PlayArrow color="primary" />;
      case 'failed':
        return <WarningIcon color="error" />;
      default:
        return <PendingIcon color="disabled" />;
    }
  };

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Box sx={{ p: 3, height: '100%' }}>
        <Grid container spacing={3} sx={{ height: '100%' }}>
          {/* Mission List */}
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%' }}>
              <CardHeader
                title="Active Missions"
                action={
                  <Tooltip title="Create New Mission">
                    <IconButton onClick={() => setCreateDialogOpen(true)}>
                      <AddIcon />
                    </IconButton>
                  </Tooltip>
                }
              />
              <CardContent sx={{ height: 'calc(100% - 64px)', overflow: 'auto' }}>
                <List>
                  {missions.map((mission) => (
                    <ListItem
                      key={mission.id}
                      button
                      selected={selectedMission === mission.id}
                      onClick={() => setSelectedMission(mission.id)}
                      sx={{ mb: 1, borderRadius: 1 }}
                    >
                      <ListItemIcon>
                        {taskTypeIcons[mission.tasks[0]?.type || 'navigation']}
                      </ListItemIcon>
                      <ListItemText
                        primary={mission.name}
                        secondary={
                          <Box>
                            <Typography variant="caption" display="block">
                              {mission.assignedAsset || 'Unassigned'}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                              <Chip
                                label={mission.status.toUpperCase()}
                                size="small"
                                color={getStatusColor(mission.status) as any}
                              />
                              {mission.status === 'active' && (
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                  <Typography variant="caption">
                                    {mission.progress}%
                                  </Typography>
                                  <LinearProgress
                                    variant="determinate"
                                    value={mission.progress}
                                    sx={{ width: 40, height: 4 }}
                                  />
                                </Box>
                              )}
                            </Box>
                          </Box>
                        }
                      />
                      <ListItemSecondaryAction>
                        <IconButton size="small">
                          <EditIcon />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Mission Details */}
          <Grid item xs={12} md={8}>
            {selectedMissionData ? (
              <Grid container spacing={2} sx={{ height: '100%' }}>
                {/* Mission Overview */}
                <Grid item xs={12}>
                  <Card>
                    <CardHeader
                      title={selectedMissionData.name}
                      subheader={selectedMissionData.description}
                      action={
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Chip
                            label={selectedMissionData.status.toUpperCase()}
                            color={getStatusColor(selectedMissionData.status) as any}
                          />
                          {selectedMissionData.status === 'planned' && (
                            <Button startIcon={<StartIcon />} variant="contained" size="small">
                              Launch
                            </Button>
                          )}
                          {selectedMissionData.status === 'active' && (
                            <>
                              <Button startIcon={<PauseIcon />} variant="outlined" size="small">
                                Pause
                              </Button>
                              <Button startIcon={<StopIcon />} variant="outlined" color="error" size="small">
                                Abort
                              </Button>
                            </>
                          )}
                        </Box>
                      }
                    />
                    <CardContent>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="caption" color="text.secondary">
                            Assigned Asset
                          </Typography>
                          <Typography variant="body2" fontWeight={600}>
                            {selectedMissionData.assignedAsset || 'Unassigned'}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="caption" color="text.secondary">
                            Progress
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={selectedMissionData.progress}
                              sx={{ flexGrow: 1, height: 6 }}
                            />
                            <Typography variant="body2">
                              {selectedMissionData.progress}%
                            </Typography>
                          </Box>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="caption" color="text.secondary">
                            Start Date
                          </Typography>
                          <Typography variant="body2">
                            {selectedMissionData.startDate.toLocaleDateString()}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="caption" color="text.secondary">
                            End Date
                          </Typography>
                          <Typography variant="body2">
                            {selectedMissionData.endDate.toLocaleDateString()}
                          </Typography>
                        </Grid>
                      </Grid>

                      {selectedMissionData.risks.length > 0 && (
                        <Alert severity="warning" sx={{ mt: 2 }}>
                          <Typography variant="body2" fontWeight={600}>
                            Risk Factors:
                          </Typography>
                          <Typography variant="body2">
                            {selectedMissionData.risks.join(', ')}
                          </Typography>
                        </Alert>
                      )}
                    </CardContent>
                  </Card>
                </Grid>

                {/* Mission Timeline */}
                <Grid item xs={12}>
                  <Card sx={{ height: 400 }}>
                    <CardHeader title="Mission Timeline" />
                    <CardContent sx={{ height: 'calc(100% - 64px)', overflow: 'auto' }}>
                      {selectedMissionData.tasks.length > 0 ? (
                        <Timeline>
                          {selectedMissionData.tasks.map((task, index) => (
                            <TimelineItem key={task.id}>
                              <TimelineOppositeContent sx={{ m: 'auto 0' }} align="right" variant="body2" color="text.secondary">
                                {task.estimatedDuration}h
                              </TimelineOppositeContent>
                              <TimelineSeparator>
                                <TimelineDot color={getStatusColor(task.status) as any}>
                                  {getTaskStatusIcon(task.status)}
                                </TimelineDot>
                                {index < selectedMissionData.tasks.length - 1 && <TimelineConnector />}
                              </TimelineSeparator>
                              <TimelineContent sx={{ py: '12px', px: 2 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                  <Typography variant="h6" component="span">
                                    {task.name}
                                  </Typography>
                                  <Chip
                                    label={task.priority.toUpperCase()}
                                    size="small"
                                    color={getPriorityColor(task.priority) as any}
                                  />
                                </Box>
                                <Typography variant="body2" color="text.secondary">
                                  {task.description}
                                </Typography>
                                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                                  <Chip
                                    icon={<BatteryIcon />}
                                    label={`${task.requirements.battery}%`}
                                    size="small"
                                    variant="outlined"
                                  />
                                  <Chip
                                    icon={<ThermostatIcon />}
                                    label={`${task.requirements.temperature.min}°C to ${task.requirements.temperature.max}°C`}
                                    size="small"
                                    variant="outlined"
                                  />
                                </Box>
                              </TimelineContent>
                            </TimelineItem>
                          ))}
                        </Timeline>
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
                          <ScheduleIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
                          <Typography variant="h6" color="text.secondary" gutterBottom>
                            No Tasks Defined
                          </Typography>
                          <Typography variant="body2" color="text.disabled">
                            Add tasks to create a mission timeline
                          </Typography>
                          <Button
                            variant="contained"
                            startIcon={<AddIcon />}
                            sx={{ mt: 2 }}
                          >
                            Add Task
                          </Button>
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            ) : (
              <Card sx={{ height: '100%' }}>
                <CardContent
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    textAlign: 'center',
                  }}
                >
                  <LaunchIcon sx={{ fontSize: 80, color: 'text.disabled', mb: 2 }} />
                  <Typography variant="h5" color="text.secondary" gutterBottom>
                    Mission Planning Center
                  </Typography>
                  <Typography variant="body1" color="text.disabled" paragraph>
                    Select a mission from the list to view details and manage tasks, or create a new mission to get started.
                  </Typography>
                  <Button
                    variant="contained"
                    size="large"
                    startIcon={<AddIcon />}
                    onClick={() => setCreateDialogOpen(true)}
                  >
                    Create New Mission
                  </Button>
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>

        {/* Create Mission Dialog */}
        <Dialog
          open={createDialogOpen}
          onClose={() => setCreateDialogOpen(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>Create New Mission</DialogTitle>
          <DialogContent>
            <Stepper activeStep={activeStep} orientation="vertical">
              {missionSteps.map((label, index) => (
                <Step key={label}>
                  <StepLabel
                    onClick={() => handleStepClick(index)}
                    sx={{ cursor: 'pointer' }}
                  >
                    {label}
                  </StepLabel>
                  <StepContent>
                    {index === 0 && (
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
                        <TextField
                          label="Mission Name"
                          value={newMission.name || ''}
                          onChange={(e) => setNewMission(prev => ({ ...prev, name: e.target.value }))}
                          fullWidth
                        />
                        <TextField
                          label="Description"
                          value={newMission.description || ''}
                          onChange={(e) => setNewMission(prev => ({ ...prev, description: e.target.value }))}
                          multiline
                          rows={3}
                          fullWidth
                        />
                        <FormControl fullWidth>
                          <InputLabel>Assigned Asset</InputLabel>
                          <Select
                            value={newMission.assignedAsset || ''}
                            onChange={(e) => setNewMission(prev => ({ ...prev, assignedAsset: e.target.value }))}
                          >
                            <MenuItem value="Rover Alpha">Rover Alpha</MenuItem>
                            <MenuItem value="Rover Beta">Rover Beta</MenuItem>
                            <MenuItem value="Lander Gamma">Lander Gamma</MenuItem>
                          </Select>
                        </FormControl>
                        <DateTimePicker
                          label="Start Date"
                          value={newMission.startDate}
                          onChange={(date) => setNewMission(prev => ({ ...prev, startDate: date || new Date() }))}
                          renderInput={(params) => <TextField {...params} fullWidth />}
                        />
                        <DateTimePicker
                          label="End Date"
                          value={newMission.endDate}
                          onChange={(date) => setNewMission(prev => ({ ...prev, endDate: date || new Date() }))}
                          renderInput={(params) => <TextField {...params} fullWidth />}
                        />
                      </Box>
                    )}
                    <Box sx={{ mt: 2 }}>
                      <Button
                        variant="contained"
                        onClick={() => setActiveStep(prev => prev + 1)}
                        sx={{ mr: 1 }}
                        disabled={index === missionSteps.length - 1}
                      >
                        {index === missionSteps.length - 1 ? 'Finish' : 'Continue'}
                      </Button>
                      <Button
                        disabled={index === 0}
                        onClick={() => setActiveStep(prev => prev - 1)}
                      >
                        Back
                      </Button>
                    </Box>
                  </StepContent>
                </Step>
              ))}
            </Stepper>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
            <Button
              onClick={handleCreateMission}
              variant="contained"
              disabled={!newMission.name || !newMission.description}
            >
              Create Mission
            </Button>
          </DialogActions>
        </Dialog>

        {/* Floating Action Button */}
        <Fab
          color="primary"
          aria-label="add mission"
          sx={{
            position: 'fixed',
            bottom: 24,
            right: 24,
          }}
          onClick={() => setCreateDialogOpen(true)}
        >
          <AddIcon />
        </Fab>
      </Box>
    </LocalizationProvider>
  );
};
