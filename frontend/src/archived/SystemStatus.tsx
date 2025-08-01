import {
    CheckCircle as CheckCircleIcon,
    Cloud as CloudIcon,
    Error as ErrorIcon,
    ExpandLess as ExpandLessIcon,
    ExpandMore as ExpandMoreIcon,
    Info as InfoIcon,
    Memory as MemoryIcon,
    NetworkCheck as NetworkIcon,
    Storage as StorageIcon,
    Warning as WarningIcon,
} from '@mui/icons-material';
import {
    Box,
    Card,
    CardContent,
    Chip,
    Collapse,
    Grid,
    IconButton,
    LinearProgress,
    Typography
} from '@mui/material';
import React, { useEffect, useState } from 'react';

interface SystemMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  status: 'healthy' | 'warning' | 'critical';
  threshold: {
    warning: number;
    critical: number;
  };
}

interface SystemService {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'degraded';
  uptime: string;
  lastCheck: Date;
}

const systemMetrics: SystemMetric[] = [
  {
    id: 'cpu',
    name: 'CPU Usage',
    value: 34,
    unit: '%',
    status: 'healthy',
    threshold: { warning: 70, critical: 90 },
  },
  {
    id: 'memory',
    name: 'Memory',
    value: 67,
    unit: '%',
    status: 'warning',
    threshold: { warning: 80, critical: 95 },
  },
  {
    id: 'storage',
    name: 'Storage',
    value: 23,
    unit: '%',
    status: 'healthy',
    threshold: { warning: 80, critical: 95 },
  },
  {
    id: 'network',
    name: 'Network',
    value: 0.2,
    unit: 'ms',
    status: 'healthy',
    threshold: { warning: 100, critical: 500 },
  },
];

const systemServices: SystemService[] = [
  {
    id: 'api',
    name: 'API Gateway',
    status: 'online',
    uptime: '99.9%',
    lastCheck: new Date(),
  },
  {
    id: 'database',
    name: 'Database',
    status: 'online',
    uptime: '99.7%',
    lastCheck: new Date(),
  },
  {
    id: 'ml-engine',
    name: 'ML Engine',
    status: 'online',
    uptime: '98.2%',
    lastCheck: new Date(),
  },
  {
    id: 'weather',
    name: 'Weather Service',
    status: 'degraded',
    uptime: '95.1%',
    lastCheck: new Date(Date.now() - 5 * 60 * 1000),
  },
  {
    id: 'communication',
    name: 'Communication',
    status: 'offline',
    uptime: '0.0%',
    lastCheck: new Date(Date.now() - 30 * 60 * 1000),
  },
];

const getStatusColor = (status: string) => {
  switch (status) {
    case 'healthy':
    case 'online':
      return 'success';
    case 'warning':
    case 'degraded':
      return 'warning';
    case 'critical':
    case 'offline':
      return 'error';
    default:
      return 'default';
  }
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'healthy':
    case 'online':
      return <CheckCircleIcon fontSize="small" color="success" />;
    case 'warning':
    case 'degraded':
      return <WarningIcon fontSize="small" color="warning" />;
    case 'critical':
    case 'offline':
      return <ErrorIcon fontSize="small" color="error" />;
    default:
      return <InfoIcon fontSize="small" />;
  }
};

const getMetricIcon = (id: string) => {
  switch (id) {
    case 'memory':
      return <MemoryIcon fontSize="small" />;
    case 'storage':
      return <StorageIcon fontSize="small" />;
    case 'network':
      return <NetworkIcon fontSize="small" />;
    default:
      return <CloudIcon fontSize="small" />;
  }
};

export const SystemStatus: React.FC = () => {
  const [expanded, setExpanded] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const overallStatus = systemServices.every(s => s.status === 'online')
    ? 'healthy'
    : systemServices.some(s => s.status === 'offline')
    ? 'critical'
    : 'warning';

  const criticalServices = systemServices.filter(s => s.status === 'offline').length;
  const degradedServices = systemServices.filter(s => s.status === 'degraded').length;

  return (
    <Card
      sx={{
        background: 'linear-gradient(135deg, rgba(255,107,53,0.1) 0%, rgba(247,147,30,0.1) 100%)',
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        {/* Header */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
          }}
          onClick={() => setExpanded(!expanded)}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {getStatusIcon(overallStatus)}
            <Typography variant="subtitle2" fontWeight={600}>
              System Status
            </Typography>
            <Chip
              label={overallStatus.toUpperCase()}
              size="small"
              color={getStatusColor(overallStatus) as any}
              sx={{ height: 20, fontSize: '0.7rem' }}
            />
          </Box>
          <IconButton size="small">
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        </Box>

        {/* Quick Overview */}
        {!expanded && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary">
              {systemServices.length - criticalServices - degradedServices} services online
              {criticalServices > 0 && `, ${criticalServices} offline`}
              {degradedServices > 0 && `, ${degradedServices} degraded`}
            </Typography>
          </Box>
        )}

        {/* Expanded Details */}
        <Collapse in={expanded}>
          <Box sx={{ mt: 2 }}>
            {/* System Metrics */}
            <Typography variant="caption" sx={{ fontWeight: 600, mb: 1, display: 'block' }}>
              System Metrics
            </Typography>
            <Grid container spacing={1} sx={{ mb: 2 }}>
              {systemMetrics.map((metric) => (
                <Grid item xs={6} key={metric.id}>
                  <Box
                    sx={{
                      p: 1,
                      borderRadius: 1,
                      backgroundColor: 'background.paper',
                      border: '1px solid',
                      borderColor: 'divider',
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                      {getMetricIcon(metric.id)}
                      <Typography variant="caption" fontWeight={500}>
                        {metric.name}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={metric.value}
                        sx={{
                          flexGrow: 1,
                          height: 4,
                          borderRadius: 2,
                          backgroundColor: 'action.selected',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor:
                              metric.status === 'healthy'
                                ? 'success.main'
                                : metric.status === 'warning'
                                ? 'warning.main'
                                : 'error.main',
                          },
                        }}
                      />
                      <Typography variant="caption" fontWeight={600} sx={{ minWidth: 'fit-content' }}>
                        {metric.value}{metric.unit}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
              ))}
            </Grid>

            {/* Services Status */}
            <Typography variant="caption" sx={{ fontWeight: 600, mb: 1, display: 'block' }}>
              Services
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              {systemServices.map((service) => (
                <Box
                  key={service.id}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    p: 0.5,
                    borderRadius: 0.5,
                    backgroundColor: 'background.paper',
                    border: '1px solid',
                    borderColor: 'divider',
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    {getStatusIcon(service.status)}
                    <Typography variant="caption" fontWeight={500}>
                      {service.name}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      {service.uptime}
                    </Typography>
                    <Chip
                      label={service.status.toUpperCase()}
                      size="small"
                      color={getStatusColor(service.status) as any}
                      sx={{ height: 16, fontSize: '0.6rem', minWidth: 50 }}
                    />
                  </Box>
                </Box>
              ))}
            </Box>

            {/* Last Updated */}
            <Typography
              variant="caption"
              color="text.disabled"
              sx={{ display: 'block', textAlign: 'center', mt: 1 }}
            >
              Last updated: {currentTime.toLocaleTimeString()}
            </Typography>
          </Box>
        </Collapse>
      </CardContent>
    </Card>
  );
};
