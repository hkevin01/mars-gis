import {
    Close as CloseIcon,
    Delete as DeleteIcon,
    Error as ErrorIcon,
    Info as InfoIcon,
    MarkAsUnread as MarkUnreadIcon,
    Notifications as NotificationsIcon,
    CheckCircle as SuccessIcon,
    Warning as WarningIcon,
} from '@mui/icons-material';
import {
    Alert,
    Badge,
    Box,
    Button,
    Chip,
    Divider,
    Drawer,
    IconButton,
    List,
    ListItem,
    ListItemIcon,
    ListItemSecondaryAction,
    ListItemText,
    Typography,
} from '@mui/material';
import { formatDistanceToNow } from 'date-fns';
import React, { useState } from 'react';

interface Notification {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  source: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

interface NotificationCenterProps {
  open: boolean;
  onClose: () => void;
}

const mockNotifications: Notification[] = [
  {
    id: '1',
    type: 'warning',
    title: 'Dust Storm Alert',
    message: 'Increased dust activity detected in Olympia Undae region. Mission planning adjustments recommended.',
    timestamp: new Date(Date.now() - 15 * 60 * 1000), // 15 minutes ago
    read: false,
    source: 'Weather Monitor',
    priority: 'high',
  },
  {
    id: '2',
    type: 'success',
    title: 'Data Processing Complete',
    message: 'Terrain analysis for Grid Sector 7-Alpha has been successfully processed.',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
    read: true,
    source: 'Analysis Engine',
    priority: 'medium',
  },
  {
    id: '3',
    type: 'info',
    title: 'New Orbital Data Available',
    message: 'Fresh satellite imagery from MRO has been integrated into the mapping system.',
    timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000), // 4 hours ago
    read: false,
    source: 'Data Ingestion',
    priority: 'low',
  },
  {
    id: '4',
    type: 'error',
    title: 'Communication Link Error',
    message: 'Temporary loss of connection with Rover Unit Charlie. Attempting reconnection.',
    timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000), // 6 hours ago
    read: false,
    source: 'Communications',
    priority: 'critical',
  },
  {
    id: '5',
    type: 'info',
    title: 'Mission Planning Update',
    message: 'Route optimization for Sol 1235 has been updated based on latest terrain data.',
    timestamp: new Date(Date.now() - 12 * 60 * 60 * 1000), // 12 hours ago
    read: true,
    source: 'Mission Control',
    priority: 'medium',
  },
];

const getNotificationIcon = (type: Notification['type']) => {
  switch (type) {
    case 'error':
      return <ErrorIcon color="error" />;
    case 'warning':
      return <WarningIcon color="warning" />;
    case 'success':
      return <SuccessIcon color="success" />;
    case 'info':
    default:
      return <InfoIcon color="info" />;
  }
};

const getPriorityColor = (priority: Notification['priority']) => {
  switch (priority) {
    case 'critical':
      return 'error';
    case 'high':
      return 'warning';
    case 'medium':
      return 'info';
    case 'low':
    default:
      return 'default';
  }
};

export const NotificationCenter: React.FC<NotificationCenterProps> = ({
  open,
  onClose,
}) => {
  const [notifications, setNotifications] = useState<Notification[]>(mockNotifications);

  const unreadCount = notifications.filter(n => !n.read).length;
  const criticalCount = notifications.filter(n => n.priority === 'critical' && !n.read).length;

  const handleMarkAsRead = (id: string) => {
    setNotifications(prev =>
      prev.map(notification =>
        notification.id === id
          ? { ...notification, read: true }
          : notification
      )
    );
  };

  const handleMarkAsUnread = (id: string) => {
    setNotifications(prev =>
      prev.map(notification =>
        notification.id === id
          ? { ...notification, read: false }
          : notification
      )
    );
  };

  const handleDelete = (id: string) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
  };

  const handleMarkAllAsRead = () => {
    setNotifications(prev =>
      prev.map(notification => ({ ...notification, read: true }))
    );
  };

  const handleClearAll = () => {
    setNotifications([]);
  };

  // Sort notifications by priority and timestamp
  const sortedNotifications = [...notifications].sort((a, b) => {
    const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
    if (a.priority !== b.priority) {
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    }
    return b.timestamp.getTime() - a.timestamp.getTime();
  });

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      sx={{
        '& .MuiDrawer-paper': {
          width: { xs: '100%', sm: 420 },
          maxWidth: '100vw',
        },
      }}
    >
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box
          sx={{
            p: 2,
            borderBottom: '1px solid',
            borderColor: 'divider',
            backgroundColor: 'background.paper',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Badge badgeContent={unreadCount} color="primary" sx={{ mr: 2 }}>
              <NotificationsIcon />
            </Badge>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              Notifications
            </Typography>
            <IconButton onClick={onClose} size="small">
              <CloseIcon />
            </IconButton>
          </Box>

          {criticalCount > 0 && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {criticalCount} critical alert{criticalCount > 1 ? 's' : ''} require immediate attention
            </Alert>
          )}

          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              size="small"
              variant="outlined"
              onClick={handleMarkAllAsRead}
              disabled={unreadCount === 0}
            >
              Mark All Read
            </Button>
            <Button
              size="small"
              variant="outlined"
              color="error"
              onClick={handleClearAll}
              disabled={notifications.length === 0}
            >
              Clear All
            </Button>
          </Box>
        </Box>

        {/* Notifications List */}
        <Box sx={{ flex: 1, overflow: 'auto' }}>
          {notifications.length === 0 ? (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                p: 4,
                textAlign: 'center',
              }}
            >
              <NotificationsIcon
                sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }}
              />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No Notifications
              </Typography>
              <Typography variant="body2" color="text.disabled">
                All caught up! New alerts will appear here.
              </Typography>
            </Box>
          ) : (
            <List sx={{ p: 0 }}>
              {sortedNotifications.map((notification, index) => (
                <React.Fragment key={notification.id}>
                  <ListItem
                    sx={{
                      backgroundColor: notification.read
                        ? 'background.paper'
                        : 'action.hover',
                      '&:hover': {
                        backgroundColor: 'action.selected',
                      },
                      py: 2,
                    }}
                  >
                    <ListItemIcon sx={{ mt: 0.5 }}>
                      {getNotificationIcon(notification.type)}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                          <Typography
                            variant="subtitle2"
                            sx={{
                              fontWeight: notification.read ? 400 : 600,
                              flexGrow: 1,
                            }}
                          >
                            {notification.title}
                          </Typography>
                          <Chip
                            label={notification.priority.toUpperCase()}
                            size="small"
                            color={getPriorityColor(notification.priority) as any}
                            sx={{ minWidth: 60, height: 20, fontSize: '0.7rem' }}
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography
                            variant="body2"
                            color="text.secondary"
                            sx={{ mb: 0.5 }}
                          >
                            {notification.message}
                          </Typography>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="caption" color="text.disabled">
                              {notification.source} • {formatDistanceToNow(notification.timestamp)} ago
                            </Typography>
                          </Box>
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                        <IconButton
                          size="small"
                          onClick={() =>
                            notification.read
                              ? handleMarkAsUnread(notification.id)
                              : handleMarkAsRead(notification.id)
                          }
                        >
                          <MarkUnreadIcon fontSize="small" />
                        </IconButton>
                        <IconButton
                          size="small"
                          onClick={() => handleDelete(notification.id)}
                          color="error"
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Box>
                    </ListItemSecondaryAction>
                  </ListItem>
                  {index < sortedNotifications.length - 1 && (
                    <Divider variant="inset" component="li" />
                  )}
                </React.Fragment>
              ))}
            </List>
          )}
        </Box>
      </Box>
    </Drawer>
  );
};
