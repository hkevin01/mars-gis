import {
    AccountCircle,
    Analytics as AnalyticsIcon,
    Dashboard as DashboardIcon,
    ExitToApp,
    Help as HelpIcon,
    Map as MapIcon,
    Public as Mars3DIcon,
    Menu as MenuIcon,
    FlightTakeoff as MissionIcon,
    Notifications,
    Route as PathIcon,
    Settings as SettingsIcon,
    Terrain as TerrainIcon,
} from '@mui/icons-material';
import {
    AppBar,
    Avatar,
    Box,
    Chip,
    Divider,
    Drawer,
    IconButton,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Menu,
    MenuItem,
    Toolbar,
    Typography,
    useMediaQuery,
    useTheme,
} from '@mui/material';
import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import { NotificationCenter } from './NotificationCenter';
import { SystemStatus } from './SystemStatus';

const DRAWER_WIDTH = 280;

interface LayoutProps {
  children: React.ReactNode;
}

interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon: React.ReactNode;
  category: string;
  badge?: number;
}

const navigationItems: NavigationItem[] = [
  {
    id: 'dashboard',
    label: 'Mission Dashboard',
    path: '/dashboard',
    icon: <DashboardIcon />,
    category: 'main',
  },
  {
    id: 'mars-3d',
    label: '3D Mars Viewer',
    path: '/mars-3d',
    icon: <Mars3DIcon />,
    category: 'visualization',
  },
  {
    id: 'interactive-map',
    label: 'Interactive Map',
    path: '/interactive-map',
    icon: <MapIcon />,
    category: 'visualization',
  },
  {
    id: 'mission-planner',
    label: 'Mission Planner',
    path: '/mission-planner',
    icon: <MissionIcon />,
    category: 'planning',
  },
  {
    id: 'path-planning',
    label: 'Path Planning',
    path: '/path-planning',
    icon: <PathIcon />,
    category: 'planning',
  },
  {
    id: 'data-analysis',
    label: 'Data Analysis',
    path: '/data-analysis',
    icon: <AnalyticsIcon />,
    category: 'analysis',
  },
  {
    id: 'terrain-analysis',
    label: 'Terrain Analysis',
    path: '/terrain-analysis',
    icon: <TerrainIcon />,
    category: 'analysis',
  },
  {
    id: 'settings',
    label: 'Settings',
    path: '/settings',
    icon: <SettingsIcon />,
    category: 'system',
  },
  {
    id: 'documentation',
    label: 'Documentation',
    path: '/documentation',
    icon: <HelpIcon />,
    category: 'system',
  },
];

const categoryLabels = {
  main: 'Mission Control',
  visualization: 'Visualization',
  planning: 'Mission Planning',
  analysis: 'Data Analysis',
  system: 'System',
};

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const [mobileOpen, setMobileOpen] = useState(false);
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);
  const [notificationOpen, setNotificationOpen] = useState(false);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };

  const handleNavigate = (path: string) => {
    navigate(path);
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  const renderNavigationItems = () => {
    const categories = navigationItems.reduce((acc, item) => {
      if (!acc[item.category]) {
        acc[item.category] = [];
      }
      acc[item.category].push(item);
      return acc;
    }, {} as Record<string, NavigationItem[]>);

    return Object.entries(categories).map(([category, items]) => (
      <Box key={category} sx={{ mb: 1 }}>
        <Typography
          variant="caption"
          sx={{
            px: 2,
            py: 1,
            color: 'text.secondary',
            fontWeight: 600,
            fontSize: '0.75rem',
            textTransform: 'uppercase',
            letterSpacing: '0.08333em',
          }}
        >
          {categoryLabels[category as keyof typeof categoryLabels]}
        </Typography>
        <List dense>
          {items.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <ListItem key={item.id} disablePadding>
                <ListItemButton
                  onClick={() => handleNavigate(item.path)}
                  selected={isActive}
                  sx={{
                    mx: 1,
                    borderRadius: 1,
                    '&.Mui-selected': {
                      backgroundColor: 'primary.main',
                      color: 'primary.contrastText',
                      '&:hover': {
                        backgroundColor: 'primary.dark',
                      },
                      '& .MuiListItemIcon-root': {
                        color: 'primary.contrastText',
                      },
                    },
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: 40,
                      color: isActive ? 'inherit' : 'text.secondary',
                    }}
                  >
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.label}
                    primaryTypographyProps={{
                      fontSize: '0.875rem',
                      fontWeight: isActive ? 600 : 400,
                    }}
                  />
                  {item.badge && (
                    <Chip
                      label={item.badge}
                      size="small"
                      color="secondary"
                      sx={{ ml: 1, height: 20, fontSize: '0.75rem' }}
                    />
                  )}
                </ListItemButton>
              </ListItem>
            );
          })}
        </List>
      </Box>
    ));
  };

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Logo and Title */}
      <Box
        sx={{
          p: 2,
          borderBottom: '1px solid',
          borderColor: 'divider',
          background: 'linear-gradient(135deg, #ff6b35 0%, #f7931e 100%)',
        }}
      >
        <Typography
          variant="h6"
          sx={{
            fontWeight: 700,
            color: 'white',
            textAlign: 'center',
            letterSpacing: '0.1em',
          }}
        >
          MARS-GIS
        </Typography>
        <Typography
          variant="caption"
          sx={{
            color: 'rgba(255, 255, 255, 0.8)',
            textAlign: 'center',
            display: 'block',
            fontSize: '0.7rem',
            letterSpacing: '0.05em',
          }}
        >
          Mission Control Platform
        </Typography>
      </Box>

      {/* System Status */}
      <Box sx={{ px: 2, py: 1 }}>
        <SystemStatus />
      </Box>

      {/* Navigation */}
      <Box sx={{ flex: 1, overflow: 'auto', py: 1 }}>
        {renderNavigationItems()}
      </Box>

      {/* Footer */}
      <Box
        sx={{
          p: 2,
          borderTop: '1px solid',
          borderColor: 'divider',
          backgroundColor: 'rgba(255, 107, 53, 0.05)',
        }}
      >
        <Typography
          variant="caption"
          sx={{
            color: 'text.secondary',
            fontSize: '0.7rem',
            textAlign: 'center',
            display: 'block',
          }}
        >
          Mars Local Time (MLT)
        </Typography>
        <Typography
          variant="body2"
          sx={{
            color: 'primary.main',
            fontSize: '0.875rem',
            fontWeight: 600,
            textAlign: 'center',
          }}
        >
          Sol 1234 | 14:32
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${DRAWER_WIDTH}px)` },
          ml: { md: `${DRAWER_WIDTH}px` },
          zIndex: theme.zIndex.drawer + 1,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            {navigationItems.find(item => item.path === location.pathname)?.label || 'MARS-GIS'}
          </Typography>

          {/* Notifications */}
          <IconButton
            color="inherit"
            onClick={() => setNotificationOpen(true)}
            sx={{ mr: 1 }}
          >
            <Notifications />
          </IconButton>

          {/* User Menu */}
          <IconButton
            color="inherit"
            onClick={handleUserMenuOpen}
            sx={{ ml: 1 }}
          >
            <Avatar
              sx={{
                width: 32,
                height: 32,
                bgcolor: 'secondary.main',
                fontSize: '0.875rem',
              }}
            >
              MG
            </Avatar>
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Navigation Drawer */}
      <Box
        component="nav"
        sx={{ width: { md: DRAWER_WIDTH }, flexShrink: { md: 0 } }}
      >
        <Drawer
          variant={isMobile ? 'temporary' : 'permanent'}
          open={isMobile ? mobileOpen : true}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better mobile performance
          }}
          sx={{
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: DRAWER_WIDTH,
              borderRight: '1px solid',
              borderColor: 'divider',
            },
          }}
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 0,
          width: { md: `calc(100% - ${DRAWER_WIDTH}px)` },
          height: '100vh',
          overflow: 'hidden',
        }}
      >
        <Toolbar /> {/* Spacer for AppBar */}
        <Box sx={{ height: 'calc(100vh - 64px)', overflow: 'auto' }}>
          {children}
        </Box>
      </Box>

      {/* User Menu */}
      <Menu
        anchorEl={userMenuAnchor}
        open={Boolean(userMenuAnchor)}
        onClose={handleUserMenuClose}
        onClick={handleUserMenuClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        <MenuItem>
          <ListItemIcon>
            <AccountCircle fontSize="small" />
          </ListItemIcon>
          Profile Settings
        </MenuItem>
        <Divider />
        <MenuItem>
          <ListItemIcon>
            <ExitToApp fontSize="small" />
          </ListItemIcon>
          Sign Out
        </MenuItem>
      </Menu>

      {/* Notification Center */}
      <NotificationCenter
        open={notificationOpen}
        onClose={() => setNotificationOpen(false)}
      />
    </Box>
  );
};
