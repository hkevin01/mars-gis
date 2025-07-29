import React, { Suspense } from 'react';
import { Helmet } from 'react-helmet-async';
import { Route, Routes } from 'react-router-dom';

import { Layout } from './components/Layout';
import { LoadingScreen } from './components/LoadingScreen';

// Lazy load main views
const Dashboard = React.lazy(() => import('./views/Dashboard'));
const Mars3DViewer = React.lazy(() => import('./views/Mars3DViewer'));
const InteractiveMap = React.lazy(() => import('./views/InteractiveMap'));
const MissionPlanner = React.lazy(() => import('./views/MissionPlanner'));
const DataAnalysis = React.lazy(() => import('./views/DataAnalysis'));
const TerrainAnalysis = React.lazy(() => import('./views/TerrainAnalysis'));
const PathPlanning = React.lazy(() => import('./views/PathPlanning'));
const Settings = React.lazy(() => import('./views/Settings'));
const Documentation = React.lazy(() => import('./views/Documentation'));
const NotFound = React.lazy(() => import('./views/NotFound'));

const App: React.FC = () => {
  return (
    <>
      <Helmet>
        <title>MARS-GIS | Professional Mars Exploration Platform</title>
        <meta 
          name="description" 
          content="Advanced geospatial analysis and mission planning platform for Mars exploration. Features 3D terrain visualization, AI-powered analysis, and comprehensive mission planning tools." 
        />
        <meta name="keywords" content="mars, gis, geospatial, space exploration, 3d visualization, mission planning" />
        <meta name="author" content="MARS-GIS Development Team" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta name="theme-color" content="#ff6b35" />
        
        {/* Open Graph / Facebook */}
        <meta property="og:type" content="website" />
        <meta property="og:title" content="MARS-GIS | Professional Mars Exploration Platform" />
        <meta property="og:description" content="Advanced geospatial analysis and mission planning platform for Mars exploration." />
        <meta property="og:url" content="https://mars-gis.space" />
        <meta property="og:site_name" content="MARS-GIS" />
        
        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="MARS-GIS | Professional Mars Exploration Platform" />
        <meta name="twitter:description" content="Advanced geospatial analysis and mission planning platform for Mars exploration." />
        
        {/* Favicon */}
        <link rel="icon" type="image/x-icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png" />
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png" />
        <link rel="manifest" href="/site.webmanifest" />
      </Helmet>

      <Layout>
        <Suspense fallback={<LoadingScreen />}>
          <Routes>
            {/* Main Dashboard */}
            <Route path="/" element={<Dashboard />} />
            <Route path="/dashboard" element={<Dashboard />} />
            
            {/* Visualization Views */}
            <Route path="/mars-3d" element={<Mars3DViewer />} />
            <Route path="/interactive-map" element={<InteractiveMap />} />
            
            {/* Mission Planning */}
            <Route path="/mission-planner" element={<MissionPlanner />} />
            <Route path="/path-planning" element={<PathPlanning />} />
            
            {/* Analysis Tools */}
            <Route path="/data-analysis" element={<DataAnalysis />} />
            <Route path="/terrain-analysis" element={<TerrainAnalysis />} />
            
            {/* System */}
            <Route path="/settings" element={<Settings />} />
            <Route path="/documentation" element={<Documentation />} />
            
            {/* 404 Not Found */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </Layout>
    </>
  );
};

export default App;
