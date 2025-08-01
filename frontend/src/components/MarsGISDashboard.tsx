import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Bell, 
  Settings, 
  Upload, 
  Download, 
  Play, 
  Pause, 
  BarChart3, 
  Globe, 
  Rocket, 
  Users, 
  FileText, 
  AlertTriangle, 
  CheckCircle,
  Clock,
  Activity,
  MapPin,
  Zap,
  TrendingUp,
  Eye,
  Filter,
  RefreshCw,
  Plus,
  Edit,
  Trash2,
  X
} from 'lucide-react';

// Types
interface Mission {
  id: string;
  name: string;
  status: 'active' | 'planned' | 'completed' | 'on-hold';
  type: 'rover' | 'orbital' | 'surface';
  coordinates: [number, number];
  progress: number;
  riskLevel: 'low' | 'medium' | 'high';
  startDate: string;
  endDate?: string;
  team: string[];
}

interface SystemHealth {
  cpu: number;
  memory: number;
  storage: number;
  network: number;
  mlModels: number;
}

interface Notification {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
}

// Sample Data
const sampleMissions: Mission[] = [
  {
    id: 'mission-001',
    name: 'Olympus Mons Survey',
    status: 'active',
    type: 'rover',
    coordinates: [-18.65, 226.2],
    progress: 75,
    riskLevel: 'medium',
    startDate: '2025-07-15',
    team: ['Dr. Sarah Chen', 'Mike Rodriguez', 'Alex Kim']
  },
  {
    id: 'mission-002',
    name: 'Polar Ice Analysis',
    status: 'planned',
    type: 'orbital',
    coordinates: [85.0, 0.0],
    progress: 0,
    riskLevel: 'low',
    startDate: '2025-08-20',
    team: ['Emma Thompson', 'Dr. James Liu']
  },
  {
    id: 'mission-003',
    name: 'Valles Marineris Exploration',
    status: 'completed',
    type: 'rover',
    coordinates: [-14.0, -59.2],
    progress: 100,
    riskLevel: 'high',
    startDate: '2025-06-01',
    endDate: '2025-07-30',
    team: ['Captain Martinez', 'Dr. Patel', 'Zhang Wei']
  },
  {
    id: 'mission-004',
    name: 'Atmospheric Monitoring',
    status: 'active',
    type: 'surface',
    coordinates: [22.5, 49.97],
    progress: 45,
    riskLevel: 'low',
    startDate: '2025-07-01',
    team: ['Lisa Park', 'Tom Anderson']
  }
];

const sampleNotifications: Notification[] = [
  {
    id: 'notif-001',
    type: 'warning',
    title: 'Mission Alert',
    message: 'Olympus Mons Survey: Dust storm detected in target area',
    timestamp: '2025-08-01T10:30:00Z',
    read: false
  },
  {
    id: 'notif-002',
    type: 'success',
    title: 'Data Processing Complete',
    message: 'Valles Marineris terrain analysis finished successfully',
    timestamp: '2025-08-01T09:15:00Z',
    read: false
  },
  {
    id: 'notif-003',
    type: 'info',
    title: 'System Update',
    message: 'ML models updated with latest Earth-Mars transfer learning data',
    timestamp: '2025-08-01T08:00:00Z',
    read: true
  }
];

// Main Dashboard Component
const MarsGISDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [missions, setMissions] = useState<Mission[]>(sampleMissions);
  const [notifications, setNotifications] = useState<Notification[]>(sampleNotifications);
  const [showModal, setShowModal] = useState(false);
  const [modalType, setModalType] = useState<'mission' | 'settings' | 'upload'>('mission');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(false);
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    cpu: 78,
    memory: 65,
    storage: 82,
    network: 95,
    mlModels: 88
  });

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemHealth(prev => ({
        cpu: Math.max(0, Math.min(100, prev.cpu + (Math.random() - 0.5) * 10)),
        memory: Math.max(0, Math.min(100, prev.memory + (Math.random() - 0.5) * 8)),
        storage: Math.max(0, Math.min(100, prev.storage + (Math.random() - 0.5) * 3)),
        network: Math.max(0, Math.min(100, prev.network + (Math.random() - 0.5) * 5)),
        mlModels: Math.max(0, Math.min(100, prev.mlModels + (Math.random() - 0.5) * 6))
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Filter missions based on search and status
  const filteredMissions = missions.filter(mission => {
    const matchesSearch = mission.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         mission.type.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || mission.status === filterStatus;
    return matchesSearch && matchesStatus;
  });

  // Utility functions
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600 bg-green-100';
      case 'planned': return 'text-blue-600 bg-blue-100';
      case 'completed': return 'text-gray-600 bg-gray-100';
      case 'on-hold': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-green-600';
      case 'medium': return 'text-yellow-600';
      case 'high': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const openModal = (type: 'mission' | 'settings' | 'upload') => {
    setModalType(type);
    setShowModal(true);
  };

  const simulateAction = async (action: string) => {
    setIsLoading(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));
    setIsLoading(false);
    
    // Add notification
    const newNotification: Notification = {
      id: `notif-${Date.now()}`,
      type: 'success',
      title: 'Action Complete',
      message: `${action} completed successfully`,
      timestamp: new Date().toISOString(),
      read: false
    };
    setNotifications(prev => [newNotification, ...prev]);
  };

  const markNotificationAsRead = (id: string) => {
    setNotifications(prev => 
      prev.map(notif => 
        notif.id === id ? { ...notif, read: true } : notif
      )
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div className="w-64 bg-white shadow-lg">
        <div className="p-6">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-red-600 rounded-lg flex items-center justify-center">
              <Globe className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">MARS-GIS</h1>
              <p className="text-sm text-gray-500">v1.0.0</p>
            </div>
          </div>
        </div>

        <nav className="px-4 pb-4">
          {[
            { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
            { id: 'missions', label: 'Missions', icon: Rocket },
            { id: 'data', label: 'Mars Data', icon: Globe },
            { id: 'analytics', label: 'Analytics', icon: TrendingUp },
            { id: 'visualization', label: '3D View', icon: Eye },
            { id: 'team', label: 'Team', icon: Users },
            { id: 'documents', label: 'Documents', icon: FileText },
            { id: 'settings', label: 'Settings', icon: Settings }
          ].map(item => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg mb-1 transition-colors ${
                  activeTab === item.id
                    ? 'bg-red-100 text-red-700 border-r-2 border-red-600'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white shadow-sm border-b">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="flex items-center space-x-4">
              <h2 className="text-2xl font-bold text-gray-900 capitalize">
                {activeTab}
              </h2>
              {isLoading && (
                <div className="flex items-center space-x-2 text-blue-600">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Processing...</span>
                </div>
              )}
            </div>

            <div className="flex items-center space-x-4">
              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <input
                  type="text"
                  placeholder="Search missions, data..."
                  className="pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>

              {/* Notifications */}
              <div className="relative">
                <button className="relative p-2 text-gray-600 hover:text-gray-900 transition-colors">
                  <Bell className="w-5 h-5" />
                  {notifications.filter(n => !n.read).length > 0 && (
                    <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
                  )}
                </button>
              </div>

              {/* User Menu */}
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center">
                  <span className="text-sm font-medium text-gray-700">JD</span>
                </div>
                <span className="text-sm font-medium text-gray-700">Dr. Jane Doe</span>
              </div>
            </div>
          </div>
        </header>

        {/* Content Area */}
        <main className="flex-1 p-6 overflow-auto">
          {activeTab === 'dashboard' && (
            <div className="space-y-6">
              {/* Quick Stats */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-white p-6 rounded-lg shadow">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Active Missions</p>
                      <p className="text-3xl font-bold text-gray-900">
                        {missions.filter(m => m.status === 'active').length}
                      </p>
                    </div>
                    <div className="p-3 bg-green-100 rounded-full">
                      <Rocket className="w-6 h-6 text-green-600" />
                    </div>
                  </div>
                  <p className="text-sm text-green-600 mt-2">↗ 12% from last month</p>
                </div>

                <div className="bg-white p-6 rounded-lg shadow">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Data Processed</p>
                      <p className="text-3xl font-bold text-gray-900">2.4 TB</p>
                    </div>
                    <div className="p-3 bg-blue-100 rounded-full">
                      <Activity className="w-6 h-6 text-blue-600" />
                    </div>
                  </div>
                  <p className="text-sm text-blue-600 mt-2">↗ 8% from last week</p>
                </div>

                <div className="bg-white p-6 rounded-lg shadow">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">ML Accuracy</p>
                      <p className="text-3xl font-bold text-gray-900">97.2%</p>
                    </div>
                    <div className="p-3 bg-purple-100 rounded-full">
                      <Zap className="w-6 h-6 text-purple-600" />
                    </div>
                  </div>
                  <p className="text-sm text-purple-600 mt-2">↗ 0.3% improvement</p>
                </div>

                <div className="bg-white p-6 rounded-lg shadow">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Coverage Area</p>
                      <p className="text-3xl font-bold text-gray-900">847 km²</p>
                    </div>
                    <div className="p-3 bg-red-100 rounded-full">
                      <MapPin className="w-6 h-6 text-red-600" />
                    </div>
                  </div>
                  <p className="text-sm text-red-600 mt-2">↗ 15% expansion</p>
                </div>
              </div>

              {/* System Health */}
              <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">System Health</h3>
                <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                  {Object.entries(systemHealth).map(([key, value]) => {
                    const numValue = typeof value === 'number' ? value : 0;
                    return (
                    <div key={key} className="text-center">
                      <p className="text-sm font-medium text-gray-600 capitalize mb-2">
                        {key.replace(/([A-Z])/g, ' $1').trim()}
                      </p>
                      <div className="relative w-16 h-16 mx-auto">
                        <svg className="w-16 h-16 transform -rotate-90">
                          <circle
                            cx="32"
                            cy="32"
                            r="28"
                            stroke="currentColor"
                            strokeWidth="4"
                            fill="none"
                            className="text-gray-200"
                          />
                          <circle
                            cx="32"
                            cy="32"
                            r="28"
                            stroke="currentColor"
                            strokeWidth="4"
                            fill="none"
                            strokeDasharray={`${2 * Math.PI * 28}`}
                            strokeDashoffset={`${2 * Math.PI * 28 * (1 - numValue / 100)}`}
                            className={`${
                              numValue > 80 ? 'text-green-500' :
                              numValue > 60 ? 'text-yellow-500' : 'text-red-500'
                            }`}
                          />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center">
                          <span className="text-sm font-bold">{Math.round(numValue)}%</span>
                        </div>
                      </div>
                    </div>
                    );
                  })}
                </div>
              </div>

              {/* Recent Activity */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white p-6 rounded-lg shadow">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Missions</h3>
                  <div className="space-y-4">
                    {missions.slice(0, 3).map(mission => (
                      <div key={mission.id} className="flex items-center space-x-4">
                        <div className={`w-3 h-3 rounded-full ${
                          mission.status === 'active' ? 'bg-green-500' :
                          mission.status === 'planned' ? 'bg-blue-500' :
                          mission.status === 'completed' ? 'bg-gray-500' : 'bg-yellow-500'
                        }`}></div>
                        <div className="flex-1">
                          <p className="font-medium text-gray-900">{mission.name}</p>
                          <p className="text-sm text-gray-500">{mission.type} • {mission.progress}% complete</p>
                        </div>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(mission.status)}`}>
                          {mission.status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-white p-6 rounded-lg shadow">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Notifications</h3>
                  <div className="space-y-4">
                    {notifications.slice(0, 3).map(notification => (
                      <div 
                        key={notification.id} 
                        className={`p-3 rounded-lg cursor-pointer transition-colors ${
                          notification.read ? 'bg-gray-50' : 'bg-blue-50 border border-blue-200'
                        }`}
                        onClick={() => markNotificationAsRead(notification.id)}
                      >
                        <div className="flex items-start space-x-3">
                          <div className={`w-2 h-2 rounded-full mt-2 ${
                            notification.type === 'error' ? 'bg-red-500' :
                            notification.type === 'warning' ? 'bg-yellow-500' :
                            notification.type === 'success' ? 'bg-green-500' : 'bg-blue-500'
                          }`}></div>
                          <div className="flex-1">
                            <p className="font-medium text-gray-900">{notification.title}</p>
                            <p className="text-sm text-gray-600">{notification.message}</p>
                            <p className="text-xs text-gray-400 mt-1">
                              {new Date(notification.timestamp).toLocaleString()}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'missions' && (
            <div className="space-y-6">
              {/* Mission Controls */}
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
                <div className="flex items-center space-x-4">
                  <button
                    onClick={() => openModal('mission')}
                    className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                  >
                    <Plus className="w-4 h-4" />
                    <span>New Mission</span>
                  </button>
                  <button
                    onClick={() => simulateAction('Mission data refresh')}
                    className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <RefreshCw className="w-4 h-4" />
                    <span>Refresh</span>
                  </button>
                </div>

                <div className="flex items-center space-x-4">
                  <select
                    value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value)}
                    className="border rounded-lg px-3 py-2 focus:ring-2 focus:ring-red-500 focus:border-transparent"
                  >
                    <option value="all">All Status</option>
                    <option value="active">Active</option>
                    <option value="planned">Planned</option>
                    <option value="completed">Completed</option>
                    <option value="on-hold">On Hold</option>
                  </select>
                  <button className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
                    <Filter className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Mission Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                {filteredMissions.map(mission => (
                  <div key={mission.id} className="bg-white p-6 rounded-lg shadow hover:shadow-lg transition-shadow">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="font-semibold text-gray-900">{mission.name}</h3>
                        <p className="text-sm text-gray-500">{mission.type} mission</p>
                      </div>
                      <div className="flex space-x-2">
                        <button className="p-1 text-gray-400 hover:text-gray-600">
                          <Edit className="w-4 h-4" />
                        </button>
                        <button className="p-1 text-gray-400 hover:text-red-600">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Status</span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(mission.status)}`}>
                          {mission.status}
                        </span>
                      </div>

                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Progress</span>
                        <span className="text-sm font-medium">{mission.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-red-600 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${mission.progress}%` }}
                        ></div>
                      </div>

                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Risk Level</span>
                        <span className={`text-sm font-medium ${getRiskColor(mission.riskLevel)}`}>
                          {mission.riskLevel}
                        </span>
                      </div>

                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Coordinates</span>
                        <span className="text-sm font-mono">
                          {mission.coordinates[0].toFixed(2)}, {mission.coordinates[1].toFixed(2)}
                        </span>
                      </div>

                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Team</span>
                        <span className="text-sm">{mission.team.length} members</span>
                      </div>
                    </div>

                    <div className="mt-4 pt-4 border-t flex space-x-2">
                      <button className="flex-1 px-3 py-2 bg-red-100 text-red-700 rounded hover:bg-red-200 transition-colors text-sm">
                        View Details
                      </button>
                      {mission.status === 'active' && (
                        <button className="px-3 py-2 border border-gray-300 rounded hover:bg-gray-50 transition-colors text-sm">
                          <Pause className="w-4 h-4" />
                        </button>
                      )}
                      {mission.status === 'planned' && (
                        <button className="px-3 py-2 border border-gray-300 rounded hover:bg-gray-50 transition-colors text-sm">
                          <Play className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Other tabs content would be implemented similarly */}
          {activeTab !== 'dashboard' && activeTab !== 'missions' && (
            <div className="bg-white p-8 rounded-lg shadow text-center">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <FileText className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Module
              </h3>
              <p className="text-gray-600 mb-4">
                This section is under development. The {activeTab} functionality will be available in the next update.
              </p>
              <button
                onClick={() => simulateAction(`${activeTab} module initialization`)}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Initialize {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
              </button>
            </div>
          )}
        </main>
      </div>

      {/* Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">
                {modalType === 'mission' && 'Create New Mission'}
                {modalType === 'settings' && 'System Settings'}
                {modalType === 'upload' && 'Upload Data'}
              </h3>
              <button
                onClick={() => setShowModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {modalType === 'mission' && (
              <form className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Mission Name
                  </label>
                  <input
                    type="text"
                    className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-red-500 focus:border-transparent"
                    placeholder="Enter mission name"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Mission Type
                  </label>
                  <select className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-red-500 focus:border-transparent">
                    <option value="rover">Rover Mission</option>
                    <option value="orbital">Orbital Survey</option>
                    <option value="surface">Surface Analysis</option>
                  </select>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Latitude
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-red-500 focus:border-transparent"
                      placeholder="-90 to 90"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Longitude
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-red-500 focus:border-transparent"
                      placeholder="-180 to 180"
                    />
                  </div>
                </div>
                <div className="flex space-x-3 pt-4">
                  <button
                    type="button"
                    onClick={() => {
                      simulateAction('Mission creation');
                      setShowModal(false);
                    }}
                    className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                  >
                    Create Mission
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowModal(false)}
                    className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            )}

            {modalType === 'upload' && (
              <div className="space-y-4">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 mb-2">Drop files here or click to browse</p>
                  <p className="text-sm text-gray-500">Supports: .tif, .jpg, .png, .csv, .json</p>
                </div>
                <div className="flex space-x-3">
                  <button
                    onClick={() => {
                      simulateAction('File upload');
                      setShowModal(false);
                    }}
                    className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                  >
                    Upload
                  </button>
                  <button
                    onClick={() => setShowModal(false)}
                    className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Quick Action Buttons */}
      <div className="fixed bottom-6 right-6 space-y-3">
        <button
          onClick={() => openModal('upload')}
          className="w-12 h-12 bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 transition-colors flex items-center justify-center"
          title="Upload Data"
        >
          <Upload className="w-5 h-5" />
        </button>
        <button
          onClick={() => simulateAction('Data export')}
          className="w-12 h-12 bg-green-600 text-white rounded-full shadow-lg hover:bg-green-700 transition-colors flex items-center justify-center"
          title="Download Report"
        >
          <Download className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};

export default MarsGISDashboard;
