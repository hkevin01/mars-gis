import './App.css';
import { ApiStatusBanner, ApiStatusProvider } from './components/ApiStatusProvider';
import IntegratedMarsExplorer from './components/IntegratedMarsExplorer';

function App() {
  return (
    <ApiStatusProvider>
      <IntegratedMarsExplorer initialView="2d" />
      <ApiStatusBanner />
    </ApiStatusProvider>
  );
}

export default App;
