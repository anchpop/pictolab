import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import LabInversion from './pages/LabInversion';
import EqualizeLight from './pages/EqualizeLight';
import Laberation from './pages/Laberation';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/lab-inversion" element={<LabInversion />} />
        <Route path="/equalize-light" element={<EqualizeLight />} />
        <Route path="/laberation" element={<Laberation />} />
      </Routes>
    </Router>
  );
}

export default App;
