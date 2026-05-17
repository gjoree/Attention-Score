import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home  from './pages/Home';
import ModeA from './pages/ModeA';
import ModeB from './pages/ModeB';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"       element={<Home />}  />
        <Route path="/mode-a" element={<ModeA />} />
        <Route path="/mode-b" element={<ModeB />} />
      </Routes>
    </BrowserRouter>
  );
}
