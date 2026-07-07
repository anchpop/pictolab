import Editor from './pages/Editor';
import Wiggle from './pages/Wiggle';

function App() {
  // Path-based routing without a router: Vercel rewrites every path to
  // index.html, so plain <a href> navigation between pages works.
  const path = window.location.pathname.replace(/\/+$/, '');
  if (path === '/wiggle') return <Wiggle />;
  return <Editor />;
}

export default App;
