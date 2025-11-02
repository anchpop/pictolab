import { Link } from 'react-router-dom';
import './Home.css';

function Home() {
  return (
    <div className="home">
      <header className="home-header">
        <h1>Pictolab</h1>
        <p className="tagline">Experimental photo generation and manipulation playground</p>
      </header>

      <main className="home-main">
        <section className="experiments">
          <h2>Experiments</h2>
          <div className="experiment-grid">
            <Link to="/lab-inversion" className="experiment-card">
              <h3>Smart Invert</h3>
              <p>Invert the brightness without messing up the colors</p>
            </Link>

            <Link to="/equalize-light" className="experiment-card">
              <h3>Equalize Light</h3>
              <p>Flatten all the brightness while keeping the colors</p>
            </Link>

            <Link to="/laberation" className="experiment-card">
              <h3>Laberation</h3>
              <p>Chromatic aberration, but in LAB color space</p>
            </Link>

            <div className="experiment-card coming-soon">
              <h3>More Coming Soon</h3>
              <p>Stay tuned for more photo experiments</p>
            </div>
          </div>
        </section>
      </main>

      <footer className="home-footer">
        <p>Built with Rust + WASM + React</p>
      </footer>
    </div>
  );
}

export default Home;
