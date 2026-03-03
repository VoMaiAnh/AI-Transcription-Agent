/**
 * Layout Components
 */
import { Link, useLocation } from 'react-router-dom';

export function Navbar() {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Link to="/" className="navbar-logo">
          AI Transcription
        </Link>
      </div>
      <div className="navbar-menu">
        <Link
          to="/"
          className={`navbar-item ${isActive('/') ? 'active' : ''}`}
        >
          <span className="navbar-icon">🎙️</span>
          Transcription
        </Link>
        <Link
          to="/tts"
          className={`navbar-item ${isActive('/tts') ? 'active' : ''}`}
        >
          <span className="navbar-icon">🔊</span>
          Text-to-Speech
        </Link>
        <Link
          to="/history"
          className={`navbar-item ${isActive('/history') ? 'active' : ''}`}
        >
          <span className="navbar-icon">📋</span>
          History
        </Link>
      </div>
    </nav>
  );
}

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="layout">
      <Navbar />
      <main className="main-content">
        {children}
      </main>
      <footer className="footer">
        <p>Powered by Whisper, Qwen3-ASR, and Qwen3-TTS</p>
      </footer>
    </div>
  );
}