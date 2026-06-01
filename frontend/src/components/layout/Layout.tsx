import { NavLink, useLocation } from 'react-router-dom';

const navItems = [
  { to: '/', label: 'Dashboard', icon: 'DB' },
  { to: '/editor', label: 'Live Editor', icon: 'LE' },
  { to: '/studio', label: 'Dubbing Studio', icon: 'DS' },
  { to: '/archive', label: 'Archive', icon: 'AR' },
  { to: '/status', label: 'AI Status', icon: 'AI' },
];

const pageTitles: Record<string, string> = {
  '/': 'Project Dashboard',
  '/editor': 'Live Editor',
  '/studio': 'Dubbing Studio',
  '/archive': 'Archive',
  '/status': 'AI Status',
};

function Sidebar() {
  return (
    <aside className="studio-sidebar">
      <div className="brand-block">
        <div className="brand-title">AI Transcription Agent</div>
        <div className="brand-subtitle">Deep Space Engine</div>
      </div>

      <nav className="studio-nav" aria-label="Primary navigation">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) => `studio-nav-item ${isActive ? 'active' : ''}`}
          >
            <span className="nav-glyph">{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <NavLink to="/editor" className="primary-action">
          <span className="nav-glyph">+</span>
          New Project
        </NavLink>
        <div className="sidebar-meta">Powered by Whisper, Parakeet TDT, and OmniVoice.</div>
      </div>
    </aside>
  );
}

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const title = pageTitles[location.pathname] || 'AI Transcription Agent';

  return (
    <div className="studio-shell">
      <Sidebar />
      <div className="studio-workspace">
        <header className="studio-topbar">
          <div className="topbar-left">
            <strong>{title}</strong>
            <nav className="topbar-tabs" aria-label="Secondary navigation">
              <span className="topbar-tab active">Projects</span>
              <span className="topbar-tab">Models</span>
              <span className="topbar-tab">API</span>
            </nav>
          </div>
          <div className="topbar-right">
            <label className="search-control">
              <span>Search</span>
              <input type="search" placeholder="Search tasks..." />
            </label>
            <div className="status-pill">
              <span className="status-dot" />
              Systems Nominal
            </div>
          </div>
        </header>
        <main className="studio-content">{children}</main>
      </div>
    </div>
  );
}
