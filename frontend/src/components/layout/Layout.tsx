import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/", label: "Dashboard", icon: "DB" },
  { to: "/editor", label: "Live Editor", icon: "LE" },
  { to: "/studio", label: "Dubbing Studio", icon: "DS" },
  { to: "/archive", label: "Archive", icon: "AR" },
  { to: "/status", label: "AI Status", icon: "AI" },
];

const syncraLogo = "/Logo/V1/Logo_without_BG.png";
const sidebarStorageKey = "syncra-sidebar-collapsed";

function PlusIcon() {
  return (
    <svg
      className="primary-action-plus"
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

function SidebarToggleIcon({ collapsed }: { collapsed: boolean }) {
  return (
    <svg
      className="sidebar-toggle-icon"
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
    >
      {collapsed ? <path d="M9 6l6 6-6 6" /> : <path d="M15 6l-6 6 6 6" />}
    </svg>
  );
}

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

function Sidebar({ collapsed, onToggle }: SidebarProps) {
  return (
    <aside className="studio-sidebar" aria-label="Studio navigation">
      <div className="brand-row">
        <div className="brand-block">
          <img
            className="brand-logo"
            src={syncraLogo}
            alt=""
            aria-hidden="true"
          />
          <div className="brand-copy">
            <div className="brand-title">Syncra</div>
            <div className="brand-subtitle">AI Media Studio</div>
          </div>
        </div>

        <button
          type="button"
          className="sidebar-collapse-button"
          onClick={onToggle}
          aria-label={collapsed ? "Expand sidebar" : "Narrow sidebar"}
          title={collapsed ? "Expand sidebar" : "Narrow sidebar"}
        >
          <SidebarToggleIcon collapsed={collapsed} />
        </button>
      </div>

      <nav className="studio-nav" aria-label="Primary navigation">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              `studio-nav-item ${isActive ? "active" : ""}`
            }
            title={item.label}
          >
            <span className="nav-glyph">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <NavLink to="/editor" className="primary-action">
          <span className="primary-action-icon" aria-hidden="true">
            <PlusIcon />
          </span>
          <span className="primary-action-label">New Project</span>
        </NavLink>
        <div className="sidebar-meta">
          Powered by Whisper, Parakeet TDT, and Supertonic 3.
        </div>
      </div>
    </aside>
  );
}

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }

    return window.localStorage.getItem(sidebarStorageKey) === "true";
  });

  useEffect(() => {
    window.localStorage.setItem(sidebarStorageKey, String(sidebarCollapsed));
  }, [sidebarCollapsed]);

  return (
    <div className={`studio-shell ${sidebarCollapsed ? "sidebar-narrow" : ""}`}>
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((current) => !current)}
      />
      <div className="studio-workspace">
        <main className="studio-content">{children}</main>
      </div>
    </div>
  );
}
