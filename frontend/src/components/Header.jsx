import React from "react";

/**
 * Header Component for Bengaluru Road Traffic Congestion Risk Dashboard.
 * Displays title, live API status badge (polled every 30s), and light/dark theme switch.
 *
 * @param {Object} props
 * @param {"checking"|"online"|"offline"} props.apiStatus - Current backend connectivity status
 * @param {"light"|"dark"} props.theme - Active visual theme
 * @param {Function} props.onToggleTheme - Callback when user clicks the theme toggle button
 */
export default function Header({ apiStatus, theme, onToggleTheme }) {
  const statusConfig = {
    online: { label: "API Online", className: "status-online" },
    offline: { label: "API Offline", className: "status-offline" },
    checking: { label: "Checking API...", className: "status-checking" },
  }[apiStatus] || { label: "Checking API...", className: "status-checking" };

  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="header-title-row">
          <span className="header-icon" aria-hidden="true">🚦</span>
          <h1 className="header-title">Bengaluru Traffic Congestion Risk</h1>
        </div>
        <p className="header-subtitle">
          Calibrated Logistic Regression & Surface Road Intelligence across 16 Primary Corridors
        </p>
      </div>

      <div className="header-actions">
        {/* API Health Status Badge - checked on load and every 30s */}
        <div
          className={`api-status-badge ${statusConfig.className}`}
          role="status"
          aria-live="polite"
          title={`Backend status: ${statusConfig.label}`}
        >
          <span className="status-dot" aria-hidden="true" />
          <span className="status-label">{statusConfig.label}</span>
        </div>

        {/* Theme Toggle Button (Light/Dark Mode) */}
        <button
          type="button"
          className="btn-theme-toggle"
          onClick={onToggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          {theme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode"}
        </button>
      </div>
    </header>
  );
}
