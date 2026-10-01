/**
 * Bengaluru Road Traffic Congestion Risk Prediction
 * React 18 + Vite Operations Dashboard Root
 */

import { useState, useEffect, useCallback } from "react";
import Header from "./components/Header";
import Footer from "./components/Footer";
import PredictionForm from "./components/PredictionForm";
import ResultCard from "./components/ResultCard";
import RecentPredictions from "./components/RecentPredictions";
import Chat from "./Chat";
import { CopilotAvatarIcon } from "./components/ChatIcons";
import { checkHealth, getPrediction, getOptions } from "./api";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const INITIAL_FORM_STATE = {
  road_name: "Sony World Junction",
  area_name: "Koramangala",
  weather_condition: "Clear",
  roadwork: false,
  day_of_week: 0,
  month: 10,
};

function getInitialTheme() {
  try {
    const saved = localStorage.getItem("bengaluru_traffic_theme");
    if (saved === "light" || saved === "dark") {
      return saved;
    }
  } catch (err) {
    console.warn("localStorage access denied, using system preference", err);
  }
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export default function App() {
  const [theme, setTheme] = useState(getInitialTheme);
  const [apiStatus, setApiStatus] = useState("checking");
  const [optionsData, setOptionsData] = useState(null);
  const [formData, setFormData] = useState(INITIAL_FORM_STATE);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isChatWide, setIsChatWide] = useState(false);

  // Keyboard shortcut listener (Escape to close, Ctrl+/ to open)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape" && isChatOpen) {
        setIsChatOpen(false);
      } else if ((e.ctrlKey || e.metaKey) && e.key === "/") {
        e.preventDefault();
        setIsChatOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isChatOpen]);

  // Lock body scroll when chat drawer is open on mobile
  useEffect(() => {
    if (isChatOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isChatOpen]);

  // 1. Sync theme with root HTML element and localStorage
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("bengaluru_traffic_theme", theme);
    } catch (err) {
      console.warn("Unable to persist theme to localStorage", err);
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  // 2. Periodic API health check & initial options fetch
  useEffect(() => {
    let isMounted = true;

    async function pollHealth() {
      const res = await checkHealth();
      if (isMounted) {
        setApiStatus(res.status);
      }
    }

    async function fetchMetadata() {
      try {
        const opts = await getOptions();
        if (isMounted) {
          setOptionsData(opts);
        }
      } catch (err) {
        console.warn("Could not load backend options metadata:", err);
      }
    }

    pollHealth();
    fetchMetadata();
    const interval = setInterval(pollHealth, 30000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // 3. Inference Execution Handler
  const executePrediction = useCallback(async (inputsToUse) => {
    const inputs = inputsToUse || formData;
    setIsLoading(true);
    setError(null);

    try {
      const predData = await getPrediction(inputs);
      setResult(predData);

      // Add to recent predictions history (retain max 5)
      setHistory((prev) => {
        const entry = {
          id: `${Date.now()}-${Math.random().toString(36).substr(2, 4)}`,
          inputs: { ...inputs },
          result: predData,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        };
        return [entry, ...prev.slice(0, 4)];
      });
    } catch (err) {
      console.error("Prediction failed:", err);
      setError(err.message || "Failed to retrieve traffic prediction from backend.");
    } finally {
      setIsLoading(false);
    }
  }, [formData]);

  // Run initial prediction on mount
  useEffect(() => {
    executePrediction(INITIAL_FORM_STATE);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle Preset Selection
  const handleSelectPreset = (preset) => {
    const updated = {
      ...formData,
      road_name: preset.road_name,
      area_name: preset.area_name || formData.area_name,
      day_of_week: preset.day_of_week,
      month: preset.month,
      weather_condition: preset.weather_condition,
      roadwork: preset.roadwork,
    };
    setFormData(updated);
    executePrediction(updated);
  };

  // Handle History Item Recall
  const handleSelectHistoryItem = (item) => {
    setFormData(item.inputs);
    setResult(item.result);
  };

  // Clear History
  const handleClearHistory = () => {
    setHistory([]);
  };

  // Form Reset
  const handleReset = () => {
    setFormData(INITIAL_FORM_STATE);
  };

  return (
    <div className="app-shell">
      <div className="app-container">
        {/* Header Component */}
        <Header apiStatus={apiStatus} theme={theme} onToggleTheme={toggleTheme} />

        {/* Main Content Layout */}
        <main className="app-main">
          <div className="main-grid">
            {/* Left Column: Form & Recent Predictions */}
            <div className="left-column">
              <PredictionForm
                formData={formData}
                optionsData={optionsData}
                onChange={setFormData}
                onSubmit={() => executePrediction(formData)}
                onReset={handleReset}
                isLoading={isLoading}
                onSelectPreset={handleSelectPreset}
              />

              <RecentPredictions
                history={history}
                onSelectHistoryItem={handleSelectHistoryItem}
                onClearHistory={handleClearHistory}
              />
            </div>

            {/* Right Column: Result Card & Risk Assessment */}
            <div className="right-column">
              <ResultCard
                result={result}
                isLoading={isLoading}
                error={error}
                onRetry={() => executePrediction(formData)}
              />
            </div>
          </div>
        </main>

        {/* Footer Component with ML Specifications */}
        <Footer optionsData={optionsData} />
      </div>

      {/* Floating AI Traffic Copilot FAB Button */}
      <button
        type="button"
        className="chat-fab-btn"
        onClick={() => setIsChatOpen(true)}
        aria-label="Open Bengaluru AI Traffic Assistant Copilot (Ctrl + /)"
        title="Open Bengaluru AI Traffic Copilot (Ctrl + /)"
      >
        <div className="fab-icon-wrapper">
          <CopilotAvatarIcon size={22} />
          <span className="fab-pulse-ping" />
        </div>
        <span className="fab-label">AI Traffic Copilot</span>
        <span className="fab-status-badge">
          <span className="fab-live-dot" /> Live
        </span>
      </button>

      {/* AI Assistant Off-Canvas Drawer */}
      {isChatOpen && (
        <div
          className="chat-drawer-overlay"
          onClick={() => setIsChatOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-label="Bengaluru AI Traffic Copilot Drawer"
        >
          <div
            className={`chat-drawer-container ${isChatWide ? "drawer-wide" : ""}`}
            onClick={(e) => e.stopPropagation()}
          >
            <Chat
              apiUrl={API_URL}
              onClose={() => setIsChatOpen(false)}
              isWide={isChatWide}
              onToggleWide={() => setIsChatWide((prev) => !prev)}
            />
          </div>
        </div>
      )}
    </div>
  );
}