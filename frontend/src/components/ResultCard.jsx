/**
 * ResultCard Component
 * Displays the ML model inference results for Bengaluru road traffic:
 * - Semi-circular RiskGauge with calibrated decision threshold (0.45)
 * - Grounded scientific probability statement:
 *   "The supplied conditions correspond to a predicted congestion probability of X%."
 * - Operational impact metrics (Predicted State, Corridor Speed in km/h, Queue Delay in min)
 * - Engineering traffic recommendation advisory
 * - Loading skeleton & error handling with retry
 */

import React from "react";
import RiskGauge from "./RiskGauge";

export default function ResultCard({
  result,
  isLoading,
  error,
  onRetry,
}) {
  // 1. Loading Skeleton State
  if (isLoading) {
    return (
      <section className="card result-card loading-state" aria-busy="true" aria-live="polite">
        <div className="skeleton-header">
          <div className="skeleton skeleton-title" />
          <div className="skeleton skeleton-subtitle" />
        </div>
        <div className="skeleton-gauge-box">
          <div className="skeleton skeleton-gauge" />
        </div>
        <div className="skeleton-metrics-grid">
          <div className="skeleton skeleton-metric-card" />
          <div className="skeleton skeleton-metric-card" />
          <div className="skeleton skeleton-metric-card" />
        </div>
        <div className="skeleton skeleton-banner" />
      </section>
    );
  }

  // 2. Error State with Retry Button
  if (error) {
    return (
      <section className="card result-card error-state" role="alert" aria-live="assertive">
        <div className="error-icon" aria-hidden="true">⚠️</div>
        <h3 className="error-title">Inference Engine Error</h3>
        <p className="error-message">
          {error || "Unable to reach the prediction service. Please ensure the backend is online."}
        </p>
        <button
          type="button"
          className="btn-retry"
          onClick={onRetry}
        >
          🔄 Retry Analysis
        </button>
      </section>
    );
  }

  // 3. Empty State (Before first prediction)
  if (!result) {
    return (
      <section className="card result-card empty-state">
        <div className="empty-icon" aria-hidden="true">🚦</div>
        <h3 className="empty-title">Ready for Analysis</h3>
        <p className="empty-description">
          Select your Bengaluru corridor and conditions on the left, then click{" "}
          <strong>Calculate Congestion Risk</strong> to inspect real-time predictions.
        </p>
      </section>
    );
  }

  // Normalize probability value
  const prob = result.congestion_probability ?? 0;
  const probPercent = (prob * 100).toFixed(1);
  const threshold = result.threshold ?? 0.45;

  return (
    <section className="card result-card active-result" aria-live="polite">
      {/* Result Header */}
      <div className="result-header">
        <div className="result-header-text">
          <span className="result-sub-badge">Logistic Regression Risk Assessment</span>
          <h2 className="result-road-title">
            {result.road_name}
          </h2>
          {result.area_name && (
            <span className="result-area-tag">📍 {result.area_name}, Bengaluru</span>
          )}
        </div>
      </div>

      {/* Probability Gauge */}
      <div className="gauge-section">
        <RiskGauge
          probability={prob}
          riskLevel={result.risk_level}
          threshold={threshold}
        />
      </div>

      {/* Grounded Scientific Probability Statement */}
      <div className="scientific-statement-banner">
        <span className="statement-icon" aria-hidden="true">ℹ️</span>
        <p className="statement-text">
          The supplied conditions correspond to a predicted congestion probability of{" "}
          <strong>{probPercent}%</strong>.
        </p>
      </div>

      {/* Operational Metrics Cards Grid */}
      <div className="metrics-grid">
        {/* Metric 1: Corridor Status */}
        <div className="metric-card">
          <span className="metric-label">Predicted State</span>
          <span className={`metric-value status-${result.risk_level.toLowerCase()}`}>
            {result.label || (result.prediction === 1 ? "High Congestion" : "Normal Flow")}
          </span>
          <span className="metric-subtext">
            {result.prediction === 1 ? "Corridor Saturated" : "Within Road Capacity"}
          </span>
        </div>

        {/* Metric 2: Estimated Speed */}
        <div className="metric-card">
          <span className="metric-label">Corridor Speed</span>
          <span className="metric-value">
            {result.estimated_speed_kmh} km/h
          </span>
          <span className="metric-subtext">Estimated average speed</span>
        </div>

        {/* Metric 3: Queue Delay */}
        <div className="metric-card">
          <span className="metric-label">Queue Delay</span>
          <span className="metric-value highlight-delay">
            +{result.estimated_delay_min} min
          </span>
          <span className="metric-subtext">Added intersection queue</span>
        </div>
      </div>

      {/* Recommendation Advisory Box */}
      {result.recommendation && (
        <div className={`recommendation-box rec-${result.risk_level.toLowerCase()}`}>
          <div className="rec-header">
            <span className="rec-icon" aria-hidden="true">💡</span>
            <span className="rec-title">Traffic Engineering Advisory</span>
          </div>
          <p className="rec-body">{result.recommendation}</p>
        </div>
      )}
    </section>
  );
}
