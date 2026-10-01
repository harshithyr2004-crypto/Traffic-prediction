/**
 * RecentPredictions Component
 * Displays the 5 most recent Bengaluru traffic predictions with 1-click parameter reload.
 */

import React from "react";
import { DAYS_OF_WEEK } from "../presets";

export default function RecentPredictions({
  history = [],
  onSelectHistoryItem,
  onClearHistory,
}) {
  if (!history || history.length === 0) {
    return null;
  }

  return (
    <section className="card recent-predictions-card" aria-labelledby="history-heading">
      <div className="recent-header">
        <div className="recent-title-group">
          <span className="recent-icon" aria-hidden="true">🕒</span>
          <h3 id="history-heading" className="recent-title">
            Recent Predictions ({history.length})
          </h3>
        </div>
        <button
          type="button"
          className="btn-clear-history"
          onClick={onClearHistory}
          title="Clear recent predictions history"
        >
          Clear History
        </button>
      </div>

      <div className="recent-list" role="list">
        {history.map((item, idx) => {
          const prob = item.result.congestion_probability ?? 0;
          const probPct = (prob * 100).toFixed(0);
          const dayName =
            DAYS_OF_WEEK.find((d) => d.id === item.inputs.day_of_week)?.short || "Day";
          const riskClass = `risk-badge-${(item.result.risk_level || "low").toLowerCase()}`;

          return (
            <div
              key={item.id || idx}
              className="recent-item"
              role="listitem"
              onClick={() => onSelectHistoryItem(item)}
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelectHistoryItem(item);
                }
              }}
            >
              <div className="recent-item-main">
                <span className="recent-road-name">{item.inputs.road_name}</span>
                <span className="recent-meta">
                  {dayName} • {item.inputs.weather_condition}
                  {item.inputs.roadwork && " • 🚧 Roadwork"}
                </span>
              </div>

              <div className="recent-item-status">
                <span className="recent-prob-badge">{probPct}%</span>
                <span className={`recent-risk-badge ${riskClass}`}>
                  {item.result.risk_level}
                </span>
                <button
                  type="button"
                  className="btn-recall"
                  title="Reload this prediction"
                  aria-label={`Reload prediction for ${item.inputs.road_name}`}
                >
                  ↩
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
