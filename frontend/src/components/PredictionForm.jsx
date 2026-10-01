/**
 * PredictionForm Component for Bengaluru Road Traffic Congestion Prediction.
 * Built dynamically from /options:
 * - Area dropdown that dynamically filters the Road dropdown
 * - Weather dropdown with emoji icons
 * - Day-of-week segmented control (Mon-Sun with weekend indicator)
 * - Month selector dropdown
 * - Active roadwork / carriageway constriction toggle
 * - Preset scenario buttons
 * - Inline validation and clear loading state
 */

import React, { useState, useMemo } from "react";
import {
  DEFAULT_ROADS,
  WEATHER_OPTIONS,
  DAYS_OF_WEEK,
  MONTHS,
  PRESET_SCENARIOS,
} from "../presets";

export default function PredictionForm({
  formData,
  optionsData,
  onChange,
  onSubmit,
  onReset,
  isLoading,
  onSelectPreset,
}) {
  const [selectedArea, setSelectedArea] = useState(() => {
    // If formData has area_name, use it; else find area for formData.road_name
    const found = DEFAULT_ROADS.find((r) => r.name === formData.road_name);
    return found ? found.area : "All";
  });

  // Extract areas and road mappings from optionsData or fallbacks
  const areas = useMemo(() => {
    if (optionsData?.areas && optionsData.areas.length > 0) {
      return ["All", ...optionsData.areas];
    }
    const set = new Set(DEFAULT_ROADS.map((r) => r.area));
    return ["All", ...Array.from(set)];
  }, [optionsData]);

  // Roads filtered by selected area
  const filteredRoads = useMemo(() => {
    if (selectedArea === "All") {
      return optionsData?.roads
        ? optionsData.roads.map((name) => {
            const found = DEFAULT_ROADS.find((r) => r.name === name);
            return {
              name,
              area: optionsData?.road_to_area?.[name] || found?.area || "Bengaluru",
              desc: found?.desc || "",
            };
          })
        : DEFAULT_ROADS;
    }

    if (optionsData?.area_to_roads?.[selectedArea]) {
      return optionsData.area_to_roads[selectedArea].map((name) => {
        const found = DEFAULT_ROADS.find((r) => r.name === name);
        return {
          name,
          area: selectedArea,
          desc: found?.desc || "",
        };
      });
    }

    return DEFAULT_ROADS.filter((r) => r.area === selectedArea);
  }, [selectedArea, optionsData]);

  const handleInputChange = (field, value) => {
    onChange({ ...formData, [field]: value });
  };

  const handleAreaChange = (area) => {
    setSelectedArea(area);
    if (area === "All") {
      return;
    }
    // Automatically select the first road in this area if current road not in area
    const roadsInArea = optionsData?.area_to_roads?.[area] ||
      DEFAULT_ROADS.filter((r) => r.area === area).map((r) => r.name);
    if (roadsInArea && roadsInArea.length > 0 && !roadsInArea.includes(formData.road_name)) {
      handleInputChange("road_name", roadsInArea[0]);
    }
  };

  const handleRoadChange = (roadName) => {
    handleInputChange("road_name", roadName);
    // Sync area dropdown
    const roadArea = optionsData?.road_to_area?.[roadName] ||
      DEFAULT_ROADS.find((r) => r.name === roadName)?.area;
    if (roadArea && selectedArea !== "All" && selectedArea !== roadArea) {
      setSelectedArea(roadArea);
    }
  };

  return (
    <section className="card prediction-form-card" aria-labelledby="form-heading">
      <div className="form-card-header">
        <h2 id="form-heading" className="card-title">
          <span className="card-title-icon" aria-hidden="true">🧭</span> Plan Commute & Predict Risk
        </h2>
        <p className="card-description">
          Select your Bengaluru corridor, calendar timing, and meteorological conditions.
        </p>
      </div>

      {/* Preset Scenario Quick-Buttons */}
      <div className="presets-section">
        <span className="presets-label">⚡ Quick Scenarios:</span>
        <div className="presets-scroller" role="group" aria-label="Preset scenarios">
          {PRESET_SCENARIOS.map((sc) => (
            <button
              key={sc.id}
              type="button"
              className="preset-chip"
              onClick={() => {
                onSelectPreset(sc);
                if (sc.area_name) {
                  setSelectedArea(sc.area_name);
                }
              }}
              title={`${sc.name}: ${sc.road_name} (${sc.weather_condition})`}
            >
              <span className="preset-icon" aria-hidden="true">{sc.icon}</span>
              <span className="preset-name">{sc.name}</span>
            </button>
          ))}
        </div>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit();
        }}
        className="prediction-form"
      >
        {/* 1. Geographic Selection: Area Filter + Road Dropdown */}
        <div className="form-row-two-col">
          {/* Area Filter */}
          <div className="form-group">
            <label htmlFor="area-select" className="form-label">
              Urban Area / Zone
            </label>
            <div className="select-wrapper">
              <select
                id="area-select"
                className="form-select"
                value={selectedArea}
                onChange={(e) => handleAreaChange(e.target.value)}
              >
                {areas.map((a) => (
                  <option key={a} value={a}>
                    {a === "All" ? "📍 All Bengaluru Areas (16 Roads)" : `📍 ${a}`}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Road / Intersection Selection */}
          <div className="form-group">
            <label htmlFor="road-select" className="form-label">
              Corridor / Major Intersection
            </label>
            <div className="select-wrapper">
              <select
                id="road-select"
                className="form-select"
                value={formData.road_name}
                onChange={(e) => handleRoadChange(e.target.value)}
                required
              >
                {filteredRoads.map((road) => (
                  <option key={road.name} value={road.name}>
                    {road.name} ({road.area})
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* 2. Day of Week Segmented Control */}
        <div className="form-group">
          <label className="form-label" id="day-of-week-label">
            Day of the Week
          </label>
          <div
            className="segmented-control"
            role="radiogroup"
            aria-labelledby="day-of-week-label"
          >
            {DAYS_OF_WEEK.map((day) => {
              const isSelected = formData.day_of_week === day.id;
              return (
                <button
                  key={day.id}
                  type="button"
                  role="radio"
                  aria-checked={isSelected}
                  className={`segmented-button ${isSelected ? "selected" : ""} ${day.isWeekend ? "weekend" : ""}`}
                  onClick={() => handleInputChange("day_of_week", day.id)}
                >
                  <span className="seg-day-short">{day.short}</span>
                  {day.isWeekend && <span className="seg-weekend-badge">W/E</span>}
                </button>
              );
            })}
          </div>
        </div>

        {/* 3. Calendar Month & Weather Condition */}
        <div className="form-row-two-col">
          {/* Calendar Month Dropdown */}
          <div className="form-group">
            <label htmlFor="month-select" className="form-label">
              Month of Travel
            </label>
            <div className="select-wrapper">
              <select
                id="month-select"
                className="form-select"
                value={formData.month}
                onChange={(e) => handleInputChange("month", parseInt(e.target.value, 10))}
              >
                {MONTHS.map((m) => (
                  <option key={m.id} value={m.id}>
                    🗓️ {m.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Weather Condition Picker */}
          <div className="form-group">
            <label htmlFor="weather-select" className="form-label">
              Weather Condition
            </label>
            <div className="select-wrapper">
              <select
                id="weather-select"
                className="form-select"
                value={formData.weather_condition}
                onChange={(e) => handleInputChange("weather_condition", e.target.value)}
              >
                {WEATHER_OPTIONS.map((w) => (
                  <option key={w.value} value={w.value}>
                    {w.emoji} {w.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* 4. Infrastructure Disruption: Active Roadwork Toggle */}
        <div className="form-group-toggle">
          <label className="toggle-switch-label" htmlFor="roadwork-toggle">
            <span className="toggle-text">
              <span className="toggle-title">🚧 Active Carriageway Roadwork</span>
              <span className="toggle-subtitle">
                Civic digging, utility work, or lane constriction on this corridor
              </span>
            </span>
            <input
              id="roadwork-toggle"
              type="checkbox"
              className="toggle-checkbox"
              checked={formData.roadwork}
              onChange={(e) => handleInputChange("roadwork", e.target.checked)}
            />
            <span className="toggle-slider" aria-hidden="true" />
          </label>
        </div>

        {/* Form Action Buttons */}
        <div className="form-actions">
          <button
            type="submit"
            className="btn btn-primary btn-submit"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                <span>Evaluating Road Risk...</span>
              </>
            ) : (
              <>
                <span>⚡ Calculate Congestion Risk</span>
              </>
            )}
          </button>

          <button
            type="button"
            className="btn btn-secondary"
            onClick={onReset}
            disabled={isLoading}
          >
            ↺ Reset
          </button>
        </div>
      </form>
    </section>
  );
}
