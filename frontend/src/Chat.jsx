/**
 * Chat.jsx
 * Professional, clean, responsive AI Traffic Operations Assistant for Bengaluru Road Traffic.
 * Powered by Groq / OpenAI-compatible endpoint with Server-Sent Events (SSE).
 *
 * Design Architecture:
 * - Retina-crisp SVG iconography (no generic emojis)
 * - Modern executive glassmorphism styling
 * - Markdown parser supporting tables, code blocks, callouts, lists, and inline metrics
 * - Interactive tool result cards with visual gauges, live weather telemetry, ranking accordion
 * - AbortController stop-generation support
 * - 1-Click Copy message to clipboard with instant feedback
 * - Expandable wide-mode toggle (540px <-> 840px)
 * - WCAG AA accessible with full keyboard navigation
 */

import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  CopilotAvatarIcon,
  UserAvatarIcon,
  SendIcon,
  StopIcon,
  CopyIcon,
  CheckIcon,
  TrashIcon,
  CloseIcon,
  ExpandIcon,
  CollapseIcon,
  SparklesIcon,
  SpeedometerIcon,
  ClockIcon,
  CloudRainIcon,
  SunIcon,
  ConeIcon,
  LightbulbIcon,
  SatelliteIcon,
  ScaleIcon,
  MapPinIcon,
  ArrowUpRightIcon,
  ArrowDownRightIcon,
  ChevronDownIcon,
} from "./components/ChatIcons";

const QUICK_ACTIONS = [
  {
    id: "live-weather",
    icon: <SatelliteIcon size={15} />,
    label: "Live Risk Right Now",
    prompt: "What is the road traffic congestion risk right now in Bengaluru based on live weather?",
  },
  {
    id: "compare-corridors",
    icon: <ScaleIcon size={15} />,
    label: "Compare Corridors",
    prompt: "Compare traffic congestion risk between Sony World Junction (Koramangala) and Tumkur Road (Yeshwanthpur).",
  },
  {
    id: "risk-factors",
    icon: <SparklesIcon size={15} />,
    label: "Top Risk Factors",
    prompt: "Which road and weather factors increase or decrease congestion odds the most according to the model?",
  },
  {
    id: "rank-16",
    icon: <MapPinIcon size={15} />,
    label: "Rank 16 Corridors",
    prompt: "Rank all 16 monitored roads in Bengaluru by their congestion risk under normal clear conditions.",
  },
  {
    id: "commute-day",
    icon: <ClockIcon size={15} />,
    label: "Safest Commute Day",
    prompt: "What is the safest day this week to travel on Sarjapur Road to avoid traffic gridlock?",
  },
];

/**
 * Rich, safe Markdown text & table renderer
 * Zero dependencies. Parses tables, code blocks, blockquotes, lists, bold, italics, code, and metrics.
 */
function FormattedMessage({ content }) {
  if (!content) return null;

  const lines = content.split("\n");
  const renderedElements = [];
  let tableBuffer = [];
  let inCodeBlock = false;
  let codeBuffer = [];
  let codeLanguage = "";

  const parseInline = (text) => {
    if (!text) return "";
    const parts = [];
    let keyIdx = 0;

    // Pattern matching bold, code, italic, and metric pills
    const regex = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
    let match;
    let lastIndex = 0;

    while ((match = regex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      const token = match[0];
      if (token.startsWith("**") && token.endsWith("**")) {
        parts.push(
          <strong key={keyIdx++} className="msg-strong">
            {token.slice(2, -2)}
          </strong>
        );
      } else if (token.startsWith("`") && token.endsWith("`")) {
        parts.push(
          <code key={keyIdx++} className="msg-code">
            {token.slice(1, -1)}
          </code>
        );
      } else if (token.startsWith("*") && token.endsWith("*")) {
        parts.push(
          <em key={keyIdx++} className="msg-em">
            {token.slice(1, -1)}
          </em>
        );
      }
      lastIndex = regex.lastIndex;
    }

    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts.length > 0 ? parts : text;
  };

  const flushTable = (key) => {
    if (tableBuffer.length === 0) return null;

    const rows = tableBuffer.map((row) =>
      row
        .split("|")
        .map((cell) => cell.trim())
        .filter((_, idx, arr) => idx > 0 && idx < arr.length - 1)
    );

    tableBuffer = [];

    if (rows.length < 2) return null;

    const headerRow = rows[0];
    const dataRows = rows.slice(1).filter((r) => !r.every((c) => c.match(/^[-:]+$/)));

    return (
      <div key={`table-${key}`} className="msg-table-container">
        <table className="msg-table">
          <thead>
            <tr>
              {headerRow.map((h, i) => (
                <th key={i}>{parseInline(h)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {dataRows.map((row, rIdx) => (
              <tr key={rIdx}>
                {row.map((cell, cIdx) => (
                  <td key={cIdx}>{parseInline(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const flushCode = (key) => {
    if (codeBuffer.length === 0) return null;
    const codeText = codeBuffer.join("\n");
    codeBuffer = [];
    return (
      <div key={`code-${key}`} className="msg-codeblock-container">
        {codeLanguage && <div className="msg-codeblock-header">{codeLanguage}</div>}
        <pre className="msg-codeblock">
          <code>{codeText}</code>
        </pre>
      </div>
    );
  };

  for (let idx = 0; idx < lines.length; idx++) {
    const rawLine = lines[idx];
    const trimmed = rawLine.trim();

    // Check for fenced code block toggle
    if (trimmed.startsWith("```")) {
      if (inCodeBlock) {
        renderedElements.push(flushCode(idx));
        inCodeBlock = false;
        codeLanguage = "";
      } else {
        if (tableBuffer.length > 0) renderedElements.push(flushTable(`pre-${idx}`));
        inCodeBlock = true;
        codeLanguage = trimmed.slice(3).trim();
      }
      continue;
    }

    if (inCodeBlock) {
      codeBuffer.push(rawLine);
      continue;
    }

    // Check for Markdown table line
    if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
      tableBuffer.push(trimmed);
      continue;
    } else if (tableBuffer.length > 0) {
      renderedElements.push(flushTable(idx));
    }

    if (!trimmed) {
      renderedElements.push(<div key={idx} className="msg-spacer" />);
      continue;
    }

    // Heading 3: ### ...
    if (trimmed.startsWith("### ")) {
      renderedElements.push(
        <h4 key={idx} className="msg-heading-h4">
          {parseInline(trimmed.slice(4))}
        </h4>
      );
      continue;
    }

    // Heading 2: ## ...
    if (trimmed.startsWith("## ")) {
      renderedElements.push(
        <h3 key={idx} className="msg-heading-h3">
          {parseInline(trimmed.slice(3))}
        </h3>
      );
      continue;
    }

    // Blockquote: > ...
    if (trimmed.startsWith("> ")) {
      renderedElements.push(
        <blockquote key={idx} className="msg-blockquote">
          {parseInline(trimmed.slice(2))}
        </blockquote>
      );
      continue;
    }

    // Bullet point: - ... or * ...
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      renderedElements.push(
        <div key={idx} className="msg-bullet-row">
          <span className="msg-bullet-pin" aria-hidden="true" />
          <span className="msg-bullet-text">{parseInline(trimmed.slice(2))}</span>
        </div>
      );
      continue;
    }

    // Numbered list: 1. ...
    const numMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
    if (numMatch) {
      renderedElements.push(
        <div key={idx} className="msg-number-row">
          <span className="msg-number-badge">{numMatch[1]}</span>
          <span className="msg-bullet-text">{parseInline(numMatch[2])}</span>
        </div>
      );
      continue;
    }

    // Standard paragraph
    renderedElements.push(
      <p key={idx} className="msg-paragraph">
        {parseInline(rawLine)}
      </p>
    );
  }

  // Flush any remaining code or table buffers
  if (inCodeBlock && codeBuffer.length > 0) {
    renderedElements.push(flushCode("end"));
  }
  if (tableBuffer.length > 0) {
    renderedElements.push(flushTable("end"));
  }

  return <div className="formatted-msg-body">{renderedElements}</div>;
}

function getFormattedNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export default function Chat({ apiUrl, onClose, isWide, onToggleWide }) {
  const [messages, setMessages] = useState(() => [
    {
      role: "assistant",
      content:
        "Hello! I am your **Bengaluru AI Traffic Operations Copilot**, powered by calibrated Logistic Regression " +
        "on the **Bengaluru City Traffic Dataset** (8,936 observations across 16 arterial corridors).\n\n" +
        "I can assess real-time congestion risks, compare travel corridors, evaluate weather impacts, " +
        "and provide exact mathematical odds ratios ($OR = \\exp(\\beta)$). How can I assist with your Bengaluru commute today?",
      toolResults: [],
      timestamp: getFormattedNow(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState("");
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [showPromptChips, setShowPromptChips] = useState(true);
  const [expandedRankings, setExpandedRankings] = useState(false);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const abortControllerRef = useRef(null);
  const feedRef = useRef(null);
  const assistantTextRef = useRef("");
  const currentToolResultsRef = useRef([]);

  // Auto-scroll when new message or token arrives, unless user is scrolled up
  const scrollToBottom = useCallback((behavior = "smooth") => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  }, []);

  useEffect(() => {
    const feed = feedRef.current;
    if (!feed) return;
    const isNearBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 140;
    if (isNearBottom || isStreaming) {
      scrollToBottom();
    }
  }, [messages, statusMessage, isStreaming, scrollToBottom]);

  // Clean abort controller on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const handleSend = async (textToSend) => {
    const userText = (textToSend || input).trim();
    if (!userText || isStreaming) return;

    setInput("");
    setError("");

    const nowTime = getFormattedNow();

    const newMessages = [
      ...messages,
      { role: "user", content: userText, timestamp: nowTime },
    ];
    setMessages(newMessages);
    setIsStreaming(true);
    setStatusMessage("Consulting Bengaluru Traffic ML Model...");

    currentToolResultsRef.current = [];
    assistantTextRef.current = "";

    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", toolResults: [], timestamp: nowTime },
    ]);

    // Setup AbortController for user-initiated stop
    abortControllerRef.current = new AbortController();

    try {
      const payload = {
        messages: newMessages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
      };

      const response = await fetch(`${apiUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop();

        for (const evt of events) {
          const lines = evt.split("\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const jsonStr = line.slice(6).trim();
              if (!jsonStr) continue;

              try {
                const data = JSON.parse(jsonStr);

                if (data.type === "tool_call") {
                  const friendlyNames = {
                    predict_congestion: "Calculating Corridor Congestion Risk",
                    get_live_conditions: "Fetching Live Bengaluru Weather Telemetry",
                    compare_scenarios: "Simulating Corridor Comparison",
                    rank_roads_by_risk: "Ranking 16 Monitored Corridors",
                    explain_model_factors: "Extracting Model Odds Ratios",
                  };
                  setStatusMessage(`${friendlyNames[data.name] || data.name}...`);
                } else if (data.type === "tool_result") {
                  currentToolResultsRef.current.push({
                    name: data.name,
                    result: data.result,
                  });
                  const updatedResults = [...currentToolResultsRef.current];
                  setMessages((prev) => {
                    const next = [...prev];
                    const lastIdx = next.length - 1;
                    next[lastIdx] = {
                      ...next[lastIdx],
                      toolResults: updatedResults,
                    };
                    return next;
                  });
                } else if (data.type === "text" || data.type === "content_delta") {
                  setStatusMessage("");
                  const delta = data.content ?? data.delta ?? "";
                  assistantTextRef.current += delta;
                  const currentText = assistantTextRef.current;
                  setMessages((prev) => {
                    const next = [...prev];
                    const lastIdx = next.length - 1;
                    next[lastIdx] = {
                      ...next[lastIdx],
                      content: currentText,
                    };
                    return next;
                  });
                } else if (data.type === "error") {
                  setError(data.message || "An unexpected error occurred.");
                  setStatusMessage("");
                } else if (data.type === "done") {
                  setStatusMessage("");
                }
              } catch (parseErr) {
                console.warn("SSE parse error:", parseErr, jsonStr);
              }
            }
          }
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        setStatusMessage("Analysis stopped by user.");
      } else {
        console.error("Chat streaming failure:", err);
        setError(`Connection failure: ${err.message}. Ensure backend is running.`);
        setMessages((prev) => {
          const next = [...prev];
          const lastIdx = next.length - 1;
          if (!next[lastIdx].content) {
            next[lastIdx].content =
              "Unable to reach the AI engine. Please ensure the FastAPI backend is running at " + apiUrl;
          }
          return next;
        });
      }
    } finally {
      setIsStreaming(false);
      setStatusMessage("");
      abortControllerRef.current = null;
      inputRef.current?.focus();
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const handleClear = () => {
    if (isStreaming) return;
    setMessages([
      {
        role: "assistant",
        content:
          "Conversation history cleared. I am ready to evaluate another corridor, compare routes, or analyze live weather in Bengaluru.",
        toolResults: [],
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);
    setError("");
    setStatusMessage("");
  };

  const handleCopyMessage = (text, index) => {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    });
  };

  /**
   * Render custom interactive card for each tool result
   */
  const renderToolResultCard = (toolResult, idx) => {
    const { name, result } = toolResult;
    if (!result || result.status === "error") {
      return (
        <div key={idx} className="tool-card tool-card-error">
          <div className="card-error-row">
            <span className="card-error-pill">⚠️ Execution Alert</span>
            <span className="card-error-msg">{result?.message || "Tool execution failed"}</span>
          </div>
        </div>
      );
    }

    // 1. Single Prediction Tool Result Card
    if (name === "predict_congestion" && result.congestion_probability !== undefined) {
      const prob = result.congestion_probability;
      const probPct = (prob * 100).toFixed(1);
      const riskTier = (result.risk_level || "Low").toLowerCase();
      const location = result.area_name
        ? `${result.road_name} • ${result.area_name}`
        : result.road_name;

      return (
        <div key={idx} className={`tool-card prediction-card tier-${riskTier}`}>
          <div className="tool-card-top">
            <div className="risk-header-group">
              <span className={`risk-badge badge-${riskTier}`}>
                <span className="badge-beacon" />
                {result.risk_level.toUpperCase()} RISK
              </span>
              <span className="prob-pill">{probPct}% Saturated</span>
            </div>
            <span className="corridor-tag">Bengaluru Corridor</span>
          </div>

          <div className="card-location">
            <MapPinIcon size={14} className="location-pin-icon" />
            <span>{location}</span>
          </div>

          {/* Precision Saturation Progress Bar */}
          <div className="card-progress-wrapper">
            <div className="card-progress-bar-track">
              <div
                className={`card-progress-bar-fill fill-${riskTier}`}
                style={{ width: `${Math.min(prob * 100, 100)}%` }}
              />
              <div
                className="card-progress-threshold-marker"
                style={{ left: "45%" }}
                title="Model Decision Cutoff: 45%"
              />
            </div>
            <div className="card-progress-labels">
              <span>0% Flow</span>
              <span className="threshold-label">Threshold: 45%</span>
              <span>100% Gridlock</span>
            </div>
          </div>

          <div className="card-stats-grid">
            <div className="stat-item">
              <div className="stat-label">
                <SpeedometerIcon size={13} /> Speed
              </div>
              <span className="stat-val">{result.estimated_speed_kmh} km/h</span>
            </div>
            <div className="stat-item">
              <div className="stat-label">
                <ClockIcon size={13} /> Delay
              </div>
              <span className="stat-val highlight">+{result.estimated_delay_min} min</span>
            </div>
            <div className="stat-item">
              <div className="stat-label">
                {result.inputs_used?.weather_condition === "Rain" ? (
                  <CloudRainIcon size={13} />
                ) : (
                  <SunIcon size={13} />
                )}
                Weather
              </div>
              <span className="stat-val">{result.inputs_used?.weather_condition || "Clear"}</span>
            </div>
            {result.inputs_used?.roadwork && (
              <div className="stat-item roadwork-active">
                <div className="stat-label">
                  <ConeIcon size={13} /> Works
                </div>
                <span className="stat-val warning">Constriction</span>
              </div>
            )}
          </div>

          {result.recommendation && (
            <div className="card-advisory">
              <LightbulbIcon size={16} className="adv-icon" />
              <div className="adv-content">
                <strong>Traffic Advisory:</strong> {result.recommendation}
              </div>
            </div>
          )}
        </div>
      );
    }

    // 2. Live Weather Conditions Telemetry Card
    if (name === "get_live_conditions" && result.status === "success") {
      const isRain = result.weather_condition === "Rain";
      return (
        <div key={idx} className="tool-card weather-card">
          <div className="tool-card-top">
            <div className="live-radar-badge">
              <span className="radar-ping" />
              <SatelliteIcon size={14} />
              <span>Bengaluru Live Weather Telemetry</span>
            </div>
            <span className="timestamp-pill">{result.day_name} • {result.formatted_time}</span>
          </div>

          <div className="weather-stats-row">
            <div className="weather-main-stat">
              <span className="temp-large">{result.temperature_c}°C</span>
              <div className="weather-desc-col">
                <span className="weather-desc">
                  {isRain ? "Monsoon Rainfall" :
                   result.weather_condition === "Overcast" ? "Overcast Skies" :
                   result.weather_condition === "Fog" ? "Dense Fog / Mist" :
                   result.weather_condition === "Windy" ? "High Wind Conditions" : "Clear Skies"}
                </span>
                <span className="weather-coords">12.9716° N, 77.5946° E</span>
              </div>
            </div>

            <div className="weather-telemetry-tiles">
              <div className="telemetry-tile">
                <span className="tile-title">Precipitation</span>
                <span className="tile-value">{result.rain_mm} mm</span>
              </div>
              <div className="telemetry-tile">
                <span className="tile-title">Wind Speed</span>
                <span className="tile-value">{result.wind_speed_kmh} km/h</span>
              </div>
            </div>
          </div>
        </div>
      );
    }

    // 3. Compare Scenarios Card
    if (name === "compare_scenarios" && result.results) {
      const bestIndex = result.results.reduce(
        (best, curr, i, arr) => (curr.congestion_probability < arr[best].congestion_probability ? i : best),
        0
      );

      return (
        <div key={idx} className="tool-card comparison-card">
          <div className="tool-card-top">
            <div className="comp-title-group">
              <ScaleIcon size={15} />
              <span className="comp-title-badge">Side-by-Side Corridor Comparison</span>
            </div>
            <span className="count-pill">{result.scenarios_compared} Corridors</span>
          </div>

          <div className="comp-grid">
            {result.results.map((sc, sIdx) => {
              const scRisk = (sc.risk_level || "low").toLowerCase();
              const isBest = sIdx === bestIndex && result.results.length > 1;
              return (
                <div key={sIdx} className={`comp-tile tile-${scRisk} ${isBest ? "tile-recommended" : ""}`}>
                  {isBest && (
                    <div className="optimal-badge">
                      <SparklesIcon size={12} /> Optimal Route
                    </div>
                  )}
                  <div className="tile-header">
                    <strong className="tile-name">{sc.road_name}</strong>
                    <span className={`tile-tag tag-${scRisk}`}>{sc.risk_level}</span>
                  </div>
                  <div className="tile-meta">
                    <span>{sc.area_name} • {sc.weather}</span>
                  </div>
                  <div className="tile-metrics">
                    <div className="metric-pill">
                      <strong>{(sc.congestion_probability * 100).toFixed(1)}%</strong> Risk
                    </div>
                    <div className="metric-pill">
                      <strong>+{sc.estimated_delay_min}m</strong> Delay
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {result.recommendation && (
            <div className="comp-conclusion">
              <LightbulbIcon size={15} />
              <span><strong>Recommendation:</strong> {result.recommendation}</span>
            </div>
          )}
        </div>
      );
    }

    // 4. Rank Roads by Risk Card
    if (name === "rank_roads_by_risk" && result.ranked_roads) {
      const allRoads = result.ranked_roads;
      const top3 = allRoads.slice(0, 3);
      const bottom3 = allRoads.slice(-3).reverse();

      return (
        <div key={idx} className="tool-card ranking-card">
          <div className="tool-card-top">
            <div className="ranking-title-group">
              <MapPinIcon size={14} />
              <span className="ranking-title-badge">16 Monitored Corridors Ranked</span>
            </div>
            <span className="count-pill">Bengaluru Network</span>
          </div>

          <div className="ranking-split-view">
            {/* Highest Risk Group */}
            <div className="rank-column">
              <span className="column-header red">
                <ArrowUpRightIcon size={13} /> Highest Congestion Risk:
              </span>
              <div className="rank-items">
                {top3.map((r, rIdx) => (
                  <div key={rIdx} className="rank-row">
                    <span className="rank-num">#{rIdx + 1}</span>
                    <span className="rank-name">{r.road_name}</span>
                    <span className="rank-val red">{(r.congestion_probability * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Lowest Risk Group */}
            <div className="rank-column">
              <span className="column-header green">
                <ArrowDownRightIcon size={13} /> Safest Recommended Roads:
              </span>
              <div className="rank-items">
                {bottom3.map((r, rIdx) => (
                  <div key={rIdx} className="rank-row">
                    <span className="rank-num green">#{16 - rIdx}</span>
                    <span className="rank-name">{r.road_name}</span>
                    <span className="rank-val green">{(r.congestion_probability * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Interactive Accordion to View All 16 Corridors */}
          <div className="ranking-accordion-wrapper">
            <button
              type="button"
              className="ranking-toggle-btn"
              onClick={() => setExpandedRankings((prev) => !prev)}
              aria-expanded={expandedRankings}
            >
              <span>{expandedRankings ? "Collapse 16 Corridors" : "View All 16 Corridors"}</span>
              <ChevronDownIcon
                size={14}
                className={`accordion-chevron ${expandedRankings ? "chevron-open" : ""}`}
              />
            </button>

            {expandedRankings && (
              <div className="all-roads-grid">
                {allRoads.map((road, rIdx) => {
                  const prob = road.congestion_probability;
                  const tier = prob >= 0.7 ? "high" : prob >= 0.45 ? "medium" : "low";
                  return (
                    <div key={rIdx} className={`all-roads-row tier-${tier}`}>
                      <span className="all-roads-rank">#{rIdx + 1}</span>
                      <span className="all-roads-name">{road.road_name}</span>
                      <span className="all-roads-area">{road.area_name}</span>
                      <span className={`all-roads-prob ${tier}`}>{(prob * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      );
    }

    // 5. Explain Model Factors Card
    if (name === "explain_model_factors" && result.top_congested_factors) {
      return (
        <div key={idx} className="tool-card factors-card">
          <div className="tool-card-top">
            <div className="factors-title-group">
              <SparklesIcon size={14} />
              <span className="factors-title-badge">Mathematical Odds Ratios (OR = exp(β))</span>
            </div>
            <span className="count-pill">Logistic Regression</span>
          </div>

          <div className="factors-group-wrapper">
            <div className="factor-subgroup">
              <span className="factor-subtitle red">
                <ArrowUpRightIcon size={12} /> Congestion Catalysts (Increased Odds):
              </span>
              <div className="factors-chips-row">
                {result.top_congested_factors.slice(0, 4).map((f, fIdx) => (
                  <span key={fIdx} className="factor-chip high">
                    <strong>{f.feature.replace("road_", "").replace("weather_", "")}</strong>:{" "}
                    +{f.odds_ratio}x odds
                  </span>
                ))}
              </div>
            </div>

            {result.top_protective_factors && (
              <div className="factor-subgroup">
                <span className="factor-subtitle green">
                  <ArrowDownRightIcon size={12} /> Protective Buffers (Reduced Odds):
                </span>
                <div className="factors-chips-row">
                  {result.top_protective_factors.slice(0, 3).map((f, fIdx) => (
                    <span key={fIdx} className="factor-chip low">
                      <strong>{f.feature.replace("road_", "").replace("weather_", "")}</strong>:{" "}
                      {f.odds_ratio}x odds
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      );
    }

    return null;
  };

  return (
    <div className={`chat-interface ${isWide ? "drawer-wide" : ""}`}>
      {/* 1. Mobile Grab Handle for Touch Accessibility */}
      <div className="chat-mobile-handle" aria-hidden="true">
        <span className="handle-bar" />
      </div>

      {/* 2. Unified Clean Header */}
      <div className="chat-top-header">
        <div className="chat-header-left">
          <div className="copilot-avatar-wrapper" aria-hidden="true">
            <CopilotAvatarIcon size={26} />
            <span className="avatar-pulse-ring" />
          </div>
          <div className="copilot-meta">
            <div className="copilot-name-row">
              <h3 className="copilot-title">Bengaluru AI Traffic Copilot</h3>
              <span className="copilot-live-pill">
                <span className="live-dot" /> Live ML
              </span>
            </div>
            <span className="copilot-subtext">
              16 Arterial Corridors • Calibrated Logistic Regression (t = 0.45)
            </span>
          </div>
        </div>

        <div className="chat-header-actions">
          {onToggleWide && (
            <button
              type="button"
              className="chat-action-btn btn-wide-toggle"
              onClick={onToggleWide}
              title={isWide ? "Restore standard width" : "Expand to wide view"}
              aria-label={isWide ? "Restore standard width" : "Expand to wide view"}
            >
              {isWide ? <CollapseIcon size={16} /> : <ExpandIcon size={16} />}
            </button>
          )}

          <button
            type="button"
            className="chat-action-btn btn-clear"
            onClick={handleClear}
            disabled={isStreaming}
            title="Clear conversation history"
            aria-label="Clear conversation history"
          >
            <TrashIcon size={14} />
            <span className="action-btn-text">Clear</span>
          </button>

          {onClose && (
            <button
              type="button"
              className="chat-action-btn btn-close"
              onClick={onClose}
              aria-label="Close Assistant Drawer"
              title="Close Drawer"
            >
              <CloseIcon size={16} />
            </button>
          )}
        </div>
      </div>

      {/* 3. Quick Suggestions Prompt Chips Carousel */}
      {showPromptChips && (
        <div className="chat-quick-chips-wrapper" role="group" aria-label="Suggested questions">
          <div className="chips-header-row">
            <span className="chips-hint">
              <SparklesIcon size={12} /> Suggested Operations Queries:
            </span>
            <button
              type="button"
              className="chips-dismiss-btn"
              onClick={() => setShowPromptChips(false)}
              title="Hide suggested queries"
            >
              Hide
            </button>
          </div>
          <div className="chat-quick-chips">
            {QUICK_ACTIONS.map((action) => (
              <button
                key={action.id}
                type="button"
                className="quick-chip-button"
                onClick={() => handleSend(action.prompt)}
                disabled={isStreaming}
              >
                <span className="chip-icon">{action.icon}</span>
                <span className="chip-text">{action.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 4. Messages Stream Feed */}
      <div className="chat-stream-feed" ref={feedRef} role="log" aria-live="polite">
        {messages.map((msg, index) => {
          const isUser = msg.role === "user";
          return (
            <div
              key={index}
              className={`message-bubble-wrapper ${isUser ? "from-user" : "from-assistant"}`}
            >
              <div className="bubble-avatar-tag" aria-hidden="true">
                {isUser ? <UserAvatarIcon size={16} /> : <CopilotAvatarIcon size={18} />}
              </div>

              <div className="bubble-body-container">
                <div className="bubble-meta-row">
                  <span className="bubble-sender-name">
                    {isUser ? "You" : "Traffic Operations Copilot"}
                  </span>
                  {msg.timestamp && (
                    <span className="bubble-timestamp">{msg.timestamp}</span>
                  )}
                  {!isUser && msg.content && (
                    <button
                      type="button"
                      className="btn-copy-bubble"
                      onClick={() => handleCopyMessage(msg.content, index)}
                      title="Copy response to clipboard"
                      aria-label="Copy response to clipboard"
                    >
                      {copiedIndex === index ? (
                        <>
                          <CheckIcon size={13} />
                          <span className="copy-feedback">Copied</span>
                        </>
                      ) : (
                        <CopyIcon size={13} />
                      )}
                    </button>
                  )}
                </div>

                <div className={`message-bubble ${isUser ? "user-bubble" : "assistant-bubble"}`}>
                  {msg.content ? (
                    <FormattedMessage content={msg.content} />
                  ) : isStreaming && index === messages.length - 1 ? (
                    <div className="streaming-dots">
                      <span className="dot" />
                      <span className="dot" />
                      <span className="dot" />
                    </div>
                  ) : null}
                </div>

                {/* Render interactive tool cards */}
                {msg.toolResults && msg.toolResults.length > 0 && (
                  <div className="bubble-tool-results-list">
                    {msg.toolResults.map((tr, trIdx) => renderToolResultCard(tr, trIdx))}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Live Status indicator when model is analyzing */}
        {statusMessage && (
          <div className="chat-live-status-pill" role="status">
            <span className="pulse-spinner" aria-hidden="true" />
            <span className="status-text">{statusMessage}</span>
          </div>
        )}

        {/* Error notification */}
        {error && (
          <div className="chat-error-card" role="alert">
            <span className="error-icon" aria-hidden="true">⚠️</span>
            <span className="error-text">{error}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* 5. Chat Input Bar */}
      <div className="chat-input-container">
        {!showPromptChips && (
          <button
            type="button"
            className="show-chips-pill-btn"
            onClick={() => setShowPromptChips(true)}
          >
            <SparklesIcon size={12} /> Show Suggested Queries
          </button>
        )}

        <form
          className="chat-input-form"
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
        >
          <input
            ref={inputRef}
            type="text"
            className="chat-unified-input"
            placeholder="Ask about Koramangala, Hebbal, Marathahalli, live weather, or safest day..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isStreaming}
            aria-label="Message to Bengaluru AI Traffic Copilot"
          />

          {isStreaming ? (
            <button
              type="button"
              className="chat-stop-btn"
              onClick={handleStop}
              aria-label="Stop generating response"
              title="Stop response"
            >
              <StopIcon size={14} />
              <span className="stop-text">Stop</span>
            </button>
          ) : (
            <button
              type="submit"
              className="chat-send-submit-btn"
              disabled={!input.trim()}
              aria-label="Send query"
              title="Send query"
            >
              <SendIcon size={16} />
            </button>
          )}
        </form>

        <div className="chat-input-footnote">
          <span>
            🔒 Grounded in Bengaluru empirical Logistic Regression (t = 0.45). Observational data, not causation.
          </span>
        </div>
      </div>
    </div>
  );
}
