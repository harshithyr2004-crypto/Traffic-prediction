/**
 * RiskGauge Component
 * A precision semi-circular SVG gauge displaying traffic congestion risk probability,
 * calibrated color zones (Low <40%, Medium 40-70%, High >=70%), and decision threshold marker.
 */

export default function RiskGauge({
  probability = 0,
  riskLevel = "Low",
  threshold = 0.50,
}) {
  // Normalize probability to a 0.0 - 1.0 float and 0 - 100 integer
  const rawProb = typeof probability === "number" ? probability : parseFloat(probability) || 0;
  const normalizedProb = rawProb > 1 ? rawProb / 100 : rawProb;
  const clampedProb = Math.min(Math.max(normalizedProb, 0), 1);
  const percentDisplay = (clampedProb * 100).toFixed(1);

  // Semi-circle arc parameters:
  // Radius R = 72, Arc length = PI * 72 ≈ 226.19
  const radius = 72;
  const arcLength = Math.PI * radius;
  const strokeOffset = arcLength * (1 - clampedProb);

  // Dynamic theme colors based on risk tier
  let strokeColor = "var(--color-risk-low)";
  let badgeClass = "risk-badge-low";
  let tierIcon = "🟢";

  if (clampedProb >= 0.70) {
    strokeColor = "var(--color-risk-high)";
    badgeClass = "risk-badge-high";
    tierIcon = "🔴";
  } else if (clampedProb >= 0.40) {
    strokeColor = "var(--color-risk-med)";
    badgeClass = "risk-badge-med";
    tierIcon = "🟡";
  }

  // Calculate threshold marker position on the arc (threshold is usually 0.40 or 0.50)
  // Angle runs from 180 deg (left) to 0 deg (right)
  const thresholdAngleRad = Math.PI * (1 - threshold);
  const markerX = 100 + radius * Math.cos(thresholdAngleRad);
  const markerY = 88 - radius * Math.sin(thresholdAngleRad);

  return (
    <div
      className="risk-gauge-container"
      role="meter"
      aria-label="Traffic Congestion Probability"
      aria-valuenow={percentDisplay}
      aria-valuemin="0"
      aria-valuemax="100"
    >
      <div className="gauge-svg-wrapper">
        <svg
          viewBox="0 0 200 115"
          className="gauge-svg"
          aria-hidden="true"
        >
          <defs>
            {/* Soft gradient along gauge path */}
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#22c55e" />
              <stop offset="50%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
          </defs>

          {/* Background meter track */}
          <path
            d="M 28 92 A 72 72 0 0 1 172 92"
            fill="none"
            stroke="var(--bg-subtle)"
            strokeWidth="14"
            strokeLinecap="round"
          />

          {/* Active progress arc */}
          <path
            d="M 28 92 A 72 72 0 0 1 172 92"
            fill="none"
            stroke={strokeColor}
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={arcLength}
            strokeDashoffset={strokeOffset}
            className="gauge-progress-arc"
          />

          {/* Threshold anchor dot */}
          <circle
            cx={markerX}
            cy={markerY}
            r="3.5"
            fill="var(--text-primary)"
            stroke="var(--bg-card)"
            strokeWidth="1.5"
          />
        </svg>

        {/* Center Readout Overlay */}
        <div className="gauge-readout">
          <span className="gauge-percentage">{percentDisplay}%</span>
          <span className="gauge-sublabel">Congestion Probability</span>
        </div>
      </div>

      {/* Risk Badge and Threshold Context */}
      <div className="gauge-footer-meta">
        <span className={`gauge-risk-badge ${badgeClass}`}>
          <span className="badge-bullet">{tierIcon}</span>
          <span className="badge-text">{riskLevel.toUpperCase()} RISK</span>
        </span>
        <span className="gauge-threshold-note">
          Model Cutoff: <strong>{(threshold * 100).toFixed(0)}%</strong>
        </span>
      </div>
    </div>
  );
}
