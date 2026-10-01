import React from "react";

/**
 * Footer Component
 * Grounds the application in factual ML model specifications and dataset boundaries.
 * Essential for academic defense and scientific transparency.
 */
export default function Footer({ optionsData }) {
  const threshold = optionsData?.threshold ?? 0.45;
  const metrics = optionsData?.metrics;

  return (
    <footer className="app-footer">
      <div className="footer-content">
        <div className="footer-model-specs">
          <span className="spec-item">
            <strong>Inference Engine:</strong> Calibrated Logistic Regression (L2 Regularized)
          </span>
          <span className="spec-separator">•</span>
          <span className="spec-item">
            <strong>City:</strong> Bengaluru, Karnataka, India
          </span>
          <span className="spec-separator">•</span>
          <span className="spec-item">
            <strong>Decision Threshold:</strong> t = {threshold.toFixed(2)}
          </span>
          <span className="spec-separator">•</span>
          <span className="spec-item">
            <strong>Corridors:</strong> 16 Arterial Roads across 8 Urban Zones
          </span>
          {metrics && (
            <>
              <span className="spec-separator">•</span>
              <span className="spec-item">
                <strong>Validation:</strong> {(metrics.accuracy * 100).toFixed(1)}% Acc | {(metrics.recall * 100).toFixed(1)}% Recall | {(metrics.precision * 100).toFixed(1)}% Precision | {metrics.roc_auc.toFixed(4)} ROC-AUC
              </span>
            </>
          )}
        </div>

        <div className="footer-limitations">
          <p>
            <strong>Operational Boundaries:</strong> Congestion probabilities and queue delay estimates are calculated from empirical roadway patterns in the Kaggle Bengaluru City Traffic Dataset (8,936 observations, 2022–2024). Variables represent observational correlations with road saturation, not direct physical causation.
          </p>
        </div>
      </div>
    </footer>
  );
}
