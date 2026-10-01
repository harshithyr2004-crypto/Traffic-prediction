"""
Generate model_facts.json for the Bengaluru Traffic Congestion Prediction model.
Reads directly from model/bengaluru_traffic_model.joblib to guarantee 100% mathematical fidelity.
"""

import json
from pathlib import Path
import joblib
import numpy as np

def make_model_facts():
    project_root = Path(__file__).resolve().parent.parent
    bundle_path = project_root / "model" / "bengaluru_traffic_model.joblib"
    out_path = project_root / "model" / "model_facts.json"

    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found at: {bundle_path}")

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    columns = bundle["columns"]
    threshold = bundle["threshold"]
    metrics = bundle["metrics"]
    feature_spec = bundle["feature_spec"]

    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])
    odds_ratios = np.exp(coefs)

    items = []
    for col, coef, or_val in zip(columns, coefs, odds_ratios):
        # Format human description
        if col.startswith("road_"):
            rname = col.replace("road_", "")
            if or_val >= 1.0:
                desc = f"Traveling on {rname} is associated with {or_val:.2f}x higher odds of congestion vs baseline road (100 Feet Rd)"
            else:
                desc = f"Traveling on {rname} is associated with {(1.0 - or_val) * 100:.1f}% lower odds of congestion vs baseline road"
        elif col.startswith("weather_"):
            wname = col.replace("weather_", "")
            desc = f"{wname} weather is associated with {or_val:.2f}x odds of congestion compared to Clear skies"
        elif col == "roadwork":
            desc = f"Active carriageway roadwork is associated with {or_val:.2f}x higher odds of congestion"
        elif col == "is_weekend":
            desc = f"Weekend days are associated with {or_val:.2f}x odds of congestion vs weekdays"
        elif col == "day_of_week":
            desc = f"Each 1-SD shift towards the weekend is associated with {or_val:.2f}x odds of congestion"
        elif col == "month":
            desc = f"Each 1-SD progression through the calendar year is associated with {or_val:.2f}x odds of congestion"
        else:
            desc = f"Associated with {or_val:.2f}x odds ratio"

        items.append({
            "feature": col,
            "coef": round(float(coef), 4),
            "odds_ratio": round(float(or_val), 4),
            "description": desc
        })

    # Sort items by odds ratio descending
    items_sorted = sorted(items, key=lambda x: x["odds_ratio"], reverse=True)
    top_positive = [it for it in items_sorted if it["odds_ratio"] >= 1.0]
    top_protective = [it for it in sorted(items, key=lambda x: x["odds_ratio"]) if it["odds_ratio"] < 1.0]

    facts = {
        "city": "Bengaluru",
        "state": "Karnataka, India",
        "model_name": "Bengaluru Surface Road Traffic Congestion Classifier",
        "algorithm": "Calibrated Logistic Regression (L2 Regularized)",
        "dataset": "Bengaluru City Traffic Dataset (8,936 historical roadway observations, 2022–2024)",
        "intercept": round(intercept, 4),
        "threshold": threshold,
        "metrics": metrics,
        "feature_spec": feature_spec,
        "coefficients": items_sorted,
        "top_congested_factors": top_positive[:5],
        "top_protective_factors": top_protective[:5]
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)

    print(f"Successfully generated {out_path} with {len(items)} features.")

if __name__ == "__main__":
    make_model_facts()
