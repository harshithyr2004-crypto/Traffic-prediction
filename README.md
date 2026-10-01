# Bengaluru Road Traffic Congestion Prediction & Real-Time AI Chatbot

An end-to-end Machine Learning operations platform and conversational AI assistant for **Surface Road Traffic Congestion Prediction in Bengaluru (Bangalore), Karnataka, India**, powered by calibrated Logistic Regression on the empirical **Bengaluru City Traffic Dataset** and extended with a real-time AI Operations Copilot using the official OpenAI Python SDK connected to **Groq**'s high-speed inference endpoint.

---

## Project Overview

- **Location:** Bengaluru (Bangalore), Karnataka, India
- **Core Problem:** Proactive prediction of severe road traffic congestion across 16 major arterial corridors and intersections before gridlock occurs.
- **Vehicular Focus:** Surface road motor vehicles (two-wheelers, auto-rickshaws, private cars, cabs, transit buses, and commercial transport trucks).
- **Target Formulation:** Binary road congestion classification ($0 = \text{Normal Flow}, 1 = \text{High Congestion / Capacity Saturation}$), defined as reaching $100\%$ capacity saturation (75th percentile / top quartile rule).
- **Algorithm:** Regularized Logistic Regression ($L_2$ Ridge penalty) with decision threshold tuned to $t = 0.45$.
- **Validation Metrics:**
  - **Accuracy:** $71.42\%$
  - **Recall (Sensitivity):** $76.45\%$ (captures over 3 out of 4 true congestion events)
  - **Precision:** $64.50\%$
  - **$F_1$-Score:** $0.700$
  - **ROC-AUC:** $0.7795$

---

## System Architecture

```
[ Bengaluru City Traffic Dataset ]
8,936 historical roadway observations across 16 corridors (2022–2024)
                        │
                        ▼
[ Jupyter Data Science Notebook ]
notebooks/bengaluru_traffic_congestion.ipynb
- Data cleaning, date deconstruction (day of week, month, weekend)
- Outlier verification & multi-plot EDA
- Target formulation (Congestion Level >= 100%, 43.5% positive balance)
- Target leakage audit: Group (a) features only
- Multicollinearity resolution: Road implies Area
- Variance Inflation Factor (VIF) evaluation
- Stratified 80/20 train/test split
- StandardScaler (train-only fit) + pd.get_dummies(drop_first=True)
- Logistic Regression training, ROC-AUC, PR curve
- Odds Ratios calculation: OR = exp(beta)
- Threshold calibration (t = 0.45) & error analysis
                        │
                        ▼
[ Serialized Model Artifacts ]
model/bengaluru_traffic_model.joblib (Model, Scaler, Columns, Feature Spec, Threshold = 0.45)
model/model_facts.json (Odds Ratios, Learned Coefficients, Top Factors)
                        │
                        ▼
[ FastAPI Backend Engine (Port 8000) ]
backend/main.py       ──► GET /health, GET /options, GET /scenarios, POST /predict, POST /chat
backend/predictor.py  ──► Centralized road feature engineering & single-source inference
backend/tools.py      ──► 5 OpenAI Function Calling Tools:
                          1. predict_congestion(...) (Single corridor prediction)
                          2. compare_scenarios(...) (Side-by-side corridor comparison)
                          3. rank_roads_by_risk(...) (All 16 corridors ranked)
                          4. get_live_conditions() (Open-Meteo for Bengaluru: 12.9716°N, 77.5946°E)
                          5. explain_model_factors() (Mathematical odds ratios from model_facts.json)
backend/chat.py       ──► Multi-turn Groq tool execution loop streaming SSE tokens to UI
backend/prompts.py    ──► Observational wording guardrails ("associated with", never "caused")
                        │
                        ▼
[ React 18 + Vite Operations Dashboard (Port 5173) ]
Header.jsx            ──► Bengaluru title, 30s Live API health polling badge, Light/Dark theme toggle
PredictionForm.jsx    ──► Area selector filtering Road dropdown, Day-of-week segmented control,
                          Weather picker with emojis, Active roadwork toggle, Preset chips
RiskGauge.jsx         ──► Pure SVG semi-circular meter with calibrated 0.45 threshold dot
ResultCard.jsx        ──► Probability %, Speed (km/h), Queue delay (+min), Advisory box
RecentPredictions.jsx ──► 5-run history with 1-click reload
Chat.jsx              ──► Off-canvas slide-in drawer with Quick Chips, Tool Cards, and SSE stream
Footer.jsx            ──► Model facts, validation metrics, and operational limitations
```

---

## 16 Monitored Corridors Across 8 Urban Zones

| Urban Zone (Area) | Primary Arterial Corridors & Intersections |
|---|---|
| **Koramangala** | Sony World Junction, Sarjapur Road |
| **Indiranagar** | 100 Feet Road, CMH Road |
| **Whitefield** | Marathahalli Bridge, ITPL Main Road |
| **Hebbal** | Hebbal Flyover, Ballari Road |
| **M.G. Road** | Trinity Circle, Anil Kumble Circle |
| **Jayanagar** | Jayanagar 4th Block, South End Circle |
| **Yeshwanthpur** | Yeshwanthpur Circle, Tumkur Road |
| **Electronic City** | Silk Board Junction, Hosur Road |

---

## Key Model Odds Ratios ($\text{OR} = e^\beta$)

Extracted directly from `model/model_facts.json`:

| Feature | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Practical Meaning for Commuters |
|---|---|---|---|
| `road_Sony World Junction` | $+0.9417$ | **$2.56\times$** | Traveling on Sony World Junction is associated with $2.56\times$ higher odds of congestion vs baseline (100 Feet Rd) |
| `road_Sarjapur Road` | $+0.8017$ | **$2.23\times$** | Sarjapur Road corresponds to $2.23\times$ higher odds of congestion |
| `road_Trinity Circle` | $+0.3627$ | **$1.44\times$** | Elevated congestion odds along M.G. Road commercial node |
| `road_Anil Kumble Circle`| $+0.2913$ | **$1.34\times$** | High congestion odds near CBD / Cubbon Park |
| `weather_Windy` | $+0.1706$ | **$1.19\times$** | High winds correspond to $1.19\times$ odds of traffic disruption |
| `roadwork` | $+0.0967$ | **$1.10\times$** | Active carriageway construction constricting lanes increases congestion odds |
| `road_Tumkur Road` | $-2.6286$ | **$0.07\times$** | Significantly lower baseline congestion odds ($93\%$ lower) |
| `road_Silk Board Junction`| $-3.2821$ | **$0.04\times$** | Lower odds of hitting 100% capacity cap in this historical sample |

---

## Environment Variables

### In `backend/.env`:

| Variable | Meaning | Example / Recommended Value | Where to Get It |
|---|---|---|---|
| `LLM_BASE_URL` | Base URL for OpenAI-compatible endpoint | `https://api.groq.com/openai/v1` | Pre-configured for Groq; can point to Ollama |
| `LLM_API_KEY` | High-speed inference API key | `gsk_...` | Free from [Groq Console](https://console.groq.com/keys) |
| `LLM_MODEL` | Active model identifier supporting function calling | `openai/gpt-oss-120b` | [Groq Models](https://console.groq.com/docs/models) |
| `GROQ_API_KEY` | Optional fallback alias for Groq | *(Same as above)* | [Groq Console](https://console.groq.com/keys) |

### In `frontend/.env`:

| Variable | Meaning | Default Value |
|---|---|---|
| `VITE_API_URL` | Base URL of running FastAPI backend | `http://127.0.0.1:8000` |

---

## How to Run Locally

### Prerequisites
- Python 3.12+ with `uv` package manager installed (`pip install uv` or `winget install astral-sh.uv`).
- Node.js (v18+) and `npm`.

### Step 1: Start Backend (FastAPI)
Run from the project root: `D:\programs\Projects\traffic-congestion-prediction`
```powershell
uv run uvicorn backend.main:app --reload --port 8000
```
- Interactive API Documentation: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`
- Dynamic Options: `http://127.0.0.1:8000/options`

### Step 2: Start Frontend (React + Vite)
In a second PowerShell terminal:
```powershell
cd frontend
npm run dev
```
- Open in browser: `http://localhost:5173`

### Step 3: Run Automated Verification Suite
From the project root:
```powershell
uv run python tests/test_tools.py
```
Expected output:
```
Running Bengaluru Road Traffic Test Suite...
[PASS] Test 1: Shared predictor matches /predict endpoint.
[PASS] Test 2: /options matches the model bundle.
[PASS] Test 3: rank_roads_by_risk returns all 16 roads sorted by risk.
[PASS] Test 4: Live-conditions failure handled cleanly without crashing.
[PASS] Test 5: Bad tool arguments and edge cases handled safely.
[PASS] Test 6: Model facts and odds ratios verified.

ALL 6 TEST SUITES PASSED SUCCESSFULLY!
```

---

## 5 Example AI Chatbot Questions to Demo

1. **Live Ambient Telemetry:**  
   *"What is the traffic congestion risk right now in Bengaluru?"*  
   *(The AI invokes `get_live_conditions` to fetch current Bengaluru weather and date, pipes them into `predict_congestion`, and returns live risk probability).*

2. **Corridor Comparison:**  
   *"Compare congestion risk between Sony World Junction (Koramangala) and Tumkur Road (Yeshwanthpur)."*  
   *(The AI invokes `compare_scenarios` and compares odds ratios and probabilities side-by-side).*

3. **Model Explainability & Odds Ratios:**  
   *"Which factors increase or decrease traffic congestion risk the most according to the Bengaluru model?"*  
   *(The AI invokes `explain_model_factors` to quote top positive and protective odds ratios directly from `model_facts.json`).*

4. **Optimal Travel Day Guidance:**  
   *"What is the safest day this week to travel on Sarjapur Road to avoid congestion?"*  
   *(The AI evaluates days of the week, cites the lower risk probabilities, and advises the optimal departure day).*

5. **City-Wide Corridor Ranking:**  
   *"Rank all 16 monitored roads in Bengaluru by their congestion risk under normal clear conditions."*  
   *(The AI invokes `rank_roads_by_risk` to sort all 16 roads from highest to lowest congestion risk).*

---

## Known Scientific Limitations

1. **Aggregated Daily Telemetry:** The dataset records daily aggregated corridor metrics across 8 areas and 16 roads over 2022–2024. It does not record minute-by-minute signal cycle phases.
2. **Selected Spatial Scope:** Focuses on 16 primary arterial bottlenecks; neighborhood interior streets are not modeled.
3. **Observational Correlation:** Features represent empirical statistical associations with road congestion, not direct physical causation.

---

## Academic Defense & Viva Voce Documentation
For 25 in-depth viva questions with detailed answers, refer to [docs/viva_questions.md](file:///d:/programs/Projects/traffic-congestion-prediction/docs/viva_questions.md).
