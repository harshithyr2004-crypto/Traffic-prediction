/**
 * API client for Bengaluru Road Traffic Congestion Prediction backend.
 * Provides resilient fetch wrappers for health, prediction, options, and scenarios.
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

/**
 * Perform a fetch request with timeout.
 */
async function fetchWithTimeout(endpoint, options = {}, timeoutMs = 8000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
    clearTimeout(id);
    return response;
  } catch (err) {
    clearTimeout(id);
    throw err;
  }
}

/**
 * Check backend health status (polled every 30s).
 */
export async function checkHealth() {
  try {
    const res = await fetchWithTimeout("/health", { method: "GET" }, 5000);
    if (!res.ok) return { status: "offline", code: res.status };
    const data = await res.json();
    return { status: "online", data };
  } catch (err) {
    return { status: "offline", error: err.message };
  }
}

/**
 * Fetch dynamic dropdown options, area-road mappings, threshold, and metrics from bundle.
 */
export async function getOptions() {
  const res = await fetchWithTimeout("/options", { method: "GET" }, 6000);
  if (!res.ok) throw new Error("Failed to load options from backend");
  return await res.json();
}

/**
 * Fetch predefined Bengaluru scenario presets.
 */
export async function getScenarios() {
  const res = await fetchWithTimeout("/scenarios", { method: "GET" }, 6000);
  if (!res.ok) throw new Error("Failed to load scenario presets");
  return await res.json();
}

/**
 * Submit traffic conditions to /predict endpoint.
 */
export async function getPrediction(payload) {
  const res = await fetchWithTimeout(
    "/predict",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
    10000
  );

  if (!res.ok) {
    const errText = await res.text().catch(() => res.statusText);
    throw new Error(`Server returned ${res.status}: ${errText}`);
  }

  return await res.json();
}

/**
 * Fetch ranked road list by risk level.
 */
export async function getRankedRoads(params = {}) {
  const query = new URLSearchParams(params).toString();
  const endpoint = query ? `/roads/rank?${query}` : "/roads/rank";
  const res = await fetchWithTimeout(endpoint, { method: "GET" }, 6000);
  if (!res.ok) throw new Error("Failed to rank roads");
  return await res.json();
}
