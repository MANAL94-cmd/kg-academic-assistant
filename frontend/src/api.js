// Thin wrapper around the Flask backend's REST API.
// In dev, Vite proxies these /api paths to http://localhost:5000 (see vite.config.js).
// In production, the Flask app serves this bundle, so same-origin requests just work.

const API_BASE = "";

async function asJson(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.error) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

export function getStatus() {
  return fetch(`${API_BASE}/api/status`).then(asJson);
}

export function getGraph() {
  return fetch(`${API_BASE}/api/graph`).then(asJson);
}

export function ask({ question, apiKey, mode }) {
  return fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, api_key: apiKey, mode }),
  }).then(asJson);
}

export function uploadPaper(file) {
  const formData = new FormData();
  formData.append("file", file);
  return fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  }).then(asJson);
}
