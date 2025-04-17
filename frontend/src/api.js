// api.js
// Utility functions for interacting with Flask backend

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export async function predictSingle(data) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return res.json();
}

export async function predictBatch(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData
  });
  return res.json();
}

export async function getHistory() {
  const res = await fetch(`${API_BASE}/history`);
  return res.json();
}

export async function explainPrediction(input, method = 'shap') {
  const res = await fetch(`${API_BASE}/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input, method })
  });
  return res.json();
}
