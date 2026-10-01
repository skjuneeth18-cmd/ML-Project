// =========================================================
// AQUIFY — WATER QUALITY CLASSIFICATION WEBSITE
// Frontend logic (no framework, no build step)
//
// Talks to the Flask API in ../backend/app.py. If this page
// is opened directly as a file (double-clicked, not served
// by Flask), API_BASE falls back to http://localhost:5000
// so it still works as long as the backend is running.
// =========================================================

const API_BASE = location.protocol === "file:" ? "http://localhost:5000" : "";

let FEATURES = [];       // list of the 20 parameter names
let FEATURE_INFO = {};   // { name: { unit, safe_max } }

const form = document.getElementById("predict-form");
const resultPanel = document.getElementById("result");
const predictBtn = document.getElementById("predict-btn");

// ---------------------------------------------------------
// Small fetch() wrapper that throws a readable error message
// on non-2xx responses
// ---------------------------------------------------------

async function api(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Request failed (${res.status})`);
  }
  return res.json();
}

function fmt(n, digits = 3) {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function timeAgo(iso) {
  const d = new Date(iso.replace(" ", "T") + "Z");
  const diffMs = Date.now() - d.getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return d.toLocaleDateString();
}

// ---------------------------------------------------------
// Build the 20-field prediction form from GET /api/features
// ---------------------------------------------------------

function buildForm() {
  form.innerHTML = "";
  FEATURES.forEach((name) => {
    const info = FEATURE_INFO[name] || {};
    const field = document.createElement("div");
    field.className = "param-field";
    field.innerHTML = `
      <label for="f-${name}">${name}<span class="unit">${info.unit || ""}</span></label>
      <input type="number" step="any" id="f-${name}" name="${name}" placeholder="0.000" required>
    `;
    form.appendChild(field);
  });
}

function fillForm(values) {
  FEATURES.forEach((name) => {
    const input = document.getElementById(`f-${name}`);
    if (input && values[name] !== undefined) input.value = values[name];
  });
}

function readForm() {
  const values = {};
  FEATURES.forEach((name) => {
    const input = document.getElementById(`f-${name}`);
    values[name] = parseFloat(input.value);
  });
  return values;
}

// ---------------------------------------------------------
// Handle form submit -> POST /api/predict
// ---------------------------------------------------------

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const values = readForm();

  predictBtn.disabled = true;
  predictBtn.textContent = "Classifying…";

  try {
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    renderResult(result);
    loadHistory(); // refresh the history table with the new row
  } catch (err) {
    renderError(err.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Classify sample";
  }
});

document.getElementById("reset-btn").addEventListener("click", () => {
  form.reset();
  resultPanel.hidden = true;
});

document.getElementById("fill-sample").addEventListener("click", async () => {
  try {
    const [sample] = await api("/api/samples?n=1");
    fillForm(sample);
  } catch (err) {
    console.error(err);
  }
});

function renderResult(result) {
  const isSafe = result.final_prediction === 1;
  resultPanel.hidden = false;
  resultPanel.className = `result-panel ${isSafe ? "is-safe" : "is-unsafe"}`;
  resultPanel.innerHTML = `
    <div class="result-headline">
      <span>Overall call:</span>
      <span class="verdict ${isSafe ? "safe" : "unsafe"}">${result.final_label}</span>
    </div>
    <div class="result-models">
      <div class="model-card">
        <div class="model-name">Support Vector Machine</div>
        <div class="model-verdict">${result.svm.label}</div>
        <div class="model-conf">${result.svm.confidence}% confidence</div>
      </div>
      <div class="model-card">
        <div class="model-name">Random Forest</div>
        <div class="model-verdict">${result.random_forest.label}</div>
        <div class="model-conf">${result.random_forest.confidence}% confidence</div>
      </div>
    </div>
    <p class="result-note">
      ${result.agreement
        ? "Both models agree on this classification."
        : "The two models disagreed — the site defaults to the Random Forest result, which scored a slightly higher F1 on the test set."}
      This is a statistical estimate, not a certified water-safety test.
    </p>
  `;
  resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderError(message) {
  resultPanel.hidden = false;
  resultPanel.className = "result-panel is-unsafe";
  resultPanel.innerHTML = `<p class="result-note">Could not get a prediction: ${message}</p>`;
}

// ---------------------------------------------------------
// Hero stats + dataset dashboard  (GET /api/stats)
// ---------------------------------------------------------

async function loadStats() {
  try {
    const data = await api("/api/stats");
    const { total_samples, safe_count } = data.dataset;

    document.getElementById("stat-total").textContent = total_samples.toLocaleString();
    document.getElementById("stat-safe").textContent =
      `${Math.round((safe_count / total_samples) * 100)}%`;

    const bar = document.getElementById("safety-bar");
    const safePct = (safe_count / total_samples) * 100;
    bar.innerHTML = `
      <span class="seg-safe" style="width:${safePct}%"></span>
      <span class="seg-unsafe" style="width:${100 - safePct}%"></span>
    `;

    const chart = document.getElementById("feature-chart");
    const averages = data.dataset.feature_averages;
    const max = Math.max(...Object.values(averages));
    chart.innerHTML = Object.entries(averages)
      .map(
        ([name, val]) => `
        <div class="feature-row">
          <span class="fname">${name}</span>
          <span class="fbar-track"><span class="fbar-fill" style="width:${(val / max) * 100}%"></span></span>
          <span class="fval">${fmt(val)}</span>
        </div>`
      )
      .join("");
  } catch (err) {
    console.error("stats load failed", err);
  }
}

// ---------------------------------------------------------
// Model performance table  (GET /api/metrics)
// ---------------------------------------------------------

async function loadMetrics() {
  try {
    const rows = await api("/api/metrics");
    const tbody = document.querySelector("#metrics-table tbody");

    if (!rows.length) {
      tbody.innerHTML = `<tr class="empty-row"><td colspan="5">No metrics recorded yet.</td></tr>`;
      return;
    }

    const bestF1 = Math.max(...rows.map((r) => r.f1_score));

    tbody.innerHTML = rows
      .map(
        (r) => `
        <tr>
          <td>${r.model}${r.f1_score === bestF1 ? " 🏆" : ""}</td>
          <td>${(r.accuracy * 100).toFixed(2)}%</td>
          <td>${(r.precision_score * 100).toFixed(2)}%</td>
          <td>${(r.recall * 100).toFixed(2)}%</td>
          <td>${(r.f1_score * 100).toFixed(2)}%</td>
        </tr>`
      )
      .join("");

    document.getElementById("stat-f1").textContent = bestF1.toFixed(3);
  } catch (err) {
    console.error("metrics load failed", err);
  }
}

// ---------------------------------------------------------
// Prediction history table  (GET /api/history)
// ---------------------------------------------------------

async function loadHistory() {
  try {
    const rows = await api("/api/history?limit=15");
    const tbody = document.querySelector("#history-table tbody");

    if (!rows.length) {
      tbody.innerHTML = `<tr class="empty-row"><td colspan="5">No predictions logged yet — try the form above.</td></tr>`;
      return;
    }

    tbody.innerHTML = rows
      .map((r) => {
        const finalSafe = r.final_prediction === 1;
        return `
        <tr>
          <td>${timeAgo(r.created_at)}</td>
          <td>${r.svm_prediction === 1 ? "Safe" : "Unsafe"} (${fmt(r.svm_confidence, 1)}%)</td>
          <td>${r.rf_prediction === 1 ? "Safe" : "Unsafe"} (${fmt(r.rf_confidence, 1)}%)</td>
          <td>${r.models_agree ? "Agree" : "Disagreed"}</td>
          <td><span class="pill ${finalSafe ? "pill-safe" : "pill-unsafe"}">${finalSafe ? "Safe" : "Unsafe"}</span></td>
        </tr>`;
      })
      .join("");
  } catch (err) {
    console.error("history load failed", err);
  }
}

// ---------------------------------------------------------
// Boot: load feature metadata first (needed to build the
// form), then load everything else in parallel
// ---------------------------------------------------------

async function init() {
  try {
    const meta = await api("/api/features");
    FEATURES = meta.features;
    FEATURE_INFO = meta.info;
    buildForm();
  } catch (err) {
    resultPanel.hidden = false;
    resultPanel.className = "result-panel is-unsafe";
    resultPanel.innerHTML = `<p class="result-note">Could not reach the backend API at ${API_BASE || location.origin}. Make sure "python backend/app.py" is running.</p>`;
    return;
  }

  loadStats();
  loadMetrics();
  loadHistory();
}

init();
