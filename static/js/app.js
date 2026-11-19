// HR Sentinel - Frontend Application & Interactive Diagnostics

let gaugeChart = null;
let deptChart = null;
let featureChart = null;
let archetypesCache = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSliders();
  initGaugeChart();
  initModelInfo();
  loadArchetypesData();
  initBatchUpload();
  
  // Initial calculation on default values
  evaluateSingle();
});

// Tab Switching
function initTabs() {
  const buttons = document.querySelectorAll('.tab-button');
  const panes = document.querySelectorAll('.tab-pane');

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) {
        targetPane.classList.add('active');
      }
    });
  });
}

// Connect slider inputs to live value labels
function initSliders() {
  const bindings = [
    { id: 'satisfaction_level', labelId: 'val_satisfaction', format: v => `${Math.round(v * 100)}%` },
    { id: 'last_evaluation', labelId: 'val_evaluation', format: v => `${Math.round(v * 100)}%` },
    { id: 'average_monthly_hours', labelId: 'val_hours', format: v => `${v} hrs` },
    { id: 'number_projects', labelId: 'val_projects', format: v => `${v}` },
    { id: 'time_spent_company', labelId: 'val_tenure', format: v => `${v} yrs` },
    { id: 'commute_distance_km', labelId: 'val_commute', format: v => `${v} km` }
  ];

  bindings.forEach(({ id, labelId, format }) => {
    const input = document.getElementById(id);
    const label = document.getElementById(labelId);
    if (input && label) {
      input.addEventListener('input', () => {
        label.textContent = format(input.value);
      });
    }
  });
}

// Semi-circle doughnut gauge for risk probability
function initGaugeChart() {
  const ctx = document.getElementById('gaugeChart').getContext('2d');
  gaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [15, 85],
        backgroundColor: ['#10b981', '#1f2937'],
        borderWidth: 0,
        circumference: 180,
        rotation: 270
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: { enabled: false }
      },
      cutout: '78%'
    }
  });
}

function updateGauge(prob) {
  if (!gaugeChart) return;
  const pct = Math.round(prob * 100);
  
  let color = '#10b981'; // Green
  if (prob >= 0.65) {
    color = '#ef4444'; // Red
  } else if (prob >= 0.35) {
    color = '#f59e0b'; // Amber
  }

  gaugeChart.data.datasets[0].data = [pct, 100 - pct];
  gaugeChart.data.datasets[0].backgroundColor = [color, '#1f2937'];
  gaugeChart.update();
}

// Single employee prediction API call
async function evaluateSingle() {
  const payload = {
    satisfaction_level: parseFloat(document.getElementById('satisfaction_level').value),
    last_evaluation: parseFloat(document.getElementById('last_evaluation').value),
    average_monthly_hours: parseInt(document.getElementById('average_monthly_hours').value, 10),
    number_projects: parseInt(document.getElementById('number_projects').value, 10),
    time_spent_company: parseInt(document.getElementById('time_spent_company').value, 10),
    commute_distance_km: parseInt(document.getElementById('commute_distance_km').value, 10),
    department: document.getElementById('department').value,
    salary_level: document.getElementById('salary_level').value,
    overtime: document.getElementById('overtime').checked ? 1 : 0,
    promotion_last_5years: document.getElementById('promotion_last_5years').checked ? 1 : 0,
    work_accident: document.getElementById('work_accident').checked ? 1 : 0
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const json = await res.json();
    if (json.status === 'success') {
      renderDiagnosticResults(json.data);
    }
  } catch (err) {
    console.error("Evaluation error:", err);
  }
}

function renderDiagnosticResults(data) {
  // Update numbers & gauge
  document.getElementById('probNumber').textContent = `${data.turnover_probability_pct}%`;
  document.getElementById('predLabel').textContent = data.prediction_label;
  document.getElementById('replacementCost').textContent = `$${data.estimated_replacement_cost.toLocaleString()}`;
  updateGauge(data.turnover_probability);

  // Risk Badge
  const badge = document.getElementById('riskBadge');
  badge.className = `risk-pill risk-${data.risk_level.toLowerCase()}`;
  badge.textContent = `${data.risk_level} RISK`;

  // Render Risk Drivers
  const factorsList = document.getElementById('riskFactorsList');
  if (data.risk_factors && data.risk_factors.length > 0) {
    factorsList.innerHTML = data.risk_factors.map(f => `
      <div class="factor-card severity-${f.severity.toLowerCase()}">
        <div>
          <div class="factor-card-title">${f.factor}</div>
          <div class="factor-card-desc">${f.detail}</div>
        </div>
        <div class="factor-impact">${f.impact}</div>
      </div>
    `).join('');
  } else {
    factorsList.innerHTML = `<div class="empty-state">No acute flight-risk factors detected. Employee indicators are well-balanced.</div>`;
  }

  // Render Retention Recommendations
  const recsList = document.getElementById('recommendationsList');
  if (data.retention_recommendations && data.retention_recommendations.length > 0) {
    recsList.innerHTML = data.retention_recommendations.map(r => `<li>${r}</li>`).join('');
  }
}

// Load Archetypes Cache
async function loadArchetypesData() {
  try {
    const res = await fetch('/api/sample-archetypes');
    const json = await res.json();
    if (json.status === 'success') {
      archetypesCache = json.archetypes;
    }
  } catch (e) {
    console.warn("Could not load archetypes:", e);
  }
}

function loadArchetype(key) {
  if (!archetypesCache || !archetypesCache[key]) return;
  const p = archetypesCache[key];

  document.getElementById('satisfaction_level').value = p.satisfaction_level;
  document.getElementById('val_satisfaction').textContent = `${Math.round(p.satisfaction_level * 100)}%`;

  document.getElementById('last_evaluation').value = p.last_evaluation;
  document.getElementById('val_evaluation').textContent = `${Math.round(p.last_evaluation * 100)}%`;

  document.getElementById('average_monthly_hours').value = p.average_monthly_hours;
  document.getElementById('val_hours').textContent = `${p.average_monthly_hours} hrs`;

  document.getElementById('number_projects').value = p.number_projects;
  document.getElementById('val_projects').textContent = p.number_projects;

  document.getElementById('time_spent_company').value = p.time_spent_company;
  document.getElementById('val_tenure').textContent = `${p.time_spent_company} yrs`;

  document.getElementById('commute_distance_km').value = p.commute_distance_km;
  document.getElementById('val_commute').textContent = `${p.commute_distance_km} km`;

  document.getElementById('department').value = p.department;
  document.getElementById('salary_level').value = p.salary_level;

  document.getElementById('overtime').checked = p.overtime === 1;
  document.getElementById('promotion_last_5years').checked = p.promotion_last_5years === 1;
  document.getElementById('work_accident').checked = p.work_accident === 1;

  evaluateSingle();
}

// Batch CSV Processing
function initBatchUpload() {
  const fileInput = document.getElementById('batchFileInput');
  if (!fileInput) return;

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/batch', {
        method: 'POST',
        body: formData
      });
      const json = await res.json();
      if (json.status === 'success') {
        renderBatchResults(json.summary, json.preview);
      } else {
        alert("Batch error: " + json.message);
      }
    } catch (err) {
      alert("Failed to process batch CSV: " + err.message);
    }
  });
}

function renderBatchResults(summary, preview) {
  // Show sections
  document.getElementById('batchSummarySection').style.display = 'grid';
  document.getElementById('batchChartSection').style.display = 'block';
  document.getElementById('batchTableCard').style.display = 'block';

  // Fill KPI cards
  document.getElementById('batchTotal').textContent = summary.total_employees;
  document.getElementById('batchHighRisk').textContent = summary.high_risk_count;
  document.getElementById('batchMedRisk').textContent = summary.medium_risk_count;
  document.getElementById('batchExposure').textContent = `$${summary.total_financial_exposure.toLocaleString()}`;

  // Render Department Risk Bar Chart
  const depts = Object.keys(summary.department_risk_breakdown);
  const values = Object.values(summary.department_risk_breakdown);

  const ctx = document.getElementById('deptChart').getContext('2d');
  if (deptChart) deptChart.destroy();

  deptChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: depts.map(d => d.toUpperCase()),
      datasets: [{
        label: 'Average Churn Probability (%)',
        data: values,
        backgroundColor: values.map(v => v >= 50 ? '#ef4444' : (v >= 30 ? '#f59e0b' : '#4f46e5')),
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { callback: v => `${v}%` } }
      }
    }
  });

  // Render Table
  document.getElementById('tableRecordCount').textContent = `Showing Top ${preview.length} of ${summary.total_employees} Records`;
  const tbody = document.getElementById('batchTableBody');
  tbody.innerHTML = preview.map(row => {
    const riskClass = `risk-${row.risk_level.toLowerCase()}`;
    return `
      <tr>
        <td><strong>${row.employee_id || 'N/A'}</strong></td>
        <td>${(row.department || '').toUpperCase()}</td>
        <td>${Math.round(row.satisfaction_level * 100)}%</td>
        <td>${Math.round(row.last_evaluation * 100)}%</td>
        <td>${row.average_monthly_hours}</td>
        <td>${(row.salary_level || '').toUpperCase()}</td>
        <td>${row.overtime === 1 ? '<span class="text-red">Yes</span>' : 'No'}</td>
        <td><span class="risk-pill ${riskClass}">${row.turnover_probability_pct}% (${row.risk_level})</span></td>
        <td>$${(row.replacement_cost || 0).toLocaleString()}</td>
      </tr>
    `;
  }).join('');
}

// Model Architecture & Explainability Tab
async function initModelInfo() {
  try {
    const res = await fetch('/api/model-info');
    const json = await res.json();
    if (json.status !== 'success') return;
    const meta = json.metadata;

    // Header badge
    document.getElementById('headerModelName').textContent = `${meta.model_name} Model Active`;

    // Confusion Matrix
    if (meta.metrics && meta.metrics.confusion_matrix) {
      const cm = meta.metrics.confusion_matrix;
      document.getElementById('cmTN').textContent = cm.true_negatives.toLocaleString();
      document.getElementById('cmFP').textContent = cm.false_positives.toLocaleString();
      document.getElementById('cmFN').textContent = cm.false_negatives.toLocaleString();
      document.getElementById('cmTP').textContent = cm.true_positives.toLocaleString();
    }

    // Top Features Horizontal Bar Chart
    if (meta.top_features) {
      const labels = Object.keys(meta.top_features).slice(0, 10);
      const vals = Object.values(meta.top_features).slice(0, 10);

      const ctx = document.getElementById('featureChart').getContext('2d');
      if (featureChart) featureChart.destroy();

      featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels.map(l => l.replace(/_/g, ' ')),
          datasets: [{
            label: 'Feature Importance Score',
            data: vals,
            backgroundColor: '#6366f1',
            borderRadius: 4
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { x: { beginAtZero: true } }
        }
      });
    }
  } catch (e) {
    console.warn("Could not load model info:", e);
  }
}
