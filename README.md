# Enterprise Employee Turnover & Attrition Prediction Platform

An end-to-end Machine Learning and People Analytics platform designed to predict employee turnover risk, identify key flight-risk drivers, estimate the financial cost of attrition, and generate proactive, tailored HR retention strategies.

---

## Key Features

- **Production Machine Learning Pipeline**:
  - Automated feature engineering: workload-per-project intensity, satisfaction-evaluation perception gap, burnout flags, under-compensated high-performer alerts, tenure plateau metrics.
  - Multi-model evaluation: **Gradient Boosting (XGBoost)**, **Random Forest**, and **Logistic Regression**.
  - Production model achieved **0.7572 ROC-AUC** on a held-out test set of 3,000 employees.
- **Individual Employee Risk Diagnostic**:
  - Interactive UI with live parameter adjustments (Satisfaction, Evaluation, Monthly Hours, Projects, Tenure, Commute, Salary, Department, Overtime).
  - Dynamic risk gauge (Low / Medium / High Risk).
  - Personalized **Flight-Risk Driver** attribution.
  - Actionable **HR Retention Strategy Checklist** generated in real time.
- **Batch Workforce Roster Analytics**:
  - Upload entire department or company CSV rosters to instantly evaluate organizational attrition health.
  - Executive KPI cards: Total Headcount, High-Risk Headcount, Medium-Risk, Total Financial Attrition Exposure ($).
  - Interactive Department Churn Risk Distribution Bar Chart.
  - Paginated, filterable employee risk table with replacement cost impact.
- **Enterprise ROI & Cost-of-Turnover Calculator**:
  - Models financial exposure using industry benchmarks (50% of annual salary per churned employee).
- **RESTful API**:
  - `/api/predict` — Real-time single employee inference.
  - `/api/batch` — File upload & multi-employee scoring.
  - `/api/model-info` — Model architecture, confusion matrix, and feature importances.
  - `/api/sample-archetypes` — Preloaded archetype profiles for instant demoing.

---

## System Architecture

```
employee-turnover-prediction/
├── data/
│   ├── generate_dataset.py            # Generates 15,000 enterprise employee records
│   ├── hr_attrition_train.csv         # 12,000 training records
│   ├── hr_attrition_test.csv          # 3,000 held-out evaluation records
│   └── sample_batch_employees.csv     # Sample roster for batch scoring in UI
├── src/
│   ├── config.py                      # Feature columns, thresholds, hyperparameters
│   ├── data_loader.py                 # Schema validation & dataset loading
│   ├── preprocessor.py                # Pipeline: Custom feature engineering & scaling
│   ├── train.py                       # Multi-model benchmarking & artifact serialization
│   ├── evaluate.py                    # Classification metrics, confusion matrix, ROC-AUC
│   ├── predictor.py                   # High-performance inference engine
│   └── explainability.py              # Risk factor detection & retention strategy synthesis
├── models/
│   ├── best_model.joblib              # Serialized Gradient Boosting model
│   ├── preprocessor.joblib            # Fitted scikit-learn feature pipeline
│   └── model_metadata.json            # Benchmarks, metrics, and feature importances
├── app.py                             # Flask web application & REST API
├── templates/
│   └── index.html                     # Executive People Analytics dashboard
├── static/
│   ├── css/style.css                  # Modern responsive design
│   └── js/app.js                      # Chart.js gauges, dynamic advice, batch processor
├── tests/                             # Unit tests for data, pipeline & predictor
├── run_tests.py                       # Automated test runner
└── requirements.txt                   # Dependencies
```

---

## Model Evaluation & Benchmarks

All models were evaluated on the held-out test dataset (**3,000 records**):

| Algorithm | ROC-AUC | F1-Score | Accuracy | Production Status |
|---|---|---|---|---|
| **Gradient Boosting (XGBoost)** | **0.7572** | 0.3629 | **78.00%** | **Selected for Production** |
| Random Forest | 0.7444 | 0.4903 | 65.83% | Benchmark Candidate |
| Logistic Regression | 0.7445 | 0.4986 | 63.73% | Baseline Candidate |

### Top Predictive Features (Split Gain / Gini Importance):
1. `burnout_flag` — Excessive monthly hours (>240) across 5+ projects.
2. `satisfaction_level` — Employee self-reported engagement score.
3. `average_monthly_hours` — Monthly workload volume.
4. `hours_per_project` — Project load intensity ratio.
5. `unrewarded_high_performer` — High evaluation (>0.75) + Low salary + No recent promotion.
6. `time_spent_company` — Career tenure stagnation (3-5 year cliff).
7. `overtime` — Work-life balance strain.

---

## Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to **`http://localhost:5000`**.

### 3. Run Automated Tests
```bash
python run_tests.py
```

---

## 📡 REST API Reference

### 1. Single Employee Prediction
**`POST /api/predict`**

*Request Body:*
```json
{
  "satisfaction_level": 0.42,
  "last_evaluation": 0.89,
  "number_projects": 6,
  "average_monthly_hours": 270,
  "time_spent_company": 4,
  "work_accident": 0,
  "promotion_last_5years": 0,
  "department": "technical",
  "salary_level": "medium",
  "overtime": 1,
  "commute_distance_km": 25
}
```

*Response:*
```json
{
  "status": "success",
  "data": {
    "turnover_probability": 0.7812,
    "turnover_probability_pct": 78.1,
    "prediction": 1,
    "prediction_label": "Likely to Leave",
    "risk_level": "HIGH",
    "estimated_replacement_cost": 37500,
    "risk_factors": [
      {
        "factor": "Severe Workload & Burnout",
        "severity": "CRITICAL",
        "detail": "Working 270 hrs/month across 6 concurrent projects.",
        "impact": "+35% risk"
      }
    ],
    "retention_recommendations": [
      "Workload Audit: Immediately review project assignments; cap weekly overtime and redistribute at least one workstream.",
      "Wellness Support: Encourage mandatory compensatory time off and evaluate team resource allocation."
    ]
  }
}
```

### 2. Batch Roster Scoring
**`POST /api/batch`**
- Send `multipart/form-data` with `file: <employee_roster.csv>`.
- Returns summary statistics, risk breakdown, department aggregations, and individual employee predictions.

---
