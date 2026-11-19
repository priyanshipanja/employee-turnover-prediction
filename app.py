"""
Flask Web Application & REST API for Employee Turnover Prediction.
Provides interactive HR dashboard, single employee assessment, and batch file analytics.
"""

import os
import io
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

from src.config import BASE_DIR, SAMPLE_BATCH_PATH, METADATA_PATH, DEPARTMENTS, SALARY_LEVELS
from src.predictor import get_predictor

app = Flask(__name__,
            static_folder=os.path.join(BASE_DIR, "static"),
            template_folder=os.path.join(BASE_DIR, "templates"))

# Sample employee archetypes for one-click UI evaluation
SAMPLE_ARCHETYPES = {
    "burnout_performer": {
        "name": "Overworked Senior Engineer (Burnout Risk)",
        "satisfaction_level": 0.45,
        "last_evaluation": 0.92,
        "number_projects": 6,
        "average_monthly_hours": 275,
        "time_spent_company": 4,
        "work_accident": 0,
        "promotion_last_5years": 0,
        "department": "technical",
        "salary_level": "medium",
        "overtime": 1,
        "commute_distance_km": 28
    },
    "unrewarded_talent": {
        "name": "High-Potential Sales Lead (Underpaid Flight Risk)",
        "satisfaction_level": 0.52,
        "last_evaluation": 0.88,
        "number_projects": 4,
        "average_monthly_hours": 210,
        "time_spent_company": 3,
        "work_accident": 0,
        "promotion_last_5years": 0,
        "department": "sales",
        "salary_level": "low",
        "overtime": 1,
        "commute_distance_km": 35
    },
    "disengaged_junior": {
        "name": "Disengaged Support Associate (Bore-out Risk)",
        "satisfaction_level": 0.28,
        "last_evaluation": 0.51,
        "number_projects": 2,
        "average_monthly_hours": 135,
        "time_spent_company": 2,
        "work_accident": 0,
        "promotion_last_5years": 0,
        "department": "support",
        "salary_level": "low",
        "overtime": 0,
        "commute_distance_km": 12
    },
    "stable_veteran": {
        "name": "Thriving Retained Team Lead (Low Risk)",
        "satisfaction_level": 0.85,
        "last_evaluation": 0.82,
        "number_projects": 4,
        "average_monthly_hours": 195,
        "time_spent_company": 6,
        "work_accident": 0,
        "promotion_last_5years": 1,
        "department": "management",
        "salary_level": "high",
        "overtime": 0,
        "commute_distance_km": 8
    }
}

@app.route('/')
def index():
    return render_template('index.html', departments=DEPARTMENTS, salaries=SALARY_LEVELS)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        predictor = get_predictor()
        result = predictor.predict_single(data)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/batch', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "Empty file name"}), 400
            
        df = pd.read_csv(file)
        predictor = get_predictor()
        enriched_df, summary = predictor.predict_batch(df)
        
        # Return top 50 rows for table display
        preview_rows = enriched_df.head(50).to_dict(orient='records')
        
        return jsonify({
            "status": "success",
            "summary": summary,
            "preview": preview_rows,
            "total_records": len(enriched_df)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/model-info', methods=['GET'])
def model_info():
    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
            return jsonify({"status": "success", "metadata": metadata})
        return jsonify({"status": "error", "message": "Metadata not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/sample-archetypes', methods=['GET'])
def get_archetypes():
    return jsonify({"status": "success", "archetypes": SAMPLE_ARCHETYPES})

@app.route('/api/download-sample-csv', methods=['GET'])
def download_sample_csv():
    if os.path.exists(SAMPLE_BATCH_PATH):
        return send_file(SAMPLE_BATCH_PATH, as_attachment=True, download_name="sample_employees_roster.csv")
    return jsonify({"status": "error", "message": "Sample file not found"}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Employee Turnover Prediction Server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
