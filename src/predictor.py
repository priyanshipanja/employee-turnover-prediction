"""
Turnover Prediction Engine.
Provides high-performance single-employee inference and batch CSV processing.
"""

import os
import numpy as np
import pandas as pd
import joblib

from .config import (
    MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH,
    RISK_THRESHOLDS, DEFAULT_BASE_SALARIES, AVG_EMPLOYEE_REPLACEMENT_COST_FACTOR,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)
from .explainability import extract_employee_risk_factors, generate_retention_recommendations

class TurnoverPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError("Model or Preprocessor artifact not found. Run training first.")
            
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        if os.path.exists(METADATA_PATH):
            import json
            with open(METADATA_PATH, "r") as f:
                self.metadata = json.load(f)
                
    def _classify_risk(self, probability):
        if probability >= RISK_THRESHOLDS["MEDIUM"]:
            return "HIGH"
        elif probability >= RISK_THRESHOLDS["LOW"]:
            return "MEDIUM"
        return "LOW"
        
    def _estimate_replacement_cost(self, salary_level):
        base_salary = DEFAULT_BASE_SALARIES.get(str(salary_level).lower(), 75000)
        return int(base_salary * AVG_EMPLOYEE_REPLACEMENT_COST_FACTOR)

    def predict_single(self, employee_data):
        """
        Inference on a single employee dictionary.
        Returns complete diagnostic profile with probability, risk level, cost, and retention plan.
        """
        # Ensure DataFrame structure
        input_df = pd.DataFrame([employee_data])
        
        # Transform & Predict
        X_proc = self.preprocessor.transform(input_df)
        prob = float(self.model.predict_proba(X_proc)[0, 1])
        prediction = int(prob >= 0.5)
        risk_level = self._classify_risk(prob)
        
        # Financial turnover impact
        replacement_cost = self._estimate_replacement_cost(employee_data.get('salary_level', 'medium'))
        
        # Explainability & Recommendations
        risk_factors = extract_employee_risk_factors(employee_data)
        recommendations = generate_retention_recommendations(risk_level, risk_factors, employee_data)
        
        return {
            "turnover_probability": float(np.round(prob, 4)),
            "turnover_probability_pct": float(np.round(prob * 100, 1)),
            "prediction": prediction,
            "prediction_label": "Likely to Leave" if prediction == 1 else "Likely to Stay",
            "risk_level": risk_level,
            "estimated_replacement_cost": replacement_cost,
            "risk_factors": risk_factors,
            "retention_recommendations": recommendations,
            "input_summary": employee_data
        }

    def predict_batch(self, df):
        """
        Batch inference over multiple employee records.
        Returns enriched dataframe with predictions, probabilities, and aggregated workforce insights.
        """
        df = df.copy()
        
        # Keep identifier column if present
        id_col = None
        for candidate in ['employee_id', 'id', 'emp_id']:
            if candidate in df.columns:
                id_col = candidate
                break
                
        # Drop ID column for model input if exists
        features_df = df.drop(columns=[id_col]) if id_col else df
        
        # Preprocess & Predict
        X_proc = self.preprocessor.transform(features_df)
        probs = self.model.predict_proba(X_proc)[:, 1]
        
        df['turnover_probability'] = np.round(probs, 4)
        df['turnover_probability_pct'] = np.round(probs * 100, 1)
        df['risk_level'] = [self._classify_risk(p) for p in probs]
        df['replacement_cost'] = [self._estimate_replacement_cost(s) for s in df['salary_level']]
        
        # Aggregate Workforce Summary
        total_employees = len(df)
        high_risk_count = int((df['risk_level'] == 'HIGH').sum())
        med_risk_count = int((df['risk_level'] == 'MEDIUM').sum())
        low_risk_count = int((df['risk_level'] == 'LOW').sum())
        
        total_risk_exposure = int(df[df['risk_level'] == 'HIGH']['replacement_cost'].sum())
        avg_turnover_prob = float(np.round(probs.mean() * 100, 1))
        
        dept_risk = df.groupby('department')['turnover_probability'].mean().to_dict()
        dept_risk = {k: float(np.round(v * 100, 1)) for k, v in dept_risk.items()}
        
        summary = {
            "total_employees": total_employees,
            "high_risk_count": high_risk_count,
            "medium_risk_count": med_risk_count,
            "low_risk_count": low_risk_count,
            "overall_turnover_rate_pct": avg_turnover_prob,
            "total_financial_exposure": total_risk_exposure,
            "department_risk_breakdown": dept_risk
        }
        
        return df, summary

# Singleton accessor
_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = TurnoverPredictor()
    return _predictor_instance
