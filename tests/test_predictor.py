"""
Unit tests for the turnover prediction engine and explainability rules.
"""

import pandas as pd
from src.predictor import get_predictor
from data.generate_dataset import generate_hr_dataset

def test_single_prediction():
    predictor = get_predictor()
    sample_employee = {
        "satisfaction_level": 0.40,
        "last_evaluation": 0.90,
        "number_projects": 6,
        "average_monthly_hours": 260,
        "time_spent_company": 4,
        "work_accident": 0,
        "promotion_last_5years": 0,
        "department": "technical",
        "salary_level": "low",
        "overtime": 1,
        "commute_distance_km": 30
    }
    
    result = predictor.predict_single(sample_employee)
    
    assert "turnover_probability" in result
    assert 0.0 <= result["turnover_probability"] <= 1.0
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert result["estimated_replacement_cost"] > 0
    assert isinstance(result["risk_factors"], list)
    assert isinstance(result["retention_recommendations"], list)
    assert len(result["retention_recommendations"]) > 0

def test_batch_prediction():
    predictor = get_predictor()
    raw_df = generate_hr_dataset(n_samples=50, random_state=42)
    X = raw_df.drop(columns=['turnover'])
    X.insert(0, 'employee_id', [f"EMP-{i}" for i in range(50)])
    
    enriched_df, summary = predictor.predict_batch(X)
    
    assert len(enriched_df) == 50
    assert "turnover_probability" in enriched_df.columns
    assert "risk_level" in enriched_df.columns
    assert summary["total_employees"] == 50
    assert (summary["high_risk_count"] + summary["medium_risk_count"] + summary["low_risk_count"]) == 50
    assert summary["total_financial_exposure"] >= 0
