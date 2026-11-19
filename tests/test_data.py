"""
Unit tests for data generation and schema integrity.
"""

import os
import pandas as pd
from data.generate_dataset import generate_hr_dataset
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL

def test_dataset_generation():
    df = generate_hr_dataset(n_samples=500, random_state=42)
    assert len(df) == 500
    assert TARGET_COL in df.columns
    
    # Verify all expected features exist
    for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        assert col in df.columns
        
    # Check bounds
    assert (df['satisfaction_level'] >= 0.0).all() and (df['satisfaction_level'] <= 1.0).all()
    assert (df['last_evaluation'] >= 0.0).all() and (df['last_evaluation'] <= 1.0).all()
    assert (df['average_monthly_hours'] >= 80).all()
    assert set(df['turnover'].unique()).issubset({0, 1})
    assert df.isnull().sum().sum() == 0
