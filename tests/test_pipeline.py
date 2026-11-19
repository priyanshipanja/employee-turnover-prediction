"""
Unit tests for preprocessing & feature engineering pipeline.
"""

import pandas as pd
from src.preprocessor import build_preprocessing_pipeline
from data.generate_dataset import generate_hr_dataset

def test_preprocessing_pipeline():
    raw_df = generate_hr_dataset(n_samples=200, random_state=42)
    X = raw_df.drop(columns=['turnover'])
    
    pipeline = build_preprocessing_pipeline()
    X_trans = pipeline.fit_transform(X)
    
    # Check transformed shape
    assert X_trans.shape[0] == 200
    assert X_trans.shape[1] > X.shape[1] # Engineered features expand the dimension
    assert not pd.isna(X_trans).any()
