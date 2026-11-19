"""
Data Preprocessing & Feature Engineering Pipeline.
Handles automated feature engineering, scaling, and categorical encoding.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from .config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, DEPARTMENTS, SALARY_LEVELS

class HRFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer creating empirical HR domain features."""
    
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        
        # 1. Workload intensity per project
        X['hours_per_project'] = X['average_monthly_hours'] / np.maximum(X['number_projects'], 1)
        
        # 2. Perception gap between evaluation and personal satisfaction
        X['satisfaction_eval_gap'] = X['last_evaluation'] - X['satisfaction_level']
        
        # 3. Burnout condition indicator
        X['burnout_flag'] = ((X['average_monthly_hours'] > 240) & (X['number_projects'] >= 5)).astype(int)
        
        # 4. Under-compensated top performer
        X['unrewarded_high_performer'] = (
            (X['last_evaluation'] > 0.75) & 
            (X['salary_level'] == 'low') & 
            (X['promotion_last_5years'] == 0)
        ).astype(int)
        
        # 5. Career plateau risk (3-5 years without promotion)
        X['tenure_plateau'] = (
            (X['time_spent_company'].isin([3, 4, 5])) & 
            (X['promotion_last_5years'] == 0)
        ).astype(int)
        
        # 6. Disengagement indicator (low satisfaction + low hours)
        X['disengagement_flag'] = (
            (X['satisfaction_level'] < 0.40) & 
            (X['average_monthly_hours'] < 160)
        ).astype(int)
        
        return X

def build_preprocessing_pipeline():
    """Builds an end-to-end ColumnTransformer pipeline with custom feature engineering."""
    engineered_numerical = [
        'satisfaction_level',
        'last_evaluation',
        'number_projects',
        'average_monthly_hours',
        'time_spent_company',
        'work_accident',
        'promotion_last_5years',
        'overtime',
        'commute_distance_km',
        'hours_per_project',
        'satisfaction_eval_gap',
        'burnout_flag',
        'unrewarded_high_performer',
        'tenure_plateau',
        'disengagement_flag'
    ]
    
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(
            categories=[DEPARTMENTS, SALARY_LEVELS],
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])
    
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, engineered_numerical),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    full_pipeline = Pipeline([
        ('feature_engineer', HRFeatureEngineer()),
        ('transformer', column_transformer)
    ])
    
    return full_pipeline

def save_preprocessor(pipeline, file_path):
    joblib.dump(pipeline, file_path)

def load_preprocessor(file_path):
    return joblib.load(file_path)
