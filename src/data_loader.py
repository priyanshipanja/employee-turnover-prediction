"""
Data loading and validation module.
"""

import pandas as pd
import os
from .config import TARGET_COL, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TRAIN_DATA_PATH, TEST_DATA_PATH

def validate_schema(df, is_training=True):
    """Validates that incoming dataframe contains all expected feature columns."""
    required = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    if is_training:
        required = required + [TARGET_COL]
        
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {missing}")
    return True

def load_data(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH):
    """Loads and validates train and test datasets."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    validate_schema(train_df, is_training=True)
    validate_schema(test_df, is_training=True)
    
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    return X_train, y_train, X_test, y_test
