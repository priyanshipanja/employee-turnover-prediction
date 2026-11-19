"""
Configuration settings, feature definitions, and model paths for Employee Turnover Prediction.
"""

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "hr_attrition_train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "hr_attrition_test.csv")
SAMPLE_BATCH_PATH = os.path.join(DATA_DIR, "sample_batch_employees.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

TARGET_COL = "turnover"

NUMERICAL_FEATURES = [
    "satisfaction_level",
    "last_evaluation",
    "number_projects",
    "average_monthly_hours",
    "time_spent_company",
    "work_accident",
    "promotion_last_5years",
    "overtime",
    "commute_distance_km"
]

CATEGORICAL_FEATURES = [
    "department",
    "salary_level"
]

DEPARTMENTS = [
    'sales', 'technical', 'support', 'IT', 'product_mng',
    'marketing', 'RandD', 'accounting', 'hr', 'management'
]

SALARY_LEVELS = ['low', 'medium', 'high']

# Risk Categorization Thresholds
RISK_THRESHOLDS = {
    "LOW": 0.35,
    "MEDIUM": 0.65
}

# Cost of turnover benchmark parameters (for HR ROI calculator)
AVG_EMPLOYEE_REPLACEMENT_COST_FACTOR = 0.50 # 50% of annual salary to replace an employee
DEFAULT_BASE_SALARIES = {
    "low": 45000,
    "medium": 75000,
    "high": 125000
}
