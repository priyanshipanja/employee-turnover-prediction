"""
Model Training & Selection Pipeline.
Trains multiple candidate algorithms, evaluates on held-out test data,
and persists the top performing production model with full metadata.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .config import (
    MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, DEPARTMENTS, SALARY_LEVELS
)
from .data_loader import load_data
from .preprocessor import build_preprocessing_pipeline, save_preprocessor
from .evaluate import evaluate_classifier

def get_feature_names(preprocessor):
    """Recovers output feature names from the fitted ColumnTransformer pipeline."""
    feature_names = []
    
    # 1. Engineered numerical feature names
    ct = preprocessor.named_steps['transformer']
    num_cols = ct.transformers_[0][2]
    feature_names.extend(num_cols)
    
    # 2. Categorical one-hot encoded feature names
    cat_encoder = ct.transformers_[1][1].named_steps['encoder']
    for i, cat_col in enumerate(CATEGORICAL_FEATURES):
        cats = cat_encoder.categories_[i]
        for cat in cats:
            feature_names.append(f"{cat_col}_{cat}")
            
    return feature_names

def train_and_evaluate():
    print("=" * 60)
    print("🚀 EMPLOYEE TURNOVER PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load Data
    print("Loading datasets...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Testing samples:  {len(X_test):,}")
    
    # 2. Fit Preprocessor
    print("\nFitting preprocessing & feature engineering pipeline...")
    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    feature_names = get_feature_names(preprocessor)
    print(f"  Total transformed features: {len(feature_names)}")
    
    # 3. Define Candidate Classifiers
    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": (
            XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                random_state=42,
                eval_metric='logloss'
            ) if HAS_XGBOOST else GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                random_state=42
            )
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
    }
    
    results = {}
    best_model_name = None
    best_auc = -1.0
    best_model = None
    
    print("\nTraining and evaluating candidate models:")
    for name, clf in candidates.items():
        print(f"  Training {name}...", end=" ", flush=True)
        clf.fit(X_train_proc, y_train)
        metrics = evaluate_classifier(clf, X_test_proc, y_test)
        results[name] = metrics
        print(f"ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
        
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model_name = name
            best_model = clf
            
    print(f"\n🏆 Best Selected Model: {best_model_name} (ROC-AUC: {best_auc:.4f})")
    
    # 4. Extract Feature Importances
    importances = {}
    if hasattr(best_model, "feature_importances_"):
        raw_importances = best_model.feature_importances_
        sorted_indices = np.argsort(raw_importances)[::-1]
        for idx in sorted_indices[:15]:
            importances[feature_names[idx]] = float(np.round(raw_importances[idx], 4))
    elif hasattr(best_model, "coef_"):
        raw_coefs = np.abs(best_model.coef_[0])
        sorted_indices = np.argsort(raw_coefs)[::-1]
        for idx in sorted_indices[:15]:
            importances[feature_names[idx]] = float(np.round(raw_coefs[idx], 4))
            
    # 5. Save Artifacts
    print(f"\nSaving production model artifacts to {os.path.dirname(MODEL_PATH)}...")
    joblib.dump(best_model, MODEL_PATH)
    save_preprocessor(preprocessor, PREPROCESSOR_PATH)
    
    metadata = {
        "model_name": best_model_name,
        "trained_at": datetime.datetime.now().isoformat(),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "top_features": importances,
        "metrics": results[best_model_name],
        "all_model_benchmarks": {k: {"roc_auc": v["roc_auc"], "f1": v["f1_score"], "accuracy": v["accuracy"]} for k, v in results.items()}
    }
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("  ✓ Model artifact:      best_model.joblib")
    print("  ✓ Preprocessor:        preprocessor.joblib")
    print("  ✓ Metadata & Metrics:  model_metadata.json")
    print("=" * 60)
    print("TRAINING COMPLETE & READY FOR PRODUCTION!")
    print("=" * 60)
    
    return best_model, preprocessor, metadata

if __name__ == '__main__':
    train_and_evaluate()
