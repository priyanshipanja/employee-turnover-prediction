"""
Model evaluation module.
Computes comprehensive classification metrics, confusion matrices, and ROC-AUC.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

def evaluate_classifier(model, X_test, y_test):
    """Computes full evaluation suite for binary turnover classification."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "accuracy": float(np.round(accuracy_score(y_test, y_pred), 4)),
        "precision": float(np.round(precision_score(y_test, y_pred, zero_division=0), 4)),
        "recall": float(np.round(recall_score(y_test, y_pred, zero_division=0), 4)),
        "f1_score": float(np.round(f1_score(y_test, y_pred, zero_division=0), 4)),
        "roc_auc": float(np.round(roc_auc_score(y_test, y_prob), 4)),
        "pr_auc": float(np.round(average_precision_score(y_test, y_prob), 4)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    return metrics
