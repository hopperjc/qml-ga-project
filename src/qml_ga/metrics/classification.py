# src/qml_ga/metrics/classification.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def binarize_logits(logits):
    logits = np.asarray(logits, dtype=float).ravel()
    return np.where(logits >= 0.0, 1.0, -1.0)

def all_metrics(y_true, logits):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = binarize_logits(logits)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
