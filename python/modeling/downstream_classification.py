#!/usr/bin/env python3
"""
Module: downstream_classification.py

This module provides functions for training and evaluating downstream classification models.
It includes:

  - train_logistic_regression: Trains a logistic regression classifier in a pipeline with standard scaling.
  - train_random_forest: Trains a random forest classifier with configurable parameters.
  - evaluate_classifier: Evaluates a classifier by printing the classification report and ROC AUC score.

Usage:
  Run this module directly to see an example with random training and testing data.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# ---------------------------------------------------------------------------
# Classifier Training Functions
# ---------------------------------------------------------------------------
def train_logistic_regression(X, y, solver="sag"):
    """
    Trains a logistic regression classifier using a pipeline with standard scaling.
    
    Parameters:
      X (array-like): Training feature matrix.
      y (array-like): Training labels.
      solver (str): Solver to use in logistic regression (default: "sag").
    
    Returns:
      clf: Trained logistic regression classifier.
    """
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver=solver, max_iter=1000))
    clf.fit(X, y)
    return clf

def train_random_forest(X, y, **kwargs):
    """
    Trains a random forest classifier.
    
    Parameters:
      X (array-like): Training feature matrix.
      y (array-like): Training labels.
      **kwargs: Additional keyword arguments for RandomForestClassifier.
    
    Returns:
      clf: Trained random forest classifier.
    """
    clf = RandomForestClassifier(**kwargs)
    clf.fit(X, y)
    return clf

# ---------------------------------------------------------------------------
# Classifier Evaluation Function
# ---------------------------------------------------------------------------
def evaluate_classifier(clf, X_test, y_test):
    """
    Evaluates the classifier by printing the classification report and ROC AUC score.
    
    Parameters:
      clf: Trained classifier.
      X_test (array-like): Test feature matrix.
      y_test (array-like): True test labels.
    
    Returns:
      dict: A dictionary with keys "report" and "roc_auc" containing the evaluation results.
    """
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    print("Classification Report:\n", report)
    if auc is not None:
        print("ROC AUC: {:.4f}".format(auc))
    return {"report": report, "roc_auc": auc}

# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Generate dummy training and test data
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(50, 10)
    y_test = np.random.randint(0, 2, 50)
    
    print("Training Logistic Regression...")
    clf_lr = train_logistic_regression(X_train, y_train)
    evaluate_classifier(clf_lr, X_test, y_test)
    
    print("\nTraining Random Forest...")
    clf_rf = train_random_forest(X_train, y_train, n_estimators=50, max_depth=5, random_state=42)
    evaluate_classifier(clf_rf, X_test, y_test)
