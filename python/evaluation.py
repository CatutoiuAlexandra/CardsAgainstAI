#!/usr/bin/env python3
"""
Module: evaluation.py

This module provides various evaluation functions for a CAH dataset, including:

1. White Card Prior Evaluation:
   - compute_white_card_prior: Computes a win-rate prior for each white card from training data
     and joins it with the test set.
   - evaluate_by_white_prior: Evaluates ranking based on white card prior (top-1, top-2, top-3 accuracy).

2. Group-Level Evaluation:
   - evaluate_group_predictions: Evaluates predictions on a group basis (using group id such as fake_round_id).
     It prints classification reports and computes ROC-AUC.

3. Round-Level Evaluation:
   - evaluate_round_predictions: Evaluates predictions on a per-round basis. It computes a binary
     "correct" flag based on the highest prediction per round and prints classification reports.

Usage:
  Run this module directly to see example outputs for group and round evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

# ============================================================================
# White Card Prior Evaluation
# ============================================================================
def compute_white_card_prior(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a win-rate prior for each white card from training data and joins it with the test set.
    
    Parameters:
      df_train (pd.DataFrame): Training data with at least 'white_card_text' and 'won' columns.
      df_test (pd.DataFrame): Test data with a 'white_card_text' column.
    
    Returns:
      pd.DataFrame: The test DataFrame with an added 'white_prior' column (using global prior if missing).
    """
    df_white_prior = df_train.groupby("white_card_text", as_index=False)["won"].mean()
    df_white_prior.rename(columns={"won": "white_prior"}, inplace=True)
    df_white_prior.set_index("white_card_text", inplace=True)
    df_test = df_test.join(df_white_prior, on="white_card_text", how="left")
    global_prior = df_train["won"].mean()
    df_test["white_prior"] = df_test["white_prior"].fillna(global_prior)
    return df_test

def evaluate_by_white_prior(df_test: pd.DataFrame):
    """
    Evaluates the ranking based on white card prior by computing top-1, top-2, and top-3 accuracies.
    
    Parameters:
      df_test (pd.DataFrame): Test DataFrame containing at least 'fake_round_id', 'won', and 'white_prior'.
    
    Prints:
      The top-1, top-2, and top-3 accuracies.
    """
    df_sorted = df_test.sort_values(["fake_round_id", "white_prior"], ascending=False)
    top1 = df_sorted.groupby("fake_round_id").head(1)["won"].mean()
    top2 = df_sorted.groupby("fake_round_id").head(2).groupby("fake_round_id")["won"].max().mean()
    top3 = df_sorted.groupby("fake_round_id").head(3).groupby("fake_round_id")["won"].max().mean()
    
    print("White Card Prior Evaluation:")
    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-2 Accuracy: {top2:.4f}")
    print(f"Top-3 Accuracy: {top3:.4f}")

# ============================================================================
# Group-Level Evaluation
# ============================================================================
def evaluate_group_predictions(df, preds, group_col="fake_round_id", label_col="won"):
    """
    Evaluates predictions on a group basis.
    
    The function assumes that:
      - df has a column for group id (e.g., fake_round_id) and true labels.
      - preds is an array of prediction scores corresponding to each row.
    
    It computes:
      - For each group, the maximum prediction score.
      - A binary flag (selected) if the prediction equals the group maximum.
      - A binary 'correct' flag if the selected row is a true positive.
      - Various metrics including a classification report and ROC-AUC.
    
    Parameters:
      df (pd.DataFrame): DataFrame containing at least the group identifier and true label.
      preds (array-like): Prediction scores.
      group_col (str): Column name for grouping.
      label_col (str): Column name for the true label.
      
    Returns:
      dict: A dictionary containing the classification report, ROC-AUC, and top-1 group accuracy.
    """
    df_eval = df[[group_col, label_col]].copy()
    df_eval["preds"] = preds
    # Compute maximum prediction per group
    df_eval["max_pred"] = df_eval.groupby(group_col)["preds"].transform("max")
    # Mark selected rows (those that equal the group maximum)
    df_eval["selected"] = (df_eval["preds"] == df_eval["max_pred"]).astype(int)
    # A row is correct if it is selected and its label is positive.
    df_eval["correct"] = ((df_eval["selected"] == 1) & (df_eval[label_col] > 0)).astype(int)
    
    # Compute overall metrics
    report = classification_report(df_eval[label_col], df_eval["correct"], output_dict=True)
    roc_auc = roc_auc_score(df_eval[label_col], df_eval["preds"])
    
    # Compute group-level accuracy: take one prediction per group (the maximum)
    group_top = df_eval.drop_duplicates(subset=[group_col])
    top1_acc = group_top[label_col].mean()
    
    metrics = {
        "classification_report": report,
        "roc_auc": roc_auc,
        "top1_group_accuracy": top1_acc
    }
    
    print("=== Group-Level Evaluation ===")
    print("Top-1 accuracy (mean label per group):", round(top1_acc, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("Classification Report (per row selection):")
    print(classification_report(df_eval[label_col], df_eval["correct"]))
    
    return metrics

# ============================================================================
# Round-Level Evaluation
# ============================================================================
def evaluate_round_predictions(df_test: pd.DataFrame, preds):
    """
    Evaluates predictions on a per-round basis.
    
    The function expects df_test to contain at least:
      - "fake_round_id": group identifier for a round.
      - "won": the binary ground-truth label.
    And a vector of predictions (preds) corresponding to each row.
    
    It computes:
      - A new column "m_score" (max prediction per round).
      - A binary "correct" flag: 1 if the prediction equals the round max and the row is positive.
      - Various aggregated metrics (top-1, top-2, top-3 accuracies by round).
    
    Prints:
      A full classification report (rowwise) and reports computed using one prediction per round.
    """
    df_eval = df_test[["fake_round_id", "won"]].copy()
    df_eval["preds"] = preds
    
    if df_eval["preds"].nunique() < 2:
        raise ValueError("Predictions appear constant (or NaN). Check your inputs.")
    
    # For each round, mark the maximum prediction
    df_eval["m_score"] = df_eval.groupby("fake_round_id")["preds"].transform("max")
    # A row is correct if it has the max prediction for its round and is a win.
    df_eval["correct"] = ((df_eval["preds"] == df_eval["m_score"]) & (df_eval["won"] > 0)).astype(int)
    
    # Shuffle and sort by prediction score descending
    df_eval = df_eval.sample(frac=1)
    df_eval = df_eval.sort_values("preds", ascending=False)
    
    print("=== Full Classification Report (per row) ===")
    print(classification_report(df_eval["won"], df_eval["correct"]))
    
    print("\n=== Top-1 Evaluation (one prediction per round) ===")
    top1_df = df_eval.drop_duplicates(subset=["fake_round_id"])
    print(classification_report(top1_df["won"], top1_df["correct"]))
    print("\nAlternative top-1 accuracy (group-level):", top1_df["won"].mean())
    print("Top-2 accuracy by round:",
          df_eval.groupby("fake_round_id").head(2).groupby("fake_round_id")["won"].max().mean())
    print("Top-3 accuracy by round:",
          df_eval.groupby("fake_round_id").head(3).groupby("fake_round_id")["won"].max().mean())

# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    # --- Dummy example for Group-Level Evaluation ---
    print("=== Group-Level Evaluation Example ===")
    df_dummy = pd.DataFrame({
        'fake_round_id': [1, 1, 2, 2],
        'won': [1, 0, 0, 1]
    })
    preds_dummy = np.array([0.9, 0.1, 0.3, 0.7])
    evaluate_group_predictions(df_dummy, preds_dummy)
    
    # --- Dummy example for Round-Level Evaluation ---
    print("\n=== Round-Level Evaluation Example ===")
    df_round_dummy = pd.DataFrame({
        'fake_round_id': [1, 1, 2, 2, 3, 3],
        'won': [1, 0, 0, 1, 1, 0]
    })
    preds_round_dummy = np.array([0.8, 0.2, 0.4, 0.6, 0.7, 0.3])
    evaluate_round_predictions(df_round_dummy, preds_round_dummy)
    
    # --- Dummy example for White Card Prior Evaluation ---
    print("\n=== White Card Prior Evaluation Example ===")
    # Create dummy training data
    df_train = pd.DataFrame({
        "white_card_text": ["joke1", "joke2", "joke1", "joke3"],
        "won": [1, 0, 1, 0]
    })
    # Create dummy test data
    df_test = pd.DataFrame({
        "white_card_text": ["joke1", "joke2", "joke4"],
        "won": [1, 0, 1],
        "fake_round_id": [1, 2, 3]
    })
    df_test = compute_white_card_prior(df_train, df_test)
    evaluate_by_white_prior(df_test)
