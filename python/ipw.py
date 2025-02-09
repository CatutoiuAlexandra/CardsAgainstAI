#!/usr/bin/env python3
"""
Module: ipw.py

This module provides functions for calculating inverse probability weights (IPW)
and computing a target encoding (prior) for white card texts.

Functions:
  - calculate_ipw_weights(y_true, y_pred):
      Given binary labels and predicted probabilities, returns the inverse probability weights.
  - compute_white_prior(df, white_col="white_card_text", target_col="pick_ratio", cv=5, min_samples_leaf=3, smoothing=0.2):
      Computes a target encoding for white card texts using NestedCVWrapper with CatBoostEncoder.

Usage:
  Run this module directly to see example outputs.
"""

import numpy as np
import pandas as pd

def calculate_ipw_weights(y_true, y_pred):
    """
    Calculates inverse probability weights for a binary target.
    
    For a treated unit (y_true == 1), the weight is 1 / p.
    For an untreated unit (y_true == 0), the weight is 1 / (1 - p).
    
    Parameters:
      y_true: Array-like of true binary labels (0 or 1).
      y_pred: Array-like of predicted probabilities.
    
    Returns:
      Numpy array of IPW weights.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = np.where(y_true == 1, 1.0 / y_pred, 1.0 / (1.0 - y_pred))
    return weights

def compute_white_prior(df, white_col="white_card_text", target_col="pick_ratio", cv=5, min_samples_leaf=3, smoothing=0.2):
    """
    Computes a target encoding (prior) for the white card text.
    
    Parameters:
      df (DataFrame): Data with white card texts and a target (e.g., pick_ratio).
      white_col (str): Column name for white card text.
      target_col (str): Column name for the target.
      cv (int): Number of cross-validation folds.
      min_samples_leaf (int): Parameter for the encoder.
      smoothing (float): Parameter for the encoder.
    
    Returns:
      Series: The encoded target (prior) for each white card.
    """
    try:
        from category_encoders.wrapper import NestedCVWrapper
        from category_encoders.cat_boost import CatBoostEncoder
    except ImportError:
        raise ImportError("Please install category_encoders: pip install category_encoders")
    
    encoder = NestedCVWrapper(feature_encoder=CatBoostEncoder(min_samples_leaf=min_samples_leaf, smoothing=smoothing), cv=cv)
    encoder.fit(df[white_col], df[target_col])
    encoded = encoder.transform(df[white_col])
    return encoded

# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example for IPW weights:
    y_true = [1, 0, 1, 0]
    y_pred = [0.8, 0.3, 0.6, 0.4]
    weights = calculate_ipw_weights(y_true, y_pred)
    print("IPW weights:", weights)
    
    # Example for computing white prior encoding:
    df_dummy = pd.DataFrame({
        "white_card_text": ["joke1", "joke2", "joke1", "joke3"],
        "pick_ratio": [0.1, 0.2, 0.1, 0.05]
    })
    prior = compute_white_prior(df_dummy)
    print("\nWhite card prior encoding:")
    print(prior.head())
