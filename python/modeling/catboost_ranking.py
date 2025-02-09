#!/usr/bin/env python3
"""
Module: catboost_ranking.py

This module provides functions for training ranking models using CatBoost.
It includes two types of models:

1. CatBoost Ranker:
   - Uses CatBoostRanker for learning-to-rank tasks.
   - Function: fit_catboost_ranker()

2. CatBoost Ranking Model:
   - Uses CatBoostClassifier in ranking mode.
   - Functions: create_text_pool() and fit_catboost_ranking()

Usage:
  Run this module directly to see a dummy example.
"""

import pandas as pd
import numpy as np
import logging
from copy import deepcopy
from catboost import CatBoostRanker, CatBoostClassifier, Pool

# ---------------------------------------------------------------------------
# CatBoost Ranker Section
# ---------------------------------------------------------------------------
DEFAULT_RANKER_PARAMS = {
    'iterations': 1000,
    'custom_metric': ["AUC", "AverageGain:top=1"],
    'verbose': 50,
    'random_seed': 0,
    "early_stopping_rounds": 40,
    'metric_period': 5,
    'task_type': 'GPU',  # Change to 'CPU' if GPU is not available
}

def fit_catboost_ranker(train_pool, test_pool, loss_function="PairLogitPairwise", additional_params=None, **kwargs):
    """
    Fits a CatBoostRanker model on the provided training and test Pools.
    
    Parameters:
      - train_pool: CatBoost Pool object for training.
      - test_pool: CatBoost Pool object for evaluation.
      - loss_function (str): Loss function to use (e.g., "PairLogitPairwise", "YetiRank").
      - additional_params (dict): Dictionary of parameters to update the default parameters.
      - kwargs: Additional keyword arguments for CatBoostRanker.
      
    Returns:
      - model: The fitted CatBoostRanker model.
    """
    params = deepcopy(DEFAULT_RANKER_PARAMS)
    params['loss_function'] = loss_function
    if additional_params:
        params.update(additional_params)
    
    model = CatBoostRanker(**params, **kwargs)
    model.fit(train_pool, eval_set=test_pool, plot=True)
    return model

# ---------------------------------------------------------------------------
# CatBoost Ranking Model Section
# ---------------------------------------------------------------------------
DEFAULT_RANKING_PARAMS = {
    'iterations': 1000,
    'verbose': 20,
    'random_seed': 0,
    "early_stopping_rounds": 5,
    'metric_period': 5,
    'task_type': 'GPU',  # Change to 'CPU' if GPU is not available.
}

def create_text_pool(df, label_col="won", group_col="fake_round_id", text_features=["text", "white_card_text"]):
    """
    Creates a CatBoost Pool object from a DataFrame.
    
    Parameters:
      - df: DataFrame that must contain the text features, a group identifier, and a label.
      - label_col (str): Name of the column with the target label.
      - group_col (str): Name of the column used for grouping (e.g. round id).
      - text_features (list): List of column names containing text features.
    
    Returns:
      - A CatBoost Pool with the specified text features and group id.
    """
    pool = Pool(
        data=df,
        label=df[label_col],
        group_id=df[group_col],
        text_features=text_features
    )
    return pool

def fit_catboost_ranking(train_pool, eval_pool, loss_function="PairLogitPairwise", additional_params=None, **kwargs):
    """
    Trains a CatBoostClassifier (used in ranking mode) on the provided Pool objects.
    
    Parameters:
      - train_pool: CatBoost Pool object for training.
      - eval_pool: CatBoost Pool object for evaluation.
      - loss_function (str): Loss function to use (e.g., "PairLogitPairwise", "YetiRank").
      - additional_params (dict): Additional parameters to update DEFAULT_RANKING_PARAMS.
      - kwargs: Additional keyword arguments for CatBoostClassifier.
    
    Returns:
      - model: The trained CatBoostClassifier model.
    """
    params = deepcopy(DEFAULT_RANKING_PARAMS)
    params['loss_function'] = loss_function
    if additional_params is not None:
        params.update(additional_params)
    
    model = CatBoostClassifier(**params, **kwargs)
    logging.info("Training CatBoost ranking model with loss_function=%s", loss_function)
    model.fit(train_pool, eval_set=eval_pool, plot=False)
    return model

# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    df_dummy = pd.DataFrame({
        'text': [
            "What makes life worth living? Pizza",
            "What makes life worth living? Kittens",
            "What makes life worth living? Pizza",
            "What makes life worth living? Rain"
        ],
        'white_card_text': ["Pizza", "Kittens", "Pizza", "Rain"],
        'fake_round_id': [1, 1, 2, 2],
        'won': [1, 0, 1, 0]
    })
    
    # Create CatBoost Pools using the dummy DataFrame
    train_pool = create_text_pool(df_dummy)
    eval_pool = create_text_pool(df_dummy)
    
    # Fit a ranking model using CatBoostClassifier in ranking mode
    model_ranking = fit_catboost_ranking(train_pool, eval_pool)
    preds_ranking = model_ranking.predict(train_pool, prediction_type="Probability")[:, 1]
    print("CatBoost ranking model predictions:", preds_ranking)
    
    # Optionally, fit a ranker model using CatBoostRanker
    model_ranker = fit_catboost_ranker(train_pool, eval_pool)
    preds_ranker = model_ranker.predict(train_pool, prediction_type="Probability")[:, 1]
    print("CatBoost ranker model predictions:", preds_ranker)
