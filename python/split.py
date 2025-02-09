#!/usr/bin/env python3
"""
Module: split.py

This module provides functions to split data into training and testing sets.

It includes:
  - split_train_test_by_round: Splits data based on the 'fake_round_id' grouping.
  - split_train_test_by_card: Splits data based on unique values in the 'white_card_text' column.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def split_train_test_by_round(df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42):
    """
    Splits the data into training and testing sets based on 'fake_round_id'.

    Parameters:
      df (pd.DataFrame): Input DataFrame containing a 'fake_round_id' column.
      test_ratio (float): Proportion of the data to assign to the test set.
      random_state (int): Random seed for reproducibility.

    Returns:
      tuple: (df_train, df_test)
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    idx1, idx2 = next(gss.split(df, groups=df["fake_round_id"].values))
    df_train = df.iloc[idx1]
    df_test  = df.iloc[idx2]
    return df_train, df_test

def split_train_test_by_card(df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42):
    """
    Splits the data into training and testing sets based on unique 'white_card_text' values.

    Parameters:
      df (pd.DataFrame): Input DataFrame containing a 'white_card_text' column.
      test_ratio (float): Proportion of unique white card texts to assign to the test set.
      random_state (int): Random seed for reproducibility.

    Returns:
      tuple: (df_train, df_test)
    """
    unique_whites = df["white_card_text"].unique()
    np.random.seed(random_state)
    test_whites = np.random.choice(unique_whites, size=int(len(unique_whites) * test_ratio), replace=False)
    df_test = df[df["white_card_text"].isin(test_whites)].copy()
    df_train = df[~df["white_card_text"].isin(test_whites)].copy()
    return df_train, df_test

# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy DataFrame for demonstration purposes
    data = {
        'fake_round_id': [1, 1, 2, 2, 3, 3],
        'white_card_text': ['CardA', 'CardB', 'CardC', 'CardD', 'CardE', 'CardF'],
        'other_column': [10, 20, 30, 40, 50, 60]
    }
    df_dummy = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df_dummy)
    
    # Test split by round
    train_round, test_round = split_train_test_by_round(df_dummy, test_ratio=0.33, random_state=42)
    print("\nSplit by round:")
    print("Train DataFrame:")
    print(train_round)
    print("Test DataFrame:")
    print(test_round)
    
    # Test split by card
    train_card, test_card = split_train_test_by_card(df_dummy, test_ratio=0.33, random_state=42)
    print("\nSplit by card:")
    print("Train DataFrame:")
    print(train_card)
    print("Test DataFrame:")
    print(test_card)
