#!/usr/bin/env python3
"""
Module: pair_processing.py

This module provides functions for analyzing card pairs from round-level data.
It includes two main functionalities:

1. Pair-Level Metrics Computation (compute_pair_metrics):
   - Calculates the occurrence count and win rate (pair_prior) of each unique (black, white) pair.
   - Computes the overall win rate for each white card (white_prior).
   - Derives outperform metrics (both difference and ratio) comparing the pair win rate to the white card win rate.
   - Filters out pairs that occur less than a specified minimum frequency.

2. Pair Data Aggregation (aggregate_pair_data):
   - Aggregates round-level data into pair-level statistics.
   - Computes the total wins (picks), number of showings (pick_opportunities), pick_ratio,
     and a binary target (picked_binary) for each unique (white, black) pair.
   - Constructs a joint text field ("joke") from the black and white card texts.

Usage:
  Run this module directly to see example outputs.
"""

import pandas as pd

# =============================================================================
# Pair-Level Metrics Computation
# =============================================================================
def compute_pair_metrics(df: pd.DataFrame, min_pair_freq: int = 3) -> pd.DataFrame:
    """
    Computes pair-level metrics on the DataFrame and filters by a minimum pair frequency.
    
    Parameters:
      df (pd.DataFrame): DataFrame containing at least the following columns: 
                         'black_card_text', 'white_card_text', 'text', and 'won'.
      min_pair_freq (int): Minimum number of occurrences for a pair to be kept.
    
    Returns:
      pd.DataFrame: One row per unique pair with additional columns:
                    - pair_count, pair_prior, white_prior, outperform_sub, and outperform_ratio.
    """
    # Count occurrences and compute win rates for each pair (using 'text' as unique identifier)
    df["pair_count"] = df.groupby("text")["won"].transform("count")
    df["pair_prior"] = df.groupby("text")["won"].transform("mean")
    # Compute overall win rate for each white card
    df["white_prior"] = df.groupby("white_card_text")["won"].transform("mean")
    
    # Filter out pairs with fewer occurrences than the minimum frequency
    df_filtered = df[df["pair_count"] >= min_pair_freq].copy()
    
    # Compute outperform metrics
    df_filtered["outperform_sub"] = df_filtered["pair_prior"] - df_filtered["white_prior"]
    df_filtered["outperform_ratio"] = (0.1 + df_filtered["pair_prior"]) / (0.1 + df_filtered["white_prior"])
    
    # Drop duplicate rows so that each unique pair (identified by 'text') appears once
    df_metrics = df_filtered.drop_duplicates(subset=["text"])
    return df_metrics

# =============================================================================
# Pair Data Aggregation
# =============================================================================
def aggregate_pair_data(df, min_pair_freq: int = 1) -> pd.DataFrame:
    """
    Aggregates round-level data into pair-level statistics.
    
    Parameters:
      df (pd.DataFrame): The raw data with columns including:
                         - 'black_card_text'
                         - 'white_card_text'
                         - 'won' (binary: 1 if picked, 0 otherwise)
                         - (optionally) other round-level info.
      min_pair_freq (int): Minimum number of times a given pair must occur.
      
    Returns:
      pd.DataFrame: One row per unique (white, black) pair with columns:
                    - picks: Sum of wins.
                    - pick_opportunities: Number of times the pair was shown.
                    - pick_ratio: Ratio of wins to opportunities.
                    - picked_binary: Binary target (1 if picked at least once, else 0).
                    - joke: Joint text combining black and white card texts.
    """
    df = df.copy()
    # Create a joint text field ("joke") for reference
    df["joke"] = df["black_card_text"].str.strip() + " " + df["white_card_text"].str.strip()
    
    # Group by the two card texts and aggregate wins and counts
    agg_funcs = {"won": ["sum", "count"]}
    df_pairs = df.groupby(["white_card_text", "black_card_text"]).agg(agg_funcs)
    df_pairs.columns = ["picks", "pick_opportunities"]
    df_pairs = df_pairs.reset_index()
    
    # Compute additional metrics
    df_pairs["pick_ratio"] = df_pairs["picks"] / df_pairs["pick_opportunities"]
    df_pairs["picked_binary"] = df_pairs["picks"].clip(upper=1)
    
    # Add the joint text for reference
    df_pairs["joke"] = df_pairs["black_card_text"].str.strip() + " " + df_pairs["white_card_text"].str.strip()
    
    # Filter pairs based on the minimum frequency requirement
    df_pairs = df_pairs[df_pairs["pick_opportunities"] >= min_pair_freq].reset_index(drop=True)
    
    return df_pairs

# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # ---------------------------
    # Example for compute_pair_metrics
    # ---------------------------
    print("=== Pair-Level Metrics Computation Example ===")
    df_example = pd.DataFrame({
        'black_card_text': ["What makes life worth living?"] * 4,
        'white_card_text': ["Pizza", "Kittens", "Pizza", "Rain"],
        'text': ["What makes life worth living? Pizza",
                 "What makes life worth living? Kittens",
                 "What makes life worth living? Pizza",
                 "What makes life worth living? Rain"],
        'won': [1, 0, 1, 0]
    })
    metrics = compute_pair_metrics(df_example, min_pair_freq=2)
    print(metrics)
    
    # ---------------------------
    # Example for aggregate_pair_data
    # ---------------------------
    print("\n=== Pair Data Aggregation Example ===")
    # Create a dummy DataFrame simulating round-level data
    df_dummy = pd.DataFrame({
        'black_card_text': ["Prompt1", "Prompt1", "Prompt2", "Prompt2"],
        'white_card_text': ["Answer1", "Answer2", "Answer1", "Answer2"],
        'won': [1, 0, 1, 0]
    })
    df_pairs = aggregate_pair_data(df_dummy, min_pair_freq=1)
    print("Aggregated pair data shape:", df_pairs.shape)
    print(df_pairs.head())
