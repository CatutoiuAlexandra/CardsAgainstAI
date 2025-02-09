#!/usr/bin/env python3
"""
Module: eda.py

This module provides functions for exploratory data analysis (EDA) and visualizations
for a CAH dataset. It includes:

  - exploratory_data_analysis: Prints summary statistics and correlations.
  - plot_card_counts: Plots histograms of card occurrence counts (for white and black cards) on a log scale.
  - plot_pick_ratio: Plots the distribution of pick ratios (win rate frequency) for white cards.

Usage:
  Run this module directly to see a demonstration with dummy data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------------------------------
def exploratory_data_analysis(df: pd.DataFrame):
    """
    Performs basic exploratory data analysis and prints summary statistics.
    
    Parameters:
      df (pd.DataFrame): The dataset to analyze.
    """
    print("=== Exploratory Data Analysis ===")
    print("Data Shape:", df.shape)
    
    if "fake_round_id" in df.columns:
        print("Unique fake_round_id:", df["fake_round_id"].nunique())
    if "black_card_text" in df.columns:
        print("Unique black cards:", df["black_card_text"].nunique())
    if "white_card_text" in df.columns:
        print("Unique white cards:", df["white_card_text"].nunique())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    cols_to_exclude = ["black_card_text", "white_card_text", "text"]
    numeric_df = df.drop(columns=cols_to_exclude, errors="ignore")
    
    if "won" in numeric_df.columns:
        print("\nCorrelation with 'won':")
        print(numeric_df.corrwith(numeric_df["won"]))
    
    if "ID_index" in df.columns:
        print("\nWin rate by position (ID_index):")
        print(df.groupby("ID_index")["won"].mean())
    
    print("=" * 40, "\n")

# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------
def plot_card_counts(df, white_col="white_card_text", black_col="black_card_text", save_path=None):
    """
    Plots histograms of card counts (for white and black cards) on a log scale.
    
    Parameters:
      df (DataFrame): The original CAH dataset.
      white_col (str): Column name for white cards.
      black_col (str): Column name for black cards.
      save_path (str): Optional path to save the figure.
    """
    counts_white = df[white_col].value_counts()
    counts_black = df[black_col].value_counts()
    
    plt.figure(figsize=(8, 6))
    ax = sns.histplot([counts_white, counts_black], bins=50)
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.legend(["Punchline Cards", "Prompt Cards"])
    plt.xlabel("Card Occurrence Count")
    plt.ylabel("Frequency (log scale)")
    plt.title("Card Count Distribution")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_pick_ratio(df_white, save_path=None):
    """
    Plots the distribution of pick ratios (win rate frequency) for white cards.
    
    Parameters:
      df_white (DataFrame): Aggregated white card data with a 'pick_ratio' column.
      save_path (str): Optional path to save the figure.
    """
    plt.figure(figsize=(8, 6))
    ax = sns.histplot(df_white["pick_ratio"], kde=True)
    plt.xlabel("Pick Ratio (win rate frequency)")
    plt.title("Punchline Win Rate Distribution")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy DataFrame for demonstration purposes
    df_dummy = pd.DataFrame({
        "fake_round_id": [1, 1, 2, 2, 3, 3],
        "black_card_text": ["prompt1", "prompt1", "prompt2", "prompt2", "prompt3", "prompt3"],
        "white_card_text": ["joke1", "joke2", "joke1", "joke3", "joke2", "joke4"],
        "text": ["prompt1 joke1", "prompt1 joke2", "prompt2 joke1", "prompt2 joke3", "prompt3 joke2", "prompt3 joke4"],
        "won": [1, 0, 1, 0, 1, 0],
        "ID_index": [1, 2, 1, 2, 1, 2]
    })
    
    # Run exploratory data analysis
    exploratory_data_analysis(df_dummy)
    
    # Plot card count distribution
    plot_card_counts(df_dummy, white_col="white_card_text", black_col="black_card_text")
    
    # Create aggregated white card data for pick ratio demonstration
    df_white = df_dummy.groupby("white_card_text").agg(
        picks=("won", "sum"),
        pick_opportunities=("won", "count")
    ).reset_index()
    df_white["pick_ratio"] = df_white["picks"] / df_white["pick_opportunities"]
    
    # Plot pick ratio distribution
    plot_pick_ratio(df_white)
