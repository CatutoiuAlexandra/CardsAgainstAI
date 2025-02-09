#!/usr/bin/env python3
"""
Module: sentiment_analysis.py

This module provides functions to load a sentiment classification pipeline
and apply it to a pandas DataFrame containing text data.

Functions:
  - load_sentiment_classifier: Loads a HuggingFace sentiment classification pipeline.
  - classify_dataframe_text: Applies the sentiment classifier to each text entry in a DataFrame,
      adding columns for the full sentiment output, the predicted label, and its score.

Usage:
  Run this module directly to see a demonstration with dummy data.
"""

from transformers import pipeline
import torch

# ---------------------------------------------------------------------------
# Sentiment Classifier Loading
# ---------------------------------------------------------------------------
def load_sentiment_classifier(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Loads a sentiment classification pipeline from HuggingFace.
    
    Parameters:
      - model_name (str): The name of the model to use.
      
    Returns:
      - A transformers pipeline for text classification.
    """
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, device=device)
    return classifier

# ---------------------------------------------------------------------------
# DataFrame Sentiment Classification
# ---------------------------------------------------------------------------
def classify_dataframe_text(df, text_column="text", classifier=None):
    """
    Applies the sentiment classifier to each text entry in the DataFrame.
    
    Parameters:
      - df (pd.DataFrame): DataFrame containing text data.
      - text_column (str): Column name containing the text.
      - classifier: A transformers pipeline object; if None, the default classifier is loaded.
      
    Returns:
      - A DataFrame with new columns 'sentiment', 'label', and 'score'.
    """
    if classifier is None:
        classifier = load_sentiment_classifier()
    
    df = df.copy()
    # Apply the classifier to each text entry (this may be slow if not batched)
    df["sentiment"] = df[text_column].apply(lambda s: classifier(s))
    # Extract the label and score from the sentiment result
    df["label"] = df["sentiment"].apply(lambda s: s[0]["label"] if isinstance(s, list) and s else None)
    df["score"] = df["sentiment"].apply(lambda s: s[0]["score"] if isinstance(s, list) and s else None)
    return df

# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    # Create a dummy DataFrame with a text column
    df_dummy = pd.DataFrame({
        "text": [
            "I love this movie!",
            "This is the worst film I've ever seen.",
            "Absolutely fantastic experience.",
            "Not good. I hated it."
        ]
    })
    
    # Load the sentiment classifier
    classifier = load_sentiment_classifier()
    
    # Apply sentiment analysis to the DataFrame
    df_result = classify_dataframe_text(df_dummy, text_column="text", classifier=classifier)
    print("Sentiment Analysis Results:")
    print(df_result)
