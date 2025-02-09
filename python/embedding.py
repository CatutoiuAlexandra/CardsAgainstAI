#!/usr/bin/env python3
"""
Module: embedding_processing.py

This module provides functions for:
  - Encoding texts into embeddings using a SentenceTransformer model.
  - Saving and loading embeddings to/from a Parquet file.
  - Computing similarity features (cosine similarity, dot product, L2 distance,
    mean absolute difference) between two sets of embeddings.
  - Concatenating feature arrays from multiple dictionaries into a single feature matrix.

Usage:
  Run this module directly to see a demonstration.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Embedding Encoding and I/O
# ---------------------------------------------------------------------------
def encode_texts(model: SentenceTransformer, texts, batch_size=64, normalize=False):
    """
    Encodes a list (or array) of text strings into embeddings using the provided SentenceTransformer model.
    
    Parameters:
      - model: a SentenceTransformer instance.
      - texts: list or array of text strings.
      - batch_size: Batch size for encoding.
      - normalize (bool): Whether to normalize embeddings.
    
    Returns:
      - embeddings: A NumPy array of embeddings.
    """
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=batch_size,
        normalize_embeddings=normalize
    )
    return embeddings

def save_embeddings(embeddings, texts, output_path="embeddings.parquet"):
    """
    Saves embeddings (and the corresponding texts) to a Parquet file.
    
    Parameters:
      - embeddings: A NumPy array of embeddings.
      - texts: A list or Series of texts (must have the same length as embeddings).
      - output_path: Path to save the Parquet file.
    """
    df_embed = pd.DataFrame(embeddings)
    df_embed["text"] = texts
    df_embed.set_index("text", inplace=True)
    df_embed.to_parquet(output_path)
    print(f"Embeddings saved to {output_path}")

def load_embeddings(path="embeddings.parquet"):
    """
    Loads embeddings from a Parquet file.
    
    Returns:
      - A DataFrame with embeddings (index is the text).
    """
    df_embed = pd.read_parquet(path)
    return df_embed

# ---------------------------------------------------------------------------
# Embedding Similarity Features
# ---------------------------------------------------------------------------
def compute_similarity_features(emb1: np.ndarray, emb2: np.ndarray):
    """
    Given two arrays of embeddings (shape: [n_samples, dim]), compute similarity features.
    
    Returns a dictionary with:
      - cosine_sim: Cosine similarity (scalar per sample)
      - dot_product: Dot product (scalar per sample)
      - l2_distance: Euclidean (L2) distance per sample
      - abs_diff_mean: Mean absolute difference per sample
    """
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)
    
    cosine_sim = np.array([float(util.cos_sim(emb1[i], emb2[i])[0]) for i in range(len(emb1))])
    dot_product = np.array([float(np.dot(emb1[i], emb2[i])) for i in range(len(emb1))])
    l2_distance = np.linalg.norm(emb1 - emb2, axis=1)
    abs_diff_mean = np.mean(np.abs(emb1 - emb2), axis=1)
    
    return {
        "cosine_sim": cosine_sim,
        "dot_product": dot_product,
        "l2_distance": l2_distance,
        "abs_diff_mean": abs_diff_mean
    }

def concatenate_features(*feature_dicts):
    """
    Concatenates features from one or more dictionaries into a single feature matrix.
    
    All arrays in the dictionaries must have shape (n_samples,).
    
    Returns:
      Numpy array of shape (n_samples, total_features)
    """
    features = []
    for feat_dict in feature_dicts:
        for key in sorted(feat_dict.keys()):
            col = feat_dict[key].reshape(-1, 1)
            features.append(col)
    return np.hstack(features) if features else None

# ---------------------------------------------------------------------------
# Demo / Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Create dummy embeddings and compute similarity features
    n_samples = 10
    dim = 768
    emb1 = np.random.rand(n_samples, dim)
    emb2 = np.random.rand(n_samples, dim)
    
    # Compute similarity features between corresponding embeddings
    feats = compute_similarity_features(emb1, emb2)
    concatenated = concatenate_features(feats)
    
    print("Feature keys:", feats.keys())
    print("Concatenated feature shape:", concatenated.shape)
