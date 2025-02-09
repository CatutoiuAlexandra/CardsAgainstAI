#!/usr/bin/env python3
"""
Module: bertopic_integration.py

This module provides functions for integrating BERTopic into your workflow.
It includes functions to create a BERTopic model, fit topics on a set of documents,
generate interactive visualizations, and compute topic statistics per class.

Functions:
  - create_bertopic_model: Initializes and returns a BERTopic model using a given embedding model.
  - fit_topics: Fits the BERTopic model on a list of documents (with optional target labels).
  - visualize_bertopic: Generates interactive visualizations from a fitted BERTopic model.
  - topics_per_class: Computes topics per class statistics given documents, topic assignments, and target classes.

Usage:
  Run this module directly to see an example using a SentenceTransformer model and sample documents.
"""

import logging
from bertopic import BERTopic

# ---------------------------------------------------------------------------
# BERTopic Model Creation
# ---------------------------------------------------------------------------
def create_bertopic_model(embedding_model, 
                          min_topic_size=12, 
                          top_n_words=7, 
                          n_gram_range=(1, 2),
                          nr_topics=None, 
                          low_memory=True):
    """
    Initializes and returns a BERTopic model using the provided SentenceTransformer embedding model.
    
    Parameters:
      - embedding_model: A SentenceTransformer instance used to embed documents.
      - min_topic_size (int): Minimum number of documents required for a topic.
      - top_n_words (int): Number of representative words to show per topic.
      - n_gram_range (tuple): Tuple defining the n-gram range for candidate words.
      - nr_topics (int or None): Optionally reduce the number of topics (merges topics if set).
      - low_memory (bool): If True, uses low-memory mode.
    
    Returns:
      BERTopic: A BERTopic model instance.
    """
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        n_gram_range=n_gram_range,
        nr_topics=nr_topics,
        low_memory=low_memory,
        verbose=True
    )
    logging.info("BERTopic model created with min_topic_size=%s, top_n_words=%s, n_gram_range=%s, nr_topics=%s",
                 min_topic_size, top_n_words, n_gram_range, nr_topics)
    return topic_model

# ---------------------------------------------------------------------------
# Topic Fitting
# ---------------------------------------------------------------------------
def fit_topics(topic_model, documents, y=None):
    """
    Fits the BERTopic model on a list of documents and (optionally) target labels.
    
    Parameters:
      - topic_model: A BERTopic instance (from create_bertopic_model).
      - documents (list): A list of document strings.
      - y (list, optional): Target labels for supervised or semi-supervised topic modeling.
    
    Returns:
      tuple: (topics, probabilities) where 'topics' is a list of topic assignments and
             'probabilities' are the topic probabilities (if available).
    """
    topics, probabilities = topic_model.fit_transform(documents, y)
    logging.info("BERTopic model fitted on %d documents.", len(documents))
    return topics, probabilities

# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------
def visualize_bertopic(topic_model, top_n_topics=10, width=800, n_clusters=20):
    """
    Generates several interactive visualizations from a fitted BERTopic model.
    
    Parameters:
      - topic_model: A fitted BERTopic instance.
      - top_n_topics (int): Number of topics to display in visualizations.
      - width (int): Width (in pixels) for visualizations that accept a width parameter.
      - n_clusters (int): Number of clusters for the heatmap visualization.
      
    Returns:
      dict: A dictionary of visualization objects.
    """
    visuals = {}
    visuals["barchart"] = topic_model.visualize_barchart(top_n_topics=top_n_topics, height=700)
    visuals["term_rank"] = topic_model.visualize_term_rank()
    visuals["topics"] = topic_model.visualize_topics(top_n_topics=top_n_topics, width=width)
    visuals["hierarchy"] = topic_model.visualize_hierarchy(top_n_topics=top_n_topics, width=width)
    visuals["heatmap"] = topic_model.visualize_heatmap(n_clusters=n_clusters, top_n_topics=top_n_topics)
    logging.info("Generated BERTopic visualizations.")
    return visuals

# ---------------------------------------------------------------------------
# Topics per Class Statistics
# ---------------------------------------------------------------------------
def topics_per_class(topic_model, documents, topics, classes):
    """
    Computes topics per class given documents, their topic assignments, and target classes.
    
    Parameters:
      - topic_model: A fitted BERTopic instance.
      - documents (list): List of document strings.
      - topics (list): Topic assignments for each document.
      - classes (list): Class labels corresponding to each document.
      
    Returns:
      DataFrame: A DataFrame with topics per class statistics.
    """
    return topic_model.topics_per_class(documents, topics, classes)

# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    from sentence_transformers import SentenceTransformer

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load a SentenceTransformer model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer("paraphrase-albert-small-v2", device=device)
    
    # Sample documents (replace with your actual corpus)
    documents = [
        "This card is hilarious and makes me laugh out loud.",
        "I love this joke about pizza.",
        "This is a serious card with no humor.",
        "Not funny at all; it's quite boring."
    ]
    
    # Create a BERTopic model with specified parameters
    topic_model = create_bertopic_model(embedding_model, min_topic_size=2, top_n_words=5, nr_topics=5)
    
    # Fit topics on the documents (optionally, pass target labels as the second argument)
    topics, probabilities = fit_topics(topic_model, documents)
    print("Topic assignments:", topics)
    
    # Generate visualizations (in a notebook, these visualizations will be interactive)
    visuals = visualize_bertopic(topic_model, top_n_topics=5)
    # For example, to display the barchart visualization, use:
    visuals["barchart"].show()
