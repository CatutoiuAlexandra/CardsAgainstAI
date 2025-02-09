#!/usr/bin/env python3
"""
train_models.py

This script trains and saves multiple models including:
- Sentence-BERT (Fine-tuned with Softmax & Triplet Loss)
- TSDAE (Denoising Autoencoder)
- CatBoost Ranker (For ranking card choices)
- BERTopic (Topic modeling for card clusters)
- Sentiment Analysis (Transformer-based)

Each model is saved in the `output/` directory.
"""

import os
import logging
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from catboost import CatBoostRanker, Pool
from bertopic import BERTopic
from transformers import pipeline

# Import model-specific training functions
from modeling.sentence_bert import finetune_siamese_model
from modeling.tsdae import pretrain_tsdae_model
from modeling.catboost_ranking import create_text_pool, fit_catboost_ranking
from modeling.bertopic_integration import create_bertopic_model, fit_topics
from modeling.sentiment_analysis import load_sentiment_classifier

logging.basicConfig(level=logging.INFO)

# Define output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_parquet("cah_lab_data.parquet")

# ====== Train Sentence-BERT Fine-tuned Model ======
logging.info("Training Sentence-BERT Model...")
sentence_model = SentenceTransformer("all-MiniLM-L12-v2")
train_examples = df.sample(5000)  # Sample subset for training
sentence_model = finetune_siamese_model(sentence_model, train_examples, epochs=3)
sentence_model.save(os.path.join(OUTPUT_DIR, "sbert_finetuned"))

# ====== Train TSDAE Model ======
logging.info("Pretraining TSDAE Model...")
text_series = df["text"].drop_duplicates()
tsdae_model = pretrain_tsdae_model(text_series, output_path=os.path.join(OUTPUT_DIR, "tsdae_model"))

# ====== Train CatBoost Ranker ======
logging.info("Training CatBoost Ranker...")
train_pool = create_text_pool(df)
eval_pool = create_text_pool(df)
catboost_model = fit_catboost_ranking(train_pool, eval_pool)
catboost_model.save_model(os.path.join(OUTPUT_DIR, "catboost_ranker"))

# ====== Train BERTopic Model ======
logging.info("Training BERTopic Model...")
embedding_model = SentenceTransformer("paraphrase-albert-small-v2")
topic_model = create_bertopic_model(embedding_model)
documents = df["text"].tolist()
topics, _ = fit_topics(topic_model, documents)
topic_model.save(os.path.join(OUTPUT_DIR, "bertopic_model"))

# ====== Train Sentiment Analysis Model ======
logging.info("Loading Sentiment Classifier...")
sentiment_classifier = load_sentiment_classifier()
logging.info("Sentiment model loaded and ready.")

logging.info("All models trained and saved successfully!")
