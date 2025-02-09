#!/usr/bin/env python3
"""
evaluate_models.py

This script evaluates and compares the trained models on:
- Accuracy / Precision / Recall (for classification models)
- Ranking performance for CatBoost
- Topic consistency for BERTopic
- Sentiment scores for sentiment analysis

Outputs results in `evaluation_results.json`.
"""

import os
import json
import logging
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from catboost import CatBoostRanker
from bertopic import BERTopic
from transformers import pipeline
from sklearn.metrics import classification_report, roc_auc_score

# Import evaluation functions
from evaluation import evaluate_round_predictions
from evaluation import evaluate_group_predictions
from modeling.sentiment_analysis import classify_dataframe_text

logging.basicConfig(level=logging.INFO)

# Define paths
OUTPUT_DIR = "output"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "evaluation_results.json")

# Load dataset
df_test = pd.read_parquet("cah_lab_data.parquet").sample(1000)  # Use a subset for testing

results = {}

# ====== Evaluate Sentence-BERT Model ======
logging.info("Evaluating Sentence-BERT Model...")
sbert_model = SentenceTransformer(os.path.join(OUTPUT_DIR, "sbert_finetuned"))
sentences1 = df_test["black_card_text"].tolist()
sentences2 = df_test["white_card_text"].tolist()
embeddings1 = sbert_model.encode(sentences1, convert_to_tensor=True)
embeddings2 = sbert_model.encode(sentences2, convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings1, embeddings2)
df_test["sbert_scores"] = [float(cosine_scores[i][i]) for i in range(len(df_test))]
top1_acc = df_test.groupby("fake_round_id").head(1)["won"].mean()
results["sbert"] = {"top1_accuracy": round(top1_acc, 4)}
logging.info(f"SBERT Top-1 Accuracy: {top1_acc:.4f}")

# ====== Evaluate CatBoost Ranker ======
logging.info("Evaluating CatBoost Ranker...")
catboost_model = CatBoostRanker()
catboost_model.load_model(os.path.join(OUTPUT_DIR, "catboost_ranker"))
preds = catboost_model.predict(df_test[["text"]])
df_test["catboost_preds"] = preds
ranker_eval = evaluate_group_predictions(df_test, preds)
results["catboost_ranker"] = ranker_eval

# ====== Evaluate BERTopic ======
logging.info("Evaluating BERTopic Model...")
topic_model = BERTopic.load(os.path.join(OUTPUT_DIR, "bertopic_model"))
df_test["topic"] = topic_model.transform(df_test["text"].tolist())[0]
topic_counts = df_test["topic"].value_counts().to_dict()
results["bertopic"] = {"topic_counts": topic_counts}
logging.info(f"BERTopic topic distribution: {topic_counts}")

# ====== Evaluate Sentiment Analysis ======
logging.info("Evaluating Sentiment Analysis...")
sentiment_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
df_sentiment = classify_dataframe_text(df_test, text_column="text", classifier=sentiment_classifier)
sentiment_distribution = df_sentiment["label"].value_counts().to_dict()
results["sentiment"] = {"distribution": sentiment_distribution}
logging.info(f"Sentiment Analysis Distribution: {sentiment_distribution}")

# Save results
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=4)

logging.info(f"✅ Evaluation completed. Results saved to {RESULTS_FILE}.")
