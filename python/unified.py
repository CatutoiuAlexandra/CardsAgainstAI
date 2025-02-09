#!/usr/bin/env python3
"""
Unified script for processing card data, training models, saving them, and evaluating the results.

This script includes:
  - Data loading and preprocessing functions.
  - Pick-2 round processing and text combining.
  - Exploratory data analysis (EDA) and visualization.
  - Data splitting routines.
  - Downstream classification model training (Logistic Regression and Random Forest).
  - Inverse probability weights (IPW) and target encoding.
  - Embedding feature computation and SentenceTransformer utilities.
  - TSDAE pretraining functions.
  - Sentence-BERT training and scoring.
  - BERTopic integration (if installed).
  - CatBoost ranking/classification functions.
  - Siamese finetuning routines (triplet, cosine, softmax).
  - Evaluation functions (round-level, group-level, white prior evaluation, sentiment analysis).
  
Finally, the main() function calls these routines in sequence.
"""

# ===============================
# Standard and Third‑Party Imports
# ===============================
import os
import re
import math
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from copy import deepcopy

# SentenceTransformer and related libraries
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    SentencesDataset,
    losses,
    evaluation,
    util
)

# Transformers (for sentiment analysis)
from transformers import pipeline

# CatBoost
from catboost import CatBoostRanker, CatBoostClassifier, Pool

# Optional: fuzzywuzzy for near-duplicate detection
try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None

# Optional: BERTopic integration
try:
    from bertopic import BERTopic
except ImportError:
    BERTopic = None

# Optional: category_encoders for target encoding
try:
    from category_encoders.wrapper import NestedCVWrapper
    from category_encoders.cat_boost import CatBoostEncoder
except ImportError:
    NestedCVWrapper = None
    CatBoostEncoder = None

# ===============================
# Data Loading and Preprocessing
# ===============================
def load_data(data_path: str,
              load_csv: bool = False,
              parquet_filename: str = "cah_lab_data.parquet",
              drop_pick2: bool = True,
              drop_skipped: bool = True) -> pd.DataFrame:
    """
    Loads the data from CSV (raw) or from a saved Parquet file.
    Optionally drops pick‑2 rows or rounds that were skipped.
    """
    if load_csv:
        df = pd.read_csv(data_path, engine="python", on_bad_lines="skip")
        # Clean up newlines/tabs
        df["white_card_text"] = df["white_card_text"].replace("\n", " \n ", regex=True)\
                                                    .replace("\t", " \t ", regex=True)
        df["black_card_text"] = df["black_card_text"].replace("\n", " \n ", regex=True)\
                                                    .replace("\t", " \t ", regex=True)
        # Add an index within each round
        df['ID_index'] = df.groupby('fake_round_id').cumcount() + 1
        df["won"] = df["won"].astype(bool)
        df.to_parquet(parquet_filename)
    else:
        df = pd.read_parquet(parquet_filename)
    
    df["won"] = df["won"].astype(bool)
    
    if drop_pick2:
        df = df.loc[df["black_card_pick_num"] == 1]
    if drop_skipped:
        df = df.loc[df["round_skipped"] == False]
        
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and normalizes the card text data.
      - Normalizes black card blanks.
      - Removes trailing punctuation from white cards.
      - Creates a combined 'text' field.
    """
    df["has_"] = df["black_card_text"].str.contains("_{2,}", regex=True)
    df['black_card_text'] = df['black_card_text'].str.replace("_{2,}", "__", regex=True)
    df["white_card_text"] = df["white_card_text"].str.replace(r"\.$", '', regex=True)
    
    df['text'] = df.apply(
        lambda x: x['black_card_text'].replace("__", x['white_card_text'])
                  if x['has_'] 
                  else (x['black_card_text'] + " " + x['white_card_text']),
        axis=1
    )
    df["won"] = df["won"].astype(int)
    return df

# ===============================
# Pick‑2 Processing and Card Text Combining
# ===============================
def process_pick2_cards(df: pd.DataFrame, drop_doubles: bool = True):
    """
    Processes pick‑2 (double) card rounds.
    Returns:
      - df_single: DataFrame with pick‑1 rounds.
      - df_pick2_processed: Processed pick‑2 rounds with combined text (or None if dropped).
    """
    df_single = df.loc[df["black_card_pick_num"] == 1].copy()
    df_pick2 = df.loc[df["black_card_pick_num"] > 1].copy()
    
    if drop_doubles or df_pick2.empty:
        return df_single, None

    df_pick2 = df_pick2.filter(['black_card_text', 'white_card_text', "won", "fake_round_id"]).drop_duplicates().copy()
    df_pick2["black_card_text"] = df_pick2["black_card_text"].str.replace("(PICK 2)", "", case=False, regex=False)
    df_pick2["black_card_text"] = df_pick2["black_card_text"].str.replace("__.", "__", case=False, regex=False)
    df_pick2.sort_values(['black_card_text', "won"], inplace=True, ascending=False)
    
    # Biased sampling: keep all winners and sample two non‑winners per black card & round
    df_pick2_winners = df_pick2.loc[df_pick2["won"] == True]
    try:
        df_pick2_non_winners = df_pick2.loc[df_pick2["won"] == False].groupby(
            ["black_card_text", "fake_round_id"], group_keys=False
        ).apply(lambda x: x.sample(n=min(2, len(x)), random_state=42))
    except ValueError:
        df_pick2_non_winners = pd.DataFrame()
    
    df_pick2 = pd.concat([df_pick2_winners, df_pick2_non_winners]).drop_duplicates(['black_card_text', 'white_card_text'])
    df_pick2 = df_pick2.set_index(["black_card_text", "fake_round_id"])
    df_join = df_pick2.join(df_pick2["white_card_text"], lsuffix="_x", rsuffix="_y").reset_index()
    df_join = df_join[df_join["white_card_text_x"] != df_join["white_card_text_y"]]
    df_join["pair_key"] = df_join.apply(lambda row: "_".join(sorted([row['white_card_text_x'], row['white_card_text_y']])), axis=1)
    df_join = df_join.drop_duplicates(subset=["pair_key"])
    df_join["white_card_text_x"] = df_join["white_card_text_x"].str.replace("\.$", '', regex=True)
    df_join['text'] = df_join.apply(
        lambda x: x['black_card_text'].replace("__", x['white_card_text_x'], 1)
                  .replace("__", x['white_card_text_y'], 1),
        axis=1
    )
    df_join['text'] = df_join['text'].str.replace("..", ".", regex=False)
    df_pick2_processed = df_join.drop(columns=["pair_key"], errors="ignore")
    
    return df_single, df_pick2_processed

def combine_card_text(black_text: str, white_text: str) -> str:
    """
    Combines a black card (prompt) with a white card (punchline).
    If the black text contains a blank (at least two underscores), replace the first blank with the white text.
    Otherwise, append the white text.
    """
    if re.search(r"_{2,}", black_text):
        combined = re.sub(r"_{2,}", white_text, black_text, count=1)
    else:
        combined = f"{black_text.strip()} {white_text.strip()}"
    return combined

def detect_near_duplicates(df, column: str = "white_card_text", threshold: int = 90):
    """
    Uses fuzzy matching to detect near‑duplicate entries in the specified column.
    Returns a list of tuples (text1, text2, similarity_score).
    """
    if fuzz is None:
        print("fuzzywuzzy is not installed. Please install via: pip install fuzzywuzzy[speedup]")
        return []
    
    unique_texts = df[column].unique()
    duplicates = []
    for i in range(len(unique_texts)):
        for j in range(i + 1, len(unique_texts)):
            score = fuzz.ratio(unique_texts[i], unique_texts[j])
            if score >= threshold:
                duplicates.append((unique_texts[i], unique_texts[j], score))
    return duplicates

# ===============================
# Exploratory Data Analysis (EDA) and Visualization
# ===============================
def exploratory_data_analysis(df: pd.DataFrame):
    """
    Performs basic exploratory data analysis and prints summary statistics.
    """
    print("=== Exploratory Data Analysis ===")
    print("Data Shape:", df.shape)
    print("Unique fake_round_id:", df["fake_round_id"].nunique())
    print("Unique black cards:", df["black_card_text"].nunique())
    print("Unique white cards:", df["white_card_text"].nunique())
    print("\nDescriptive Statistics:")
    print(df.describe())
    cols_to_exclude = ["black_card_text", "white_card_text", "text"]
    numeric_df = df.drop(columns=cols_to_exclude, errors="ignore")
    
    print("\nCorrelation with 'won':")
    print(numeric_df.corrwith(numeric_df["won"]))
    
    if "ID_index" in df.columns:
        print("\nWin rate by position (ID_index):")
        print(df.groupby("ID_index")["won"].mean())
    print("=" * 40, "\n")

def plot_card_counts(df, white_col="white_card_text", black_col="black_card_text", save_path=None):
    """
    Plots histograms of card counts (for white and black cards) on a log scale.
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
    """
    plt.figure(figsize=(8, 6))
    ax = sns.histplot(df_white["pick_ratio"], kde=True)
    plt.xlabel("Pick Ratio (win rate frequency)")
    plt.title("Punchline Win Rate Distribution")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# ===============================
# Downstream Classification and Basic Model Training
# ===============================
def train_logistic_regression(X, y, solver="sag"):
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver=solver, max_iter=1000))
    clf.fit(X, y)
    return clf

def train_random_forest(X, y, **kwargs):
    clf = RandomForestClassifier(**kwargs)
    clf.fit(X, y)
    return clf

def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    print("Classification Report:\n", report)
    if auc is not None:
        print("ROC AUC: {:.4f}".format(auc))
    return {"report": report, "roc_auc": auc}

def split_train_test_by_round(df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42):
    """
    Splits the data into training and testing sets based on fake_round_id.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    idx1, idx2 = next(gss.split(df, groups=df["fake_round_id"].values))
    df_train = df.iloc[idx1]
    df_test  = df.iloc[idx2]
    return df_train, df_test

def split_train_test_by_card(df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42):
    """
    Splits the data into training and testing sets based on unique white card texts.
    """
    unique_whites = df["white_card_text"].unique()
    np.random.seed(random_state)
    test_whites = np.random.choice(unique_whites, size=int(len(unique_whites) * test_ratio), replace=False)
    df_test = df[df["white_card_text"].isin(test_whites)].copy()
    df_train = df[~df["white_card_text"].isin(test_whites)].copy()
    return df_train, df_test

# ===============================
# Inverse Probability Weights and Target Encoding
# ===============================
def calculate_ipw_weights(y_true, y_pred):
    """
    Calculates inverse probability weights for a binary target.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = np.where(y_true == 1, 1.0 / y_pred, 1.0 / (1.0 - y_pred))
    return weights

def compute_white_prior(df, white_col="white_card_text", target_col="pick_ratio", cv=5, min_samples_leaf=3, smoothing=0.2):
    """
    Computes a target encoding (prior) for the white card text.
    """
    if NestedCVWrapper is None or CatBoostEncoder is None:
        raise ImportError("category_encoders is required for target encoding. Install via pip install category_encoders")
    encoder = NestedCVWrapper(feature_encoder=CatBoostEncoder(min_samples_leaf=min_samples_leaf, smoothing=smoothing), cv=cv)
    encoder.fit(df[[white_col]], df[target_col])
    encoded = encoder.transform(df[[white_col]])
    return encoded

# ===============================
# Embedding Features and Utilities
# ===============================
def compute_similarity_features(emb1: np.ndarray, emb2: np.ndarray):
    """
    Given two arrays of embeddings, compute similarity features.
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
    Concatenates features from dictionaries into a single feature matrix.
    """
    features = []
    for feat_dict in feature_dicts:
        for key in sorted(feat_dict.keys()):
            col = feat_dict[key].reshape(-1, 1)
            features.append(col)
    return np.hstack(features) if features else None

def encode_texts(model: SentenceTransformer, texts, batch_size=64, normalize=False):
    """
    Encodes texts into embeddings using SentenceTransformer.
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
    Saves embeddings and texts to a Parquet file.
    """
    df_embed = pd.DataFrame(embeddings)
    df_embed["text"] = texts
    df_embed.set_index("text", inplace=True)
    df_embed.to_parquet(output_path)
    print(f"Embeddings saved to {output_path}")

def load_embeddings(path="embeddings.parquet"):
    """
    Loads embeddings from a Parquet file.
    """
    df_embed = pd.read_parquet(path)
    return df_embed

# ===============================
# Pretraining with TSDAE
# ===============================
def load_text_data(file_path: str, min_words: int = 5, min_length: int = 15) -> pd.Series:
    """
    Loads text data from a CSV file and filters out rows that are too short.
    """
    df = pd.read_csv(file_path).drop_duplicates()
    if df.columns[0] != "text":
        df.rename(columns={df.columns[0]: "text"}, inplace=True)
    filtered = df[(df["text"].str.split().str.len() >= min_words) & 
                  (df["text"].str.len() >= min_length)]["text"].drop_duplicates()
    return filtered

def pretrain_tsdae_model(file_path: str,
                         model_name: str = "paraphrase-albert-small-v2",
                         output_path: str = "output/tsdae_model",
                         batch_size: int = 32,
                         epochs: int = 5,
                         lr: float = 6e-5,
                         warmup_steps: int = 2000) -> SentenceTransformer:
    """
    Pretrains a SentenceTransformer model using the TSDAE objective.
    """
    logging.info(f"Loading text data from {file_path}...")
    texts = load_text_data(file_path)
    logging.info(f"Loaded {len(texts)} text entries after filtering.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    logging.info(f"Using model {model_name} on device {device}.")

    from torch.utils.data import DataLoader
    from sentence_transformers import datasets, losses

    train_dataset = datasets.DenoisingAutoEncoderDataset(texts.tolist())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss = losses.DenoisingAutoEncoderLoss(model,
                                                 decoder_name_or_path=model_name,
                                                 tie_encoder_decoder=True)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': lr},
        show_progress_bar=True,
        use_amp=True,
        output_path=output_path,
    )
    model.save(output_path)
    logging.info(f"Model saved to {output_path}")
    return model

# ===============================
# Sentence‑BERT Finetuning and Training Examples
# ===============================
def create_input_examples_from_pairs(df, text_col="joke", target_col="picked_binary", format="one_col"):
    """
    Creates a list of InputExample objects for supervised finetuning.
    """
    examples = []
    if format == "one_col":
        for idx, row in df.iterrows():
            examples.append(InputExample(texts=[row[text_col]], label=float(row[target_col])))
    elif format == "two_col":
        for idx, row in df.iterrows():
            examples.append(InputExample(texts=[row["white_card_text"], row["black_card_text"]], label=float(row[target_col])))
    else:
        raise ValueError("format must be either 'one_col' or 'two_col'")
    return examples

def train_sentence_bert_model(df_train, df_dev, model_name: str = "all-MiniLM-L12-v2", 
                              num_epochs: int = 2, batch_size: int = 256, 
                              learning_rate: float = 3e-4):
    """
    Trains a Sentence‑BERT model using CosineSimilarityLoss.
    """
    model = SentenceTransformer(model_name)
    
    def map_label(x):
        return 1.0 if x == 1 else -1.0
    
    train_examples = df_train.apply(
        lambda row: InputExample(texts=[row["white_card_text"], row["black_card_text"]],
                                 label=map_label(row["won"])), axis=1
    ).tolist()
    
    dev_examples = df_dev.apply(
        lambda row: InputExample(texts=[row["white_card_text"], row["black_card_text"]],
                                 label=map_label(row["won"])), axis=1
    ).tolist()
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    dev_evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=[ex.texts[0] for ex in dev_examples],
        sentences2=[ex.texts[1] for ex in dev_examples],
        labels=[1 if ex.label == 1 else 0 for ex in dev_examples],
        batch_size=batch_size,
        show_progress_bar=True
    )
    
    train_loss = losses.CosineSimilarityLoss(model=model)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    
    print("Starting Sentence‑BERT training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        evaluator=dev_evaluator,
        evaluation_steps=1000,
        optimizer_params={'lr': learning_rate},
        use_amp=True,
        output_path="./output/sentence_bert_model",
        save_best_model=True
    )
    return model

def score_cards_with_sentence_transformer(df, model_name: str = "all-MiniLM-L12-v2", num_samples: int = 5000):
    """
    Demonstrates scoring card pairs using a Sentence‑BERT model.
    """
    model = SentenceTransformer(model_name)
    df_sample = df.head(num_samples).copy()
    
    sentences1 = df_sample["black_card_text"].tolist()
    sentences2 = df_sample["white_card_text"].tolist()
    
    embeddings1 = model.encode(sentences1, batch_size=256, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, batch_size=256, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    scores = [float(cosine_scores[i][i]) for i in range(len(df_sample))]
    df_sample["cos_scores"] = scores
    
    df_sample.sort_values(["fake_round_id", "cos_scores"], ascending=False, inplace=True)
    top1_acc = df_sample.groupby("fake_round_id").head(1)["won"].mean()
    print(f"Top‑1 accuracy using cosine similarity on {model_name} embeddings: {top1_acc:.4f}")
    
    return df_sample

# ===============================
# BERTopic Integration (Optional)
# ===============================
def create_bertopic_model(embedding_model, 
                          min_topic_size=12, 
                          top_n_words=7, 
                          n_gram_range=(1, 2),
                          nr_topics=None, 
                          low_memory=True):
    """
    Initializes and returns a BERTopic model.
    """
    if BERTopic is None:
        raise ImportError("BERTopic is not installed. Install via pip install bertopic")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        n_gram_range=n_gram_range,
        nr_topics=nr_topics,
        low_memory=low_memory,
        verbose=True
    )
    logging.info("BERTopic model created.")
    return topic_model

def fit_topics(topic_model, documents, y=None):
    """
    Fits the BERTopic model on documents.
    """
    topics, probabilities = topic_model.fit_transform(documents, y)
    logging.info("BERTopic model fitted on %d documents.", len(documents))
    return topics, probabilities

def visualize_bertopic(topic_model, top_n_topics=10, width=800, n_clusters=20):
    """
    Generates visualizations from a fitted BERTopic model.
    """
    visuals = {}
    visuals["barchart"] = topic_model.visualize_barchart(top_n_topics=top_n_topics, height=700)
    visuals["term_rank"] = topic_model.visualize_term_rank()
    visuals["topics"] = topic_model.visualize_topics(top_n_topics=top_n_topics, width=width)
    visuals["hierarchy"] = topic_model.visualize_hierarchy(top_n_topics=top_n_topics, width=width)
    visuals["heatmap"] = topic_model.visualize_heatmap(n_clusters=n_clusters, top_n_topics=top_n_topics)
    logging.info("Generated BERTopic visualizations.")
    return visuals

def topics_per_class(topic_model, documents, topics, classes):
    """
    Computes topics per class.
    """
    return topic_model.topics_per_class(documents, topics, classes)

# ===============================
# CatBoost Ranking and Classification
# ===============================
DEFAULT_CATBOOST_PARAMS = {
    'iterations': 1000,
    'verbose': 50,
    'random_seed': 0,
    "early_stopping_rounds": 40,
    'metric_period': 5,
    'task_type': 'GPU',  # Change to 'CPU' if GPU is not available
}

def fit_catboost_ranker(train_pool, test_pool, loss_function="PairLogitPairwise", additional_params=None, **kwargs):
    params = deepcopy(DEFAULT_CATBOOST_PARAMS)
    params['loss_function'] = loss_function
    if additional_params:
        params.update(additional_params)
    
    model = CatBoostRanker(**params, **kwargs)
    model.fit(train_pool, eval_set=test_pool, plot=True)
    return model

DEFAULT_CATBOOST_CLASSIFIER_PARAMS = {
    'iterations': 1000,
    'verbose': 20,
    'random_seed': 0,
    "early_stopping_rounds": 5,
    'metric_period': 5,
    'task_type': 'GPU',  # Change to 'CPU' if GPU is not available.
}

def create_text_pool(df, label_col="won", group_col="fake_round_id", text_features=["text", "white_card_text"]):
    pool = Pool(
        data=df,
        label=df[label_col],
        group_id=df[group_col],
        text_features=text_features
    )
    return pool

def fit_catboost_ranking(train_pool, eval_pool, loss_function="PairLogitPairwise", additional_params=None, **kwargs):
    params = deepcopy(DEFAULT_CATBOOST_CLASSIFIER_PARAMS)
    params['loss_function'] = loss_function
    if additional_params is not None:
        params.update(additional_params)
    
    model = CatBoostClassifier(**params, **kwargs)
    logging.info("Training CatBoost ranking model.")
    model.fit(train_pool, eval_set=eval_pool, plot=False)
    return model

# ===============================
# Siamese Finetuning
# ===============================
def create_input_examples(texts, labels, format: str = "one_col"):
    examples = []
    if format == "one_col":
        for text, label in zip(texts, labels):
            examples.append(InputExample(texts=[text], label=float(label)))
    elif format == "two_col":
        for pair, label in zip(texts, labels):
            examples.append(InputExample(texts=list(pair), label=float(label)))
    else:
        raise ValueError("format must be 'one_col' or 'two_col'")
    return examples

def finetune_siamese_model(model: SentenceTransformer,
                           train_examples,
                           epochs: int = 1,
                           batch_size: int = 64,
                           warmup_steps: int = 100,
                           learning_rate: float = 5e-5,
                           loss_type: str = "triplet") -> SentenceTransformer:
    train_dataset = SentencesDataset(train_examples, model=model)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
    if loss_type == "triplet":
        loss = losses.BatchSemiHardTripletLoss(model=model)
    elif loss_type == "cosine":
        loss = losses.CosineSimilarityLoss(model=model)
    else:
        raise ValueError("Unsupported loss_type. Use 'triplet' or 'cosine'.")
    
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True,
        output_path="output/siamese_finetuned_model"
    )
    return model

def create_input_examples_softmax(texts, labels, format: str = "one_col"):
    examples = []
    if format == "one_col":
        for text, label in zip(texts, labels):
            examples.append(InputExample(texts=[text], label=int(label)))
    elif format == "two_col":
        for pair, label in zip(texts, labels):
            examples.append(InputExample(texts=list(pair), label=int(label)))
    else:
        raise ValueError("format must be 'one_col' or 'two_col'")
    return examples

def finetune_softmax_model(model: SentenceTransformer,
                           train_examples,
                           epochs: int = 1,
                           batch_size: int = 64,
                           warmup_steps: int = 100,
                           learning_rate: float = 5e-5,
                           num_labels: int = 2) -> SentenceTransformer:
    train_dataset = SentencesDataset(train_examples, model=model)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
    loss = losses.SoftmaxLoss(model=model,
                              sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                              num_labels=num_labels,
                              concatenation_sent_rep=True,
                              concatenation_sent_difference=True,
                              concatenation_sent_multiplication=True)
    
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True,
        output_path="output/siamese_softmax_finetuned_model"
    )
    return model

# ===============================
# Round and Group Evaluation
# ===============================
def evaluate_round_predictions(df_test: pd.DataFrame, preds):
    """
    Evaluates predictions on a per‑round basis.
    """
    df_eval = df_test[["fake_round_id", "won"]].copy()
    df_eval["preds"] = preds
    if df_eval["preds"].nunique() < 2:
        raise ValueError("Predictions appear constant (or NaN). Check your inputs.")

    df_eval["m_score"] = df_eval.groupby("fake_round_id")["preds"].transform("max")
    df_eval["correct"] = ((df_eval["preds"] == df_eval["m_score"]) & (df_eval["won"] > 0)).astype(int)
    
    df_eval = df_eval.sample(frac=1)
    df_eval = df_eval.sort_values("preds", ascending=False)
    
    print("=== Full Classification Report (per row) ===")
    print(classification_report(df_eval["won"], df_eval["correct"]))
    print("\n=== Top‑1 Evaluation (one prediction per round) ===")
    top1_df = df_eval.drop_duplicates(subset=["fake_round_id"])
    print(classification_report(top1_df["won"], top1_df["correct"]))
    print("\nAlternative top‑1 accuracy (group‑level):", top1_df["won"].mean())
    print("Top‑2 accuracy by round:",
          df_eval.groupby("fake_round_id").head(2).groupby("fake_round_id")["won"].max().mean())
    print("Top‑3 accuracy by round:",
          df_eval.groupby("fake_round_id").head(3).groupby("fake_round_id")["won"].max().mean())

def evaluate_group_predictions(df, preds, group_col="fake_round_id", label_col="won"):
    """
    Evaluates predictions on a group basis.
    """
    df_eval = df[[group_col, label_col]].copy()
    df_eval["preds"] = preds
    df_eval["max_pred"] = df_eval.groupby(group_col)["preds"].transform("max")
    df_eval["selected"] = (df_eval["preds"] == df_eval["max_pred"]).astype(int)
    df_eval["correct"] = ((df_eval["selected"] == 1) & (df_eval[label_col] > 0)).astype(int)
    
    report = classification_report(df_eval[label_col], df_eval["correct"], output_dict=True)
    roc_auc = roc_auc_score(df_eval[label_col], df_eval["preds"])
    
    group_top = df_eval.drop_duplicates(subset=[group_col])
    top1_acc = group_top[label_col].mean()
    
    metrics = {
        "classification_report": report,
        "roc_auc": roc_auc,
        "top1_group_accuracy": top1_acc
    }
    
    print("=== Group‑Level Evaluation ===")
    print("Top‑1 accuracy (mean label per group):", round(top1_acc, 4))
    print("ROC‑AUC:", round(roc_auc, 4))
    print("Classification Report (per row selection):")
    print(classification_report(df_eval[label_col], df_eval["correct"]))
    
    return metrics

def compute_white_card_prior(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a win‑rate prior for each white card from training data and joins it with the test set.
    """
    df_white_prior = df_train.groupby("white_card_text", as_index=False)["won"].mean()
    df_white_prior.rename(columns={"won": "white_prior"}, inplace=True)
    df_white_prior.set_index("white_card_text", inplace=True)
    df_test = df_test.join(df_white_prior, on="white_card_text", how="left")
    global_prior = df_train["won"].mean()
    df_test["white_prior"] = df_test["white_prior"].fillna(global_prior)
    return df_test

def evaluate_by_white_prior(df_test: pd.DataFrame):
    """
    Evaluates the ranking based on white card prior (top‑1, top‑2, top‑3 accuracy).
    """
    df_sorted = df_test.sort_values(["fake_round_id", "white_prior"], ascending=False)
    top1 = df_sorted.groupby("fake_round_id").head(1)["won"].mean()
    top2 = df_sorted.groupby("fake_round_id").head(2).groupby("fake_round_id")["won"].max().mean()
    top3 = df_sorted.groupby("fake_round_id").head(3).groupby("fake_round_id")["won"].max().mean()
    
    print("White Card Prior Evaluation:")
    print(f"Top‑1 Accuracy: {top1:.4f}")
    print(f"Top‑2 Accuracy: {top2:.4f}")
    print(f"Top‑3 Accuracy: {top3:.4f}")

# ===============================
# Sentiment Analysis (Optional)
# ===============================
def load_sentiment_classifier(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Loads a sentiment classification pipeline.
    """
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, device=device)
    return classifier

def classify_dataframe_text(df, text_column="text", classifier=None):
    """
    Applies the sentiment classifier to each text entry in the DataFrame.
    """
    if classifier is None:
        classifier = load_sentiment_classifier()
    
    df = df.copy()
    df["sentiment"] = df[text_column].apply(lambda s: classifier(s))
    df["label"] = df["sentiment"].apply(lambda s: s[0]["label"] if isinstance(s, list) and s else None)
    df["score"] = df["sentiment"].apply(lambda s: s[0]["score"] if isinstance(s, list) and s else None)
    return df

# ===============================
# Main Pipeline: Process Data, Train, Save and Evaluate Models
# ===============================
def main():
    logging.basicConfig(level=logging.INFO)
    
    # === Data Paths and Settings ===
    data_path = "cah_lab_data.csv"      # update as needed (if using CSV)
    parquet_filename = "cah_lab_data.parquet"
    
    # === Load and Preprocess Data ===
    print("Loading data...")
    # Set load_csv=True if you want to load the raw CSV; otherwise, use the parquet file.
    df = load_data(data_path, load_csv=False, parquet_filename=parquet_filename)
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    # Optionally process pick‑2 cards (if needed)
    df_single, df_pick2 = process_pick2_cards(df)
    
    # === EDA and (optional) Visualization ===
    exploratory_data_analysis(df)
    # Uncomment to display plots:
    # plot_card_counts(df)
    # plot_pick_ratio( ... )  # (first aggregate white card stats if available)
    
    # === Split Data into Train and Test Sets ===
    df_train, df_test = split_train_test_by_round(df)
    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    
    # === Downstream Classification: Train Basic Models (Using Dummy Features) ===
    # For demonstration we use random features; in practice use meaningful features.
    num_features = 10
    X_train = np.random.rand(len(df_train), num_features)
    y_train = df_train["won"].values
    X_test = np.random.rand(len(df_test), num_features)
    y_test = df_test["won"].values
    
    print("Training Logistic Regression model...")
    clf_lr = train_logistic_regression(X_train, y_train)
    evaluate_classifier(clf_lr, X_test, y_test)
    
    print("Training Random Forest model...")
    clf_rf = train_random_forest(X_train, y_train, n_estimators=50, max_depth=5, random_state=42)
    evaluate_classifier(clf_rf, X_test, y_test)
    
    # === Sentence‑BERT Training and Scoring ===
    if not df_train.empty and "white_card_text" in df_train.columns and "black_card_text" in df_train.columns:
        print("Training Sentence‑BERT model (1 epoch demo)...")
        model_bert = train_sentence_bert_model(df_train, df_test, num_epochs=1, batch_size=16)
        # Score a sample of card pairs using the trained model
        df_scored = score_cards_with_sentence_transformer(df_test, model_name="all-MiniLM-L12-v2", num_samples=100)
    
    # === White Card Prior Evaluation ===
    df_test_eval = compute_white_card_prior(df_train, df_test)
    evaluate_by_white_prior(df_test_eval)
    
    # === Optional: Sentiment Analysis on Test Data ===
    print("Performing sentiment analysis on test data...")
    # (Assumes a column named "text" exists; adjust as needed.)
    df_sentiment = classify_dataframe_text(df_test, text_column="text")
    print(df_sentiment.head())
    
    # === Save Trained Models ===
    model_save_path = "output/sentence_bert_model"
    model_bert.save(model_save_path)
    print(f"Sentence‑BERT model saved to {model_save_path}")
    
    # === Additional Evaluations (e.g., round‑level or group‑level) can be added here ===
    # For example, if you have prediction scores (here we use dummy scores):
    # dummy_preds = np.random.rand(len(df_test))
    # evaluate_round_predictions(df_test, dummy_preds)
    # evaluate_group_predictions(df_test, dummy_preds)

if __name__ == "__main__":
    main()
