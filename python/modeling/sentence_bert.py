#!/usr/bin/env python3
"""
Module: sbert_finetuning.py

This module provides functions for training and finetuning SentenceTransformer models
for pair-similarity tasks in CAH applications. It includes:

1. S-BERT Training with CosineSimilarityLoss:
   - train_sentence_bert_model: Trains a Sentence-BERT model using CosineSimilarityLoss.
   - score_cards_with_sentence_transformer: Demonstrates scoring card pairs using the model.

2. Siamese Finetuning (Triplet/Cosine Loss):
   - create_input_examples: Creates InputExample objects for finetuning.
   - finetune_siamese_model: Finetunes the model using either CosineSimilarityLoss or BatchSemiHardTripletLoss.
   - evaluate_siamese_model: Evaluates the finetuned model on a development set.

3. Siamese Finetuning with Softmax Loss:
   - create_input_examples_softmax: Creates InputExample objects for finetuning using softmax loss.
   - finetune_softmax_model: Finetunes the model using SoftmaxLoss with concatenated sentence representations.

Usage:
  Run this module directly to see demonstration examples with dummy data.
"""

import math
import logging
from torch.utils.data import DataLoader
import numpy as np

# Ensure that sentence_transformers is installed.
try:
    from sentence_transformers import (
        SentenceTransformer,
        InputExample,
        SentencesDataset,
        losses,
        evaluation,
        util
    )
except ImportError:
    raise ImportError("Install sentence-transformers via pip install sentence-transformers")

# ============================================================================
# Section 1: S-BERT Training with CosineSimilarityLoss
# ============================================================================
def train_sentence_bert_model(df_train, df_dev, model_name: str = "all-MiniLM-L12-v2", 
                              num_epochs: int = 2, batch_size: int = 256, 
                              learning_rate: float = 3e-4):
    """
    Trains a Sentence-BERT model using CosineSimilarityLoss.

    Parameters:
      - df_train: Training DataFrame with columns 'white_card_text', 'black_card_text', 'won'.
      - df_dev: Development DataFrame with the same columns.
      - model_name: Name of the pretrained model.
      - num_epochs: Number of epochs.
      - batch_size: Batch size.
      - learning_rate: Learning rate.
    
    Returns:
      Trained SentenceTransformer model.
    """
    model = SentenceTransformer(model_name)

    # Map labels: 1 stays 1, 0 becomes -1.
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
    
    print("Starting Sentence-BERT training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        evaluator=dev_evaluator,
        evaluation_steps=1000,
        optimizer_params={'lr': learning_rate},
        use_amp=True,
        output_path="./output",
        save_best_model=True
    )
    return model

def score_cards_with_sentence_transformer(df, model_name: str = "all-MiniLM-L12-v2", num_samples: int = 5000):
    """
    Scores card pairs using a Sentence-BERT model.

    Parameters:
      - df: DataFrame with columns 'black_card_text', 'white_card_text', 'fake_round_id', 'won'.
      - model_name: Model name.
      - num_samples: Number of samples to score.
    
    Returns:
      DataFrame with an added 'cos_scores' column.
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
    print(f"Top-1 accuracy using cosine similarity on {model_name} embeddings: {top1_acc:.4f}")
    
    return df_sample

# ============================================================================
# Section 2: Siamese Finetuning (Triplet or Cosine Loss)
# ============================================================================
def create_input_examples(X, y, format='one_col'):
    """
    Creates a list of InputExample objects for finetuning.

    Parameters:
      - X: List of texts if format is "one_col", or list of tuples if "two_col".
      - y: List of labels (float) (e.g., 1.0 for positive, 0.0 for negative).
      - format: "one_col" or "two_col".
    
    Returns:
      List of InputExample objects.
    """
    examples = []
    if format == 'one_col':
        for text, label in zip(X, y):
            examples.append(InputExample(texts=[text], label=float(label)))
    elif format == 'two_col':
        for texts, label in zip(X, y):
            examples.append(InputExample(texts=list(texts), label=float(label)))
    else:
        raise ValueError("format must be 'one_col' or 'two_col'")
    return examples

def finetune_siamese_model(model: SentenceTransformer, train_examples, epochs=1, batch_size=64, warmup_steps=100, learning_rate=5e-5, loss_type='cosine'):
    """
    Finetunes a SentenceTransformer model using supervised examples.

    Parameters:
      - model: The SentenceTransformer model to finetune.
      - train_examples: List of InputExample objects.
      - epochs: Number of training epochs.
      - batch_size: Batch size.
      - warmup_steps: Number of warmup steps.
      - learning_rate: Learning rate.
      - loss_type: Loss to use ('cosine' for CosineSimilarityLoss or 'triplet' for BatchSemiHardTripletLoss).
    
    Returns:
      Finetuned SentenceTransformer model.
    """
    train_dataset = SentencesDataset(train_examples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
    if loss_type == 'cosine':
        train_loss = losses.CosineSimilarityLoss(model=model)
    elif loss_type == 'triplet':
        train_loss = losses.BatchSemiHardTripletLoss(model=model)
    else:
        raise ValueError("loss_type must be either 'cosine' or 'triplet'")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True,
        output_path="output/siamese_finetuned_model"
    )
    return model

def evaluate_siamese_model(model: SentenceTransformer, dev_examples, batch_size=64):
    """
    Evaluates a finetuned SentenceTransformer model on a development set.

    Parameters:
      - model: The finetuned SentenceTransformer model.
      - dev_examples: List of InputExample objects for evaluation.
      - batch_size: Batch size for evaluation.
    
    Returns:
      Evaluation score (e.g., binary classification accuracy).
    """
    texts1 = [ex.texts[0] for ex in dev_examples]
    if len(dev_examples[0].texts) > 1:
        texts2 = [ex.texts[1] for ex in dev_examples]
    else:
        texts2 = texts1
    labels = [1 if ex.label > 0.5 else 0 for ex in dev_examples]
    
    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=texts1,
        sentences2=texts2,
        labels=labels,
        batch_size=batch_size,
        show_progress_bar=True
    )
    score = evaluator(model)
    logging.info(f"Evaluation score: {score}")
    return score

# ============================================================================
# Section 3: Siamese Finetuning with Softmax Loss
# ============================================================================
def create_input_examples_softmax(texts, labels, format: str = "one_col"):
    """
    Creates a list of InputExample objects for supervised finetuning with SoftmaxLoss.

    Parameters:
      - texts: List of strings if format is "one_col"; or list of tuples if "two_col".
      - labels: List of labels (as integers, e.g. 1 for positive, 0 for negative).
      - format: "one_col" or "two_col".
    
    Returns:
      List of InputExample objects.
    """
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

def finetune_softmax_model(model: SentenceTransformer, train_examples, epochs=1, batch_size=64, warmup_steps=100, learning_rate=5e-5, num_labels=2):
    """
    Fine-tunes a SentenceTransformer using SoftmaxLoss.

    The loss is configured to use concatenated sentence representations, differences, and multiplications.

    Parameters:
      - model: The SentenceTransformer to fine-tune.
      - train_examples: List of InputExample objects.
      - epochs: Number of training epochs.
      - batch_size: Training batch size.
      - warmup_steps: Number of warmup steps.
      - learning_rate: Learning rate.
      - num_labels: Number of labels (typically 2 for binary classification).
    
    Returns:
      The fine-tuned model.
    """
    train_dataset = SentencesDataset(train_examples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=num_labels,
        concatenation_sent_rep=True,
        concatenation_sent_difference=True,
        concatenation_sent_multiplication=True
    )
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True,
        output_path="output/siamese_softmax_finetuned_model"
    )
    return model

# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # ---------------------
    # Example for S-BERT Training with CosineSimilarityLoss
    # ---------------------
    print("=== S-BERT Training Example (CosineSimilarityLoss) ===")
    import pandas as pd
    df_train = pd.DataFrame({
        "white_card_text": ["Card A", "Card B"],
        "black_card_text": ["Prompt A", "Prompt B"],
        "won": [1, 0]
    })
    df_dev = pd.DataFrame({
        "white_card_text": ["Card A", "Card B"],
        "black_card_text": ["Prompt A", "Prompt B"],
        "won": [1, 0]
    })
    
    model_sbert = train_sentence_bert_model(df_train, df_dev, num_epochs=1, batch_size=2, learning_rate=5e-5)
    df_dummy = pd.DataFrame({
        "white_card_text": ["Card A", "Card B"],
        "black_card_text": ["Prompt A", "Prompt B"],
        "fake_round_id": [1, 1],
        "won": [1, 0]
    })
    df_scored = score_cards_with_sentence_transformer(df_dummy)
    
    # ---------------------
    # Example for Siamese Finetuning (Triplet/Cosine Loss)
    # ---------------------
    print("\n=== Siamese Finetuning Example (Triplet/Cosine Loss) ===")
    texts = ["This card is hilarious!", "This card is not funny."]
    y = [1.0, 0.0]
    examples = create_input_examples(texts, y, format="one_col")
    
    model_siamese = SentenceTransformer("all-MiniLM-L12-v2")
    model_siamese = finetune_siamese_model(model_siamese, examples, epochs=1, batch_size=2, warmup_steps=1, learning_rate=5e-5, loss_type="triplet")
    score = evaluate_siamese_model(model_siamese, examples, batch_size=2)
    logging.info(f"Siamese finetuning evaluation score: {score}")
    
    # ---------------------
    # Example for Siamese Finetuning with Softmax Loss
    # ---------------------
    print("\n=== Siamese Finetuning Example (Softmax Loss) ===")
    texts_softmax = ["This card is hilarious!", "This card is not funny at all."]
    labels_softmax = [1, 0]
    examples_softmax = create_input_examples_softmax(texts_softmax, labels_softmax, format="one_col")
    
    model_softmax = SentenceTransformer("all-MiniLM-L12-v2")
    model_softmax = finetune_softmax_model(model_softmax, examples_softmax, epochs=1, batch_size=2, warmup_steps=1, learning_rate=5e-5)
    logging.info("Softmax finetuning complete.")
