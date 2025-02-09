#!/usr/bin/env python3
"""
Module: tsdae_pretraining.py

This module pretrains a SentenceTransformer model using the TSDAE (Denoising AutoEncoder)
objective. It includes functions to load and filter text data from a CSV file and to 
pretrain the model using the denoising autoencoder loss.

Functions:
  - load_text_data(file_path, min_words, min_length):
      Loads text data from a CSV (or CSV.gz) file and filters out rows that are too short.
  - pretrain_tsdae_model(text_series, model_name, output_path, batch_size, epochs, lr, warmup_steps, use_amp):
      Pretrains a SentenceTransformer model using the TSDAE objective on the provided text series.

Usage:
  Adjust the global parameters in main() as needed and run the module.
"""

import os
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, datasets, losses
import nltk

# Download necessary NLTK data (if not already available)
nltk.download("punkt")

# ---------------------------------------------------------------------------
# Data Loading and Filtering
# ---------------------------------------------------------------------------
def load_text_data(file_path: str, min_words: int = 5, min_length: int = 15) -> pd.Series:
    """
    Loads text data from a CSV (or CSV.gz) file and filters out rows that are too short.
    
    Assumes the CSV file has a single column containing text. If the header is not 'text',
    the first column is renamed to 'text'.
    
    Parameters:
        file_path (str): Path to the CSV file.
        min_words (int): Minimum number of words required.
        min_length (int): Minimum number of characters required.
    
    Returns:
        pd.Series: A series of filtered text strings.
    """
    df = pd.read_csv(file_path).drop_duplicates()
    if df.columns[0] != "text":
        df.rename(columns={df.columns[0]: "text"}, inplace=True)
    filtered = df.loc[
        (df["text"].str.split().str.len() >= min_words) &
        (df["text"].str.len() >= min_length)
    ]["text"].drop_duplicates()
    return filtered

# ---------------------------------------------------------------------------
# TSDAE Pretraining Function
# ---------------------------------------------------------------------------
def pretrain_tsdae_model(
    text_series: pd.Series,
    model_name: str = "all-MiniLM-L12-v2",
    output_path: str = "output/cah_tsdae-model",
    batch_size: int = 40,
    epochs: int = 7,
    lr: float = 6e-5,
    warmup_steps: int = 2000,
    use_amp: bool = True,
) -> SentenceTransformer:
    """
    Pretrains a SentenceTransformer model using the TSDAE (Denoising AutoEncoder) objective.
    
    Parameters:
        text_series (pd.Series): Series containing text data for pretraining.
        model_name (str): Pretrained model name to use as a backbone (and as a decoder).
        output_path (str): Directory to save the trained model.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        warmup_steps (int): Number of warmup steps.
        use_amp (bool): Whether to use automatic mixed precision.
    
    Returns:
        SentenceTransformer: The trained model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model = SentenceTransformer(model_name, device=device)
    logging.info(f"Loaded model '{model_name}' with embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Create the denoising autoencoder dataset from the text series
    train_sentences = text_series.tolist()
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define the TSDAE loss with tied encoder-decoder
    train_loss = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=model_name,
        tie_encoder_decoder=True
    )
    
    logging.info("Starting TSDAE pretraining...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': lr},
        weight_decay=0,
        scheduler='constantlr',
        show_progress_bar=True,
        use_amp=use_amp,
        checkpoint_save_total_limit=5,
        output_path=output_path,
    )
    
    # Save the final model
    model.save(output_path)
    logging.info(f"Model saved to: {output_path}")
    return model

# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------
def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Global parameters (adjust these as needed)
    TEXT_CSV = "df_text.csv.gz"         # Path to the CSV file with text data
    MODEL_NAME = "all-MiniLM-L12-v2"      # Pretrained model name
    OUTPUT_PATH = "output/cah_tsdae-model"  # Directory to save the pretrained model
    BATCH_SIZE = 40
    EPOCHS = 7
    LR = 6e-5
    WARMUP_STEPS = 2000
    MIN_WORDS = 5
    MIN_LENGTH = 15

    logging.info(f"Loading text data from {TEXT_CSV}...")
    text_series = load_text_data(TEXT_CSV, min_words=MIN_WORDS, min_length=MIN_LENGTH)
    logging.info(f"Loaded {len(text_series)} text entries after filtering.")
    
    # Pretrain the model using the TSDAE objective
    model = pretrain_tsdae_model(
        text_series,
        model_name=MODEL_NAME,
        output_path=OUTPUT_PATH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        warmup_steps=WARMUP_STEPS,
        use_amp=True,
    )

if __name__ == "__main__":
    main()
