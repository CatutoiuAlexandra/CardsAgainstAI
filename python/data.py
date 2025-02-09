#!/usr/bin/env python3
"""
CAH Processing Module

This file contains functions for:
  - Loading data (from CSV or Parquet)
  - Preprocessing card text data
  - Combining card texts into a single string
  - Detecting near-duplicate texts using fuzzy matching
  - Creating training examples for supervised finetuning
  - Generating a pretraining corpus

Each section is marked clearly below.
"""

# =============================================================================
# Data Loading
# =============================================================================
import pandas as pd

def load_data(data_path: str,
              load_csv: bool = False,
              parquet_filename: str = "cah_lab_data.parquet",
              drop_pick2: bool = True,
              drop_skipped: bool = True) -> pd.DataFrame:
    """
    Loads the data from CSV (raw) or from a saved Parquet file.
    Optionally drops pick-2 rows or rounds that were skipped.
    
    Parameters:
      data_path (str): Path to the CSV file (if load_csv is True).
      load_csv (bool): If True, load from CSV; otherwise, load from Parquet.
      parquet_filename (str): The filename for the Parquet file.
      drop_pick2 (bool): If True, only keep rows with black_card_pick_num == 1.
      drop_skipped (bool): If True, drop rows where round_skipped is True.
      
    Returns:
      pd.DataFrame: The loaded and filtered DataFrame.
    """
    if load_csv:
        df = pd.read_csv(data_path, engine="python", on_bad_lines="skip")
        # Clean up newlines and tabs in card texts
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

# =============================================================================
# Preprocessing
# =============================================================================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and normalizes the card text data:
      - Normalizes black card blanks.
      - Removes trailing punctuation from white cards.
      - Creates a combined 'text' field.
    
    Parameters:
      df (pd.DataFrame): The DataFrame with raw card texts.
    
    Returns:
      pd.DataFrame: The processed DataFrame.
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

# =============================================================================
# Card Text Combining
# =============================================================================
import re

def combine_card_text(black_text: str, white_text: str) -> str:
    """
    Combines a black card (prompt) with a white card (punchline).
    
    If the black text contains a blank indicator (two or more underscores),
    replaces the first occurrence with the white text; otherwise, appends
    the white text to the black text.
    
    Parameters:
      black_text (str): The prompt (black card) text.
      white_text (str): The punchline (white card) text.
    
    Returns:
      str: The combined text.
    """
    if re.search(r"_{2,}", black_text):
        # Replace the first occurrence of a blank with the white text
        combined = re.sub(r"_{2,}", white_text, black_text, count=1)
    else:
        combined = f"{black_text.strip()} {white_text.strip()}"
    return combined

# Example usage for card text combining:
if __name__ == "__main__":
    prompt = "Life is all about ____."
    punchline = "chocolate"
    print("Combined text:", combine_card_text(prompt, punchline))
    
    # Test with a prompt that does not contain a blank:
    prompt2 = "Life is unpredictable."
    print("Combined text:", combine_card_text(prompt2, punchline))

# =============================================================================
# Near-Duplicate Detection
# =============================================================================
def detect_near_duplicates(df, column: str = "white_card_text", threshold: int = 90):
    """
    Uses fuzzy matching to detect near-duplicate entries in the specified column.
    
    Parameters:
      df (pd.DataFrame): DataFrame containing the text data.
      column (str): Column name to check for duplicates.
      threshold (int): Similarity threshold (0-100) for duplicates.
      
    Returns:
      List of tuples (text1, text2, similarity_score).
    """
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        print("Please install fuzzywuzzy with: pip install fuzzywuzzy[speedup]")
        return []
    
    unique_texts = df[column].unique()
    duplicates = []
    for i in range(len(unique_texts)):
        for j in range(i + 1, len(unique_texts)):
            score = fuzz.ratio(unique_texts[i], unique_texts[j])
            if score >= threshold:
                duplicates.append((unique_texts[i], unique_texts[j], score))
    return duplicates

# =============================================================================
# Training Examples Preparation
# =============================================================================
from sentence_transformers import InputExample

def create_input_examples_from_pairs(df, text_col="joke", target_col="picked_binary", format="one_col"):
    """
    Creates a list of InputExample objects for supervised finetuning.
    
    Parameters:
      df (pd.DataFrame): The pair-level DataFrame containing:
                         - A column with the joint text (default 'joke')
                         - A target column (default 'picked_binary', a binary indicator)
      text_col (str): The name of the column with the text.
      target_col (str): The name of the target column.
      format (str): Either "one_col" (single joint text) or "two_col" (if using separate texts).
    
    Returns:
      List[InputExample]: A list of InputExample objects.
    """
    examples = []
    if format == "one_col":
        for _, row in df.iterrows():
            examples.append(InputExample(texts=[row[text_col]], label=float(row[target_col])))
    elif format == "two_col":
        for _, row in df.iterrows():
            examples.append(InputExample(texts=[row["white_card_text"], row["black_card_text"]], label=float(row[target_col])))
    else:
        raise ValueError("format must be either 'one_col' or 'two_col'")
    return examples

# Dummy test for training examples preparation:
if __name__ == "__main__":
    import pandas as pd
    df_dummy = pd.DataFrame({
        "black_card_text": ["Why is the sky blue?", "What is love?"],
        "white_card_text": ["Because of the ocean.", "Baby don't hurt me."],
        "joke": ["Why is the sky blue? Because of the ocean.", "What is love? Baby don't hurt me."],
        "picked_binary": [1, 0]
    })
    examples = create_input_examples_from_pairs(df_dummy, text_col="joke", target_col="picked_binary", format="one_col")
    print("Created", len(examples), "InputExample objects.")
    print("First example:", examples[0])

# =============================================================================
# Pretraining Corpus Generation
# =============================================================================
def generate_pretraining_corpus(df_single: pd.DataFrame, df_pick2: pd.DataFrame = None) -> pd.Series:
    """
    Generates a pretraining corpus by combining unique black and white card texts,
    and (if available) combined pick-2 texts.
    
    Parameters:
      df_single (pd.DataFrame): DataFrame with at least 'black_card_text' and 'white_card_text' columns.
      df_pick2 (pd.DataFrame, optional): DataFrame with a 'text' column for pick-2 rounds.
      
    Returns:
      pd.Series: A shuffled Series of unique texts for pretraining.
    """
    corpus = pd.concat([df_single["black_card_text"], df_single["white_card_text"]]).drop_duplicates()
    if df_pick2 is not None:
        corpus = pd.concat([corpus, df_pick2["text"]]).drop_duplicates()
    return corpus.sample(frac=1, random_state=42)

# =============================================================================
# End of Module
# =============================================================================
if __name__ == "__main__":
    # You can add additional testing here if needed.
    print("Module 'cah_processing.py' loaded successfully.")
