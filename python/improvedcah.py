#!/usr/bin/env python3
"""
Model de detectare a umorului pentru un joc de tip Cards Against Humanity folosind CatBoost

Acest script:
1. Încarcă datele de antrenare și de testare din fișiere Parquet.
2. Îmbină opțional embeddings precompute.
3. Realizează ingineria caracteristicilor adaptate pentru detectarea umorului:
   - Caracteristici de sentiment (polaritate și subiectivitate) pentru fiecare câmp de text.
   - Caracteristici de interacțiune (diferențe în lungimea textului și numărul de cuvinte).
4. Antrenează un CatBoostClassifier (cu opțiunea de a antrena și un model de ranking).
5. Evaluează modelul folosind metrici pe bază de grup (pe runde).
6. Salvează modelul antrenat.
"""

import argparse
import logging
import warnings
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

# Opțional: Pentru analiza de sentiment
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

warnings.simplefilter("ignore", category=ConvergenceWarning)

# Setări implicite globale
DEFAULT_TRAIN_PATH = "cah_train_games.parquet"
DEFAULT_TEST_PATH = "cah_test_games.parquet"
DEFAULT_EMBED_PATH = "cah_embed_L12.parquet"
DEFAULT_MODEL_SAVE_PATH = "cah_catboost_model.cbm"

# Parametri implici pentru CatBoost (setați implicit pentru GPU; se pot ajusta cu flag-urile din linia de comandă)
DEFAULT_PARAMETERS = {
    "iterations": 200,
    "max_depth": 6,
    "max_bin": 128,
    "verbose": 50,
    "random_seed": 42,
    "task_type": "GPU",  # Utilizează GPU dacă este disponibil; poate fi setat la "CPU" prin flag-ul --gpu
    "devices": "0",
    "early_stopping_rounds": 40,
    "metric_period": 5,
    "custom_metric": ["AUC"],
}

# Configurare logare
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model de detectare a umorului pentru Cards Against Humanity")
    parser.add_argument("--train-path", type=str, default=DEFAULT_TRAIN_PATH,
                        help="Calea către fișierul Parquet cu datele de antrenare")
    parser.add_argument("--test-path", type=str, default=DEFAULT_TEST_PATH,
                        help="Calea către fișierul Parquet cu datele de testare")
    parser.add_argument("--embed-path", type=str, default=DEFAULT_EMBED_PATH,
                        help="Calea către fișierul Parquet cu embeddings precompute")
    parser.add_argument("--use-embed", action="store_true",
                        help="Îmbină embeddings precompute în caracteristici")
    parser.add_argument("--use-sentiment", action="store_true",
                        help="Adaugă caracteristici de sentiment (necesită TextBlob)")
    parser.add_argument("--gpu", action="store_true",
                        help="Folosește GPU pentru antrenare (implicit GPU, altfel CPU)")
    parser.add_argument("--save-model", type=str, default=DEFAULT_MODEL_SAVE_PATH,
                        help="Calea pentru salvarea modelului antrenat")
    return parser.parse_args()


def load_data(train_path, test_path, use_text_cols, group_col):
    logger.info("Se încarcă datele de antrenare din %s", train_path)
    df_train = pd.read_parquet(train_path)
    logger.info("Se încarcă datele de testare din %s", test_path)
    df_test = pd.read_parquet(test_path)

    # Se asigură că avem coloanele necesare
    cols = use_text_cols + [group_col, "won"]
    df_train = df_train[cols].copy()
    df_test = df_test[cols].copy()

    # Convertește coloana țintă în intreg (dacă nu este deja)
    df_train["won"] = df_train["won"].astype(int)
    df_test["won"] = df_test["won"].astype(int)

    # Sortează sau selectează eșantioane pentru a asigura consistența grupurilor
    df_train = df_train.sort_values(group_col)
    df_test = df_test.sample(frac=1).sort_values(group_col)
    return df_train, df_test


def add_sentiment_features(df, text_cols):
    """
    Adaugă caracteristici de polaritate și subiectivitate a sentimentului folosind TextBlob.
    """
    if TextBlob is None:
        logger.warning("TextBlob nu este instalat; se omit caracteristicile de sentiment.")
        return df

    for col in text_cols:
        polarity_list = []
        subjectivity_list = []
        for text in df[col]:
            try:
                blob = TextBlob(text)
                polarity_list.append(blob.sentiment.polarity)
                subjectivity_list.append(blob.sentiment.subjectivity)
            except Exception:
                polarity_list.append(0.0)
                subjectivity_list.append(0.0)
        df[f"{col}_polarity"] = polarity_list
        df[f"{col}_subjectivity"] = subjectivity_list
    return df


def add_interaction_features(df, text_cols):
    """
    Adaugă caracteristici de interacțiune între câmpurile de text.
    Pentru Cards Against Humanity, interacțiunea dintre prompt (cartea neagră) și răspuns (cartea albă)
    poate indica umorul.
    """
    if "text" in text_cols and "white_card_text" in text_cols:
        # Calculează caracteristicile de bază: lungimea textului și numărul de cuvinte
        df["text_len"] = df["text"].str.len()
        df["white_card_text_len"] = df["white_card_text"].str.len()
        df["len_diff"] = abs(df["text_len"] - df["white_card_text_len"])
        df["text_word_count"] = df["text"].str.split().apply(len)
        df["white_card_word_count"] = df["white_card_text"].str.split().apply(len)
        df["word_count_diff"] = abs(df["text_word_count"] - df["white_card_word_count"])
    return df


def merge_embeddings(df, embed_path):
    """
    Îmbină embeddings precompute (se presupune că sunt indexate după 'text') cu df.
    """
    logger.info("Se îmbină embeddings din %s", embed_path)
    df_embed = pd.read_parquet(embed_path)
    df_embed.reset_index(inplace=True)  # asigură că cheia devine o coloană (ex.: 'text')
    df = df.merge(df_embed, on="text", how="left")
    return df


def prepare_features(df, use_text_cols, group_col, use_sentiment=False, use_interaction=True):
    """
    Aplică pași suplimentari de inginerie a caracteristicilor.
    """
    if use_sentiment:
        df = add_sentiment_features(df, use_text_cols)
    if use_interaction:
        df = add_interaction_features(df, use_text_cols)
    # Completează valorile lipsă în caracteristicile/embeddings compute
    df.fillna(0, inplace=True)
    return df


def build_pool(df, group_col, target_col, text_cols):
    """
    Construiește un obiect Pool pentru CatBoost din DataFrame.
    """
    # Identifică caracteristicile numerice prin parcurgerea modelelor de denumire ale caracteristicilor compute.
    numeric_feats = []
    for col in df.columns:
        if col in [group_col, target_col] + list(set(text_cols)):
            continue
        if col.replace(".", "", 1).isdigit() or any(sub in col for sub in
                                                        ["_len", "_diff", "count", "polarity", "subjectivity"]):
            numeric_feats.append(col)
    logger.info("Caracteristici text: %s", text_cols)
    logger.info("Caracteristici numerice: %s", numeric_feats)
    pool = Pool(
        data=df.drop(columns=[group_col, target_col], errors="ignore"),
        label=df[target_col],
        group_id=df[group_col],
        text_features=text_cols
    )
    return pool


def eval_preds(df_test, preds, group_col):
    """
    Evaluează predicțiile modelului folosind metrici pe bază de grup.
    """
    df2 = df_test[[group_col, "won"]].copy()
    df2["preds"] = preds

    # Sortează predicțiile descrescător în fiecare grup
    df2.sort_values([group_col, "preds"], ascending=[True, False], inplace=True)
    
    # Acuratețea top-1 pe grup
    top1_acc = df2.groupby(group_col).head(1)["won"].mean()
    logger.info("Acuratețea Top-1 pe rundă: %.4f", top1_acc)
    
    # Acuratețea top-2 și top-3
    top2_acc = df2.groupby(group_col).head(2).groupby(group_col)["won"].max().mean()
    top3_acc = df2.groupby(group_col).head(3).groupby(group_col)["won"].max().mean()
    logger.info("Acuratețea Top-2 pe rundă: %.4f", top2_acc)
    logger.info("Acuratețea Top-3 pe rundă: %.4f", top3_acc)
    
    # Generează un raport de clasificare bazat pe candidatul principal din fiecare grup
    df2["group_max"] = df2.groupby(group_col)["preds"].transform("max")
    df2["correct"] = ((df2["preds"] == df2["group_max"]) & (df2["won"] > 0)).astype(int)
    logger.info("\nRaport de clasificare (toate rândurile, 'correct' vs. 'won'):\n%s",
                classification_report(df2["won"], df2["correct"]))


def train_model(train_pool, test_pool, params, model_type="classifier"):
    """
    Antrenează fie un clasificator, fie un model de ranking.
    """
    if model_type == "classifier":
        model = CatBoostClassifier(**params)
    elif model_type == "ranker":
        params_rank = params.copy()
        params_rank["loss_function"] = "PairLogitPairwise"
        model = CatBoostRanker(**params_rank)
    else:
        raise ValueError("model_type trebuie să fie fie 'classifier' sau 'ranker'")
    
    logger.info("Se antrenează %s...", model_type)
    model.fit(train_pool, eval_set=test_pool, plot=False)
    return model


def main():
    args = parse_args()

    # Ajustează parametrii pe baza argumentelor din linia de comandă (GPU vs CPU)
    params = DEFAULT_PARAMETERS.copy()
    if args.gpu:
        params["task_type"] = "GPU"
    else:
        params["task_type"] = "CPU"

    # Definește coloanele de text și grup
    use_text_cols = ["text", "white_card_text"]  # de ex., textul cărții negre și al cărții albe
    group_col = "fake_round_id"
    target_col = "won"

    # Încarcă seturile de date de antrenare și testare
    df_train, df_test = load_data(args.train_path, args.test_path, use_text_cols, group_col)

    # Îmbină opțional embeddings precompute
    if args.use_embed:
        df_train = merge_embeddings(df_train, args.embed_path)
        df_test = merge_embeddings(df_test, args.embed_path)

    # Pregătește caracteristicile: adaugă caracteristici de sentiment și interacțiune, după caz
    df_train = prepare_features(df_train, use_text_cols, group_col,
                                use_sentiment=args.use_sentiment, use_interaction=True)
    df_test = prepare_features(df_test, use_text_cols, group_col,
                               use_sentiment=args.use_sentiment, use_interaction=True)

    # Construiește obiecte Pool pentru CatBoost pentru antrenare și testare
    train_pool = build_pool(df_train, group_col, target_col, use_text_cols)
    test_pool = build_pool(df_test, group_col, target_col, use_text_cols)

    # Antrenează clasificatorul CatBoost
    model = train_model(train_pool, test_pool, params, model_type="classifier")
    logger.info("Cea mai bună iterație: %s", model.get_best_iteration())
    logger.info("Cel mai bun scor: %s", model.get_best_score())

    # Evaluează clasificatorul folosind AUC și metrici pe bază de grup
    pred_probs = model.predict(test_pool, prediction_type="Probability")[:, 1]
    auc = roc_auc_score(df_test[target_col], pred_probs)
    logger.info("AUC pe setul de test: %.4f", auc)
    eval_preds(df_test, pred_probs, group_col)

    # (Opțional) Decomentează mai jos pentru a antrena un model de ranking pentru comparație
    """
    model_rank = train_model(train_pool, test_pool, params, model_type="ranker")
    preds_rank = model_rank.predict(test_pool)
    logger.info("AUC model de ranking: %.4f", roc_auc_score(df_test[target_col], preds_rank))
    eval_preds(df_test, preds_rank, group_col)
    """

    # Salvează modelul antrenat
    logger.info("Se salvează modelul în %s", args.save_model)
    model.save_model(args.save_model)
    logger.info("Totul este gata. Ieșire.")


if __name__ == "__main__":
    main()
