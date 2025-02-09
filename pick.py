#!/usr/bin/env python3
# app.py

from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostClassifier, Pool
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  
model = CatBoostClassifier()
model.load_model("cah_catboost_model.cbm")  

def pick_funniest_card(model, black_card_text, white_cards):
    """
    Given a trained CatBoost model, a black card prompt, and a list of candidate white cards,
    returns the candidate white card with the highest predicted probability.
    """
    rows = []
    for wcard in white_cards:
        combined_text = black_card_text + " " + wcard
        row = {"text": combined_text, "white_card_text": wcard}
        rows.append(row)
    df_infer = pd.DataFrame(rows)
    
    pool = Pool(data=df_infer, text_features=["text", "white_card_text"])
    
    preds = model.predict(pool, prediction_type="Probability")[:, 1]
    best_idx = preds.argmax()
    return white_cards[best_idx]

@app.route("/pick", methods=["POST"])
def pick():
    data = request.get_json(force=True)
    black_card_text = data.get("black_card_text")
    white_cards = data.get("white_cards")
    
    if not black_card_text or not white_cards:
        return jsonify({"error": "Missing black_card_text or white_cards"}), 400

    funniest = pick_funniest_card(model, black_card_text, white_cards)
    return jsonify({"funniest": funniest})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
