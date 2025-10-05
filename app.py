from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    school_id = data['school_id']
    df = pd.DataFrame(data['data'])
    target_col = data.get('target_col', 'DropoutRisk')
    feature_cols = [c for c in df.columns if c != target_col]
    model = LogisticRegression(max_iter=1000)
    model.fit(df[feature_cols], df[target_col])
    joblib.dump({"model": model, "columns": feature_cols}, f"dropout_model_{school_id}.pkl")
    return jsonify({"status": "success"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    school_id = data['school_id']
    model_path = f"dropout_model_{school_id}.pkl"
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found for this school"}), 404
    saved = joblib.load(model_path)
    model = saved["model"]
    feature_cols = saved["columns"]
    df = pd.DataFrame(data['student_data'])
    preds = model.predict(df[feature_cols])
    probs = model.predict_proba(df[feature_cols])[:, 1] * 100
    return jsonify({"predictions": preds.tolist(), "probabilities": probs.tolist()})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)