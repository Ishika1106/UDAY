from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Directory to save trained models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.json
        if not data or 'school_id' not in data or 'data' not in data:
            return jsonify({"status": "error", "message": "Missing 'school_id' or 'data' in request"}), 400

        school_id = data['school_id']
        df = pd.DataFrame(data['data'])

        if df.empty:
            return jsonify({"status": "error", "message": "Data is empty"}), 400

        target_col = data.get('target_col', 'DropoutRisk')
        if target_col not in df.columns:
            return jsonify({"status": "error", "message": f"Target column '{target_col}' not found in data"}), 400

        feature_cols = [c for c in df.columns if c != target_col]

        # Ensure features are numeric
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return jsonify({"status": "error", "message": f"Feature column '{col}' must be numeric"}), 400

        model = LogisticRegression(max_iter=1000)
        model.fit(df[feature_cols], df[target_col])

        model_path = os.path.join(MODEL_DIR, f"dropout_model_{school_id}.pkl")
        joblib.dump({"model": model, "columns": feature_cols}, model_path)

        return jsonify({"status": "success", "model_key": school_id, "features": feature_cols})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'school_id' not in data or 'student_data' not in data:
            return jsonify({"status": "error", "message": "Missing 'school_id' or 'student_data' in request"}), 400

        school_id = data['school_id']
        model_path = os.path.join(MODEL_DIR, f"dropout_model_{school_id}.pkl")
        if not os.path.exists(model_path):
            return jsonify({"status": "error", "message": "Model not found for this school"}), 404

        saved = joblib.load(model_path)
        model = saved["model"]
        feature_cols = saved["columns"]

        df = pd.DataFrame(data['student_data'])
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            return jsonify({"status": "error", "message": f"Missing feature columns in student_data: {missing_cols}"}), 400

        # Ensure features are numeric
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return jsonify({"status": "error", "message": f"Feature column '{col}' must be numeric"}), 400

        preds = model.predict(df[feature_cols])
        probs = model.predict_proba(df[feature_cols])[:, 1] * 100

        return jsonify({"predictions": preds.tolist(), "probabilities": probs.tolist()})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
