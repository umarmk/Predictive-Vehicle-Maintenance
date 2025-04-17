# Flask backend for Predictive Vehicle Maintenance System
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

from model_utils import load_model, predict_with_model
from xai_utils import get_shap_explanation, get_lime_explanation
import pandas as pd
import numpy as np

# Load models (replace with actual paths/names as needed)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_model.pkl')
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Could not load model from {MODEL_PATH}: {e}")

# In-memory history for demo purposes
PREDICTION_HISTORY = []

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        # Batch prediction from uploaded file
        file = request.files['file']
        df = pd.read_csv(file)
    else:
        # Single prediction from JSON
        data = request.get_json()
        df = pd.DataFrame([data])
    if model is None:
        return jsonify({'error': 'ML model not loaded'}), 500
    try:
        preds = predict_with_model(model, df)
        result = {'predictions': preds.tolist()}
        PREDICTION_HISTORY.append({'input': df.to_dict(), 'prediction': preds.tolist()})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    # Return the last 100 predictions for demo
    return jsonify(PREDICTION_HISTORY[-100:])

@app.route('/explain', methods=['POST'])
def explain():
    data = request.get_json()
    method = data.get('method', 'shap')
    input_data = pd.DataFrame([data['input']]) if 'input' in data else None
    if model is None or input_data is None:
        return jsonify({'error': 'Model or input data missing'}), 400
    try:
        if method == 'shap':
            shap_values = get_shap_explanation(model, input_data)
            return jsonify({'shap_values': np.array(shap_values).tolist()})
        elif method == 'lime':
            lime_values = get_lime_explanation(model, input_data, input_data.columns.tolist())
            return jsonify({'lime_values': lime_values})
        else:
            return jsonify({'error': 'Unknown explanation method'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
