import os
import joblib
import torch
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_utils import predict_with_model
from xai_utils import get_shap_explanation, get_lime_explanation
from lstm_pytorch_utils import LSTMNet
import logging
from pandas.api.types import is_numeric_dtype

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Load models at startup
def load_models():
    """
    Load LightGBM and LSTM models from disk. Raise error if LightGBM not found.
    """
    models = {}
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    lgb_path = os.path.join(models_dir, 'lightgbm_model.pkl')
    if not os.path.exists(lgb_path):
        raise FileNotFoundError(f"Missing LightGBM model at {lgb_path}")
    models['lightgbm'] = joblib.load(lgb_path)
    # Attempt to load a full LSTM nn.Module; state_dicts are ignored
    models['lstm'] = None
    lstm_path = os.path.join(models_dir, 'lstm_model.pt')
    if os.path.exists(lstm_path):
        try:
            obj = torch.load(lstm_path, map_location='cpu')
            if hasattr(obj, 'forward'):
                obj.eval()
                models['lstm'] = obj
            else:
                print("Ignoring LSTM state_dict, not a full nn.Module.")
        except Exception as e:
            print(f"Could not load LSTM model: {e}")
    return models

MODELS = load_models()
PREDICTION_HISTORY = []

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Request JSON: { ...features..., "model": "xgboost"|"lightgbm"|"ensemble" }
    Returns: { "predictions": [...] }
    """
    # Select LightGBM for classification
    model = MODELS['lightgbm']
    if 'file' in request.files:
        file = request.files['file']
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Invalid CSV upload: {e}'}), 400
    else:
        payload = request.get_json(force=True)
        df = pd.DataFrame([payload])
    # Validate model and input
    if df.empty:
        return jsonify({'error': 'No input data provided.'}), 400
    # Align features
    expected = getattr(model, 'feature_names_in_', None)
    if expected is not None:
        missing = [f for f in expected if f not in df.columns]
        extra = [f for f in df.columns if f not in expected]
        if missing:
            return jsonify({'error': f'Missing features for model: {missing}'}), 400
        if extra:
            df = df[[f for f in expected]]  # drop extras, enforce order
    # Validate numeric types
    invalid = [f for f in df.columns if not is_numeric_dtype(df[f])]
    if invalid:
        logging.warning(f"Non-numeric features: {invalid}")
        return jsonify({'error': f'Non-numeric features: {invalid}'}), 400
    try:
        preds = predict_with_model(model, df)
        PREDICTION_HISTORY.append({'input': df.to_dict(), 'prediction': preds.tolist()})
        return jsonify({'predictions': preds.tolist()})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/predict/timeseries', methods=['POST'])
def predict_timeseries():
    """
    POST /predict/timeseries
    Request JSON: { "series": [ ... ], "seq_length": 10 }
    Returns: { "prediction": ... }
    """
    data = request.get_json(force=True)
    series = data.get('series')
    # Validate series list and numeric elements
    if not isinstance(series, (list, tuple)):
        logging.warning('Invalid "series" type')
        return jsonify({'error': '"series" must be a list of numbers.'}), 400
    if len(series) == 0:
        logging.warning('Empty "series"')
        return jsonify({'error': '"series" must contain at least one element.'}), 400
    if any(not isinstance(x, (int, float)) for x in series):
        logging.warning('Non-numeric elements in "series"')
        return jsonify({'error': 'All "series" elements must be numeric.'}), 400
    # Naive forecast: return the last value
    try:
        pred = float(series[-1])
    except Exception as e:
        return jsonify({'error': f'Forecasting error: {e}'}), 500
    return jsonify({'prediction': pred})

@app.route('/history', methods=['GET'])
def history():
    # Return the last 100 predictions for demo
    return jsonify(PREDICTION_HISTORY[-100:])

@app.route('/explain', methods=['POST'])
def explain():
    payload = request.get_json(force=True)
    # Validate input structure
    if 'input' not in payload or not isinstance(payload['input'], dict):
        logging.warning('Missing or invalid "input" for explain')
        return jsonify({'error': '"input" must be a JSON object of features.'}), 400
    model = MODELS['lightgbm']
    expected = getattr(model, 'feature_names_in_', None)
    if expected is not None:
        missing = [f for f in expected if f not in payload['input']]
        if missing:
            logging.warning(f"Missing explanation features: {missing}")
            return jsonify({'error': f'Missing features for explanation: {missing}'}), 400
    df_input = pd.DataFrame([payload['input']])
    # Ensure numeric features
    non_numeric = [c for c in df_input.columns if not is_numeric_dtype(df_input[c])]
    if non_numeric:
        logging.warning(f"Non-numeric explain features: {non_numeric}")
        return jsonify({'error': f'Non-numeric features for explanation: {non_numeric}'}), 400
    try:
        method = payload.get('method', 'shap')
        if method == 'shap':
            shap_values = get_shap_explanation(model, df_input)
            return jsonify({'shap_values': np.array(shap_values).tolist()})
        elif method == 'lime':
            lime_values = get_lime_explanation(model, df_input, df_input.columns.tolist())
            return jsonify({'lime_values': lime_values})
        else:
            return jsonify({'error': 'Unknown explanation method'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
