import os
import joblib
import torch
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
import json
import time
from datetime import datetime
import uuid
import traceback
from functools import wraps

# Import modules
from time_series import MultivariateLSTM
from auth import auth_manager

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super(NumpyEncoder, self).default(o)

app = Flask(__name__)
# Configure CORS to allow requests from the frontend
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"], "supports_credentials": True}})

# Configure Flask to use our custom JSON encoder
from flask.json.provider import DefaultJSONProvider
class CustomJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs.setdefault("cls", NumpyEncoder)
        return super().dumps(obj, **kwargs)
    def loads(self, s, **kwargs):
        return super().loads(s, **kwargs)
app.json = CustomJSONProvider(app)

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'api.log')

handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=10)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
handler.setLevel(logging.INFO)

app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Global variables
MODELS = {}
PREDICTION_HISTORY = []
MODEL_VERSIONS = {}
API_VERSION = "1.1.0"

# Helper function to generate versioned API paths
def versioned_url(path):
    """
    Generate a versioned API URL path.

    Args:
        path: The API path without version prefix

    Returns:
        Versioned API path
    """
    major_minor = '.'.join(API_VERSION.split('.')[:2])  # Extract major.minor version (e.g., "1.1")
    return f"/api/v{major_minor}{path}"

def load_models():
    """
    Load all available models from the models directory.
    Returns a dictionary of model objects.
    """
    models = {}
    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    # Load classification models
    for model_name in ['lightgbm_model', 'xgboost_model', 'random_forest_model']:
        model_path = os.path.join(models_dir, f'{model_name}.pkl')
        if os.path.exists(model_path):
            try:
                models[model_name] = joblib.load(model_path)
                app.logger.info(f"Loaded {model_name}")

                # Get model version (creation date)
                MODEL_VERSIONS[model_name] = datetime.fromtimestamp(
                    os.path.getmtime(model_path)
                ).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                app.logger.error(f"Failed to load {model_name}: {e}")

    # Load time-series models - handle both naming conventions
    ts_model_files = []

    # Look for both _state_dict.pt and .pt files
    for f in os.listdir(models_dir):
        if f.endswith('_state_dict.pt') or (f.endswith('.pt') and 'lstm' in f.lower()):
            ts_model_files.append(f)

    # Also check for specific model names that should be time series models
    expected_ts_models = ['engine_temp_predictor', 'lstm_model']
    for expected_name in expected_ts_models:
        info_path = os.path.join(models_dir, f'{expected_name}_info.json')
        if os.path.exists(info_path):
            # Check if we have a corresponding .pt file
            possible_files = [f'{expected_name}.pt', f'{expected_name}_state_dict.pt', 'lstm_model.pt']
            for pt_file in possible_files:
                pt_path = os.path.join(models_dir, pt_file)
                if os.path.exists(pt_path) and pt_file not in ts_model_files:
                    ts_model_files.append(pt_file)
                    break

    for ts_model_file in ts_model_files:
        # Determine model name based on file name and available info files
        if ts_model_file.endswith('_state_dict.pt'):
            model_name = ts_model_file.replace('_state_dict.pt', '')
        else:
            model_name = ts_model_file.replace('.pt', '')

        # Special handling for lstm_model.pt - map it to engine_temp_predictor
        if model_name == 'lstm_model':
            # Check if we have engine_temp_predictor_info.json
            engine_info_path = os.path.join(models_dir, 'engine_temp_predictor_info.json')
            if os.path.exists(engine_info_path):
                model_name = 'engine_temp_predictor'

        model_path = os.path.join(models_dir, ts_model_file)
        info_path = os.path.join(models_dir, f'{model_name}_info.json')

        if os.path.exists(info_path):
            try:
                # Load model info
                with open(info_path, 'r') as f:
                    model_info = json.load(f)

                # Create model instance
                model = MultivariateLSTM(
                    input_size=model_info['input_size'],
                    hidden_size=model_info['hidden_size'],
                    num_layers=model_info['num_layers'],
                    output_size=model_info['output_size'],
                    dropout=model_info.get('dropout', 0.2)
                )

                # Load state dict
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                model.eval()

                # Store model and info
                models[model_name] = {
                    'model': model,
                    'info': model_info
                }

                app.logger.info(f"Loaded time-series model {model_name} from {ts_model_file}")

                # Get model version (creation date)
                MODEL_VERSIONS[model_name] = datetime.fromtimestamp(
                    os.path.getmtime(model_path)
                ).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                app.logger.error(f"Failed to load time-series model {model_name} from {ts_model_file}: {e}")
        else:
            app.logger.warning(f"Info file not found for time-series model: {info_path}")

    # Load label encoder if available
    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    if os.path.exists(encoder_path):
        try:
            models['label_encoder'] = joblib.load(encoder_path)
            app.logger.info("Loaded label encoder")
        except Exception as e:
            app.logger.error(f"Failed to load label encoder: {e}")

    # Load feature names if available
    feature_path = os.path.join(models_dir, 'feature_names.pkl')
    if os.path.exists(feature_path):
        try:
            models['feature_names'] = joblib.load(feature_path)
            app.logger.info("Loaded feature names")
        except Exception as e:
            app.logger.error(f"Failed to load feature names: {e}")

    return models

# Load models at startup
try:
    MODELS = load_models()
    app.logger.info(f"Loaded {len(MODELS)} models")
except Exception as e:
    app.logger.error(f"Error loading models: {e}")

# Authentication decorator
def require_auth(f):
    """Decorator to require authentication for protected endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get session token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401

        session_token = auth_header.split(' ')[1]

        # Verify session
        auth_result = auth_manager.verify_session(session_token)
        if not auth_result['success']:
            return jsonify({'error': auth_result['error']}), 401

        # Add user info to request context
        request.user = auth_result['user']
        return f(*args, **kwargs)

    return decorated_function

# Helper functions
def get_model_metadata():
    """
    Get metadata for all loaded models.
    """
    metadata = {}

    for model_name, model in MODELS.items():
        if model_name in ['label_encoder', 'feature_names']:
            continue

        if isinstance(model, dict) and 'model' in model:  # Time-series model
            # Convert any NumPy arrays to Python lists
            target_cols = model['info'].get('target_cols', [])
            if isinstance(target_cols, np.ndarray):
                target_cols = target_cols.tolist()

            input_size = model['info']['input_size']
            if isinstance(input_size, np.integer):
                input_size = int(input_size)

            output_size = model['info']['output_size']
            if isinstance(output_size, np.integer):
                output_size = int(output_size)

            seq_length = model['info'].get('seq_length', 10)
            if isinstance(seq_length, np.integer):
                seq_length = int(seq_length)

            metadata[model_name] = {
                'type': 'time-series',
                'version': MODEL_VERSIONS.get(model_name, 'unknown'),
                'input_size': input_size,
                'output_size': output_size,
                'target_cols': target_cols,
                'seq_length': seq_length
            }
        else:  # Classification model
            # Convert any NumPy arrays to Python lists
            n_features = getattr(model, 'n_features_in_', None)
            if isinstance(n_features, np.integer):
                n_features = int(n_features)

            n_classes = len(getattr(model, 'classes_', []))

            feature_names = getattr(model, 'feature_names_in_', [])
            if isinstance(feature_names, np.ndarray):
                feature_names = feature_names.tolist()

            metadata[model_name] = {
                'type': 'classification',
                'version': MODEL_VERSIONS.get(model_name, 'unknown'),
                'n_features': n_features,
                'n_classes': n_classes,
                'feature_names': feature_names
            }

    return metadata

def validate_input_data(data, model_name):
    """
    Validate input data against model requirements.
    """
    if model_name not in MODELS:
        return False, f"Model {model_name} not found"

    model = MODELS[model_name]

    # For time-series models
    if isinstance(model, dict) and 'model' in model:
        if 'series' not in data:
            return False, "Missing required parameter: 'series'"

        if not isinstance(data['series'], (list, tuple)):
            return False, "'series' must be a list of numbers"

        if len(data['series']) == 0:
            return False, "Empty series provided"

        # Check for non-numeric elements
        non_numeric = [i for i, x in enumerate(data['series']) if not isinstance(x, (int, float))]
        if non_numeric:
            positions = ', '.join(str(i) for i in non_numeric[:5])
            suffix = "..." if len(non_numeric) > 5 else ""
            return False, f"Non-numeric elements found in series at positions: {positions}{suffix}"

        # Check sequence length
        seq_length = data.get('seq_length', model['info'].get('seq_length', 10))
        if len(data['series']) < seq_length:
            return False, f"Series length ({len(data['series'])}) must be >= seq_length ({seq_length})"

        return True, None

    # For classification models
    else:
        if not isinstance(data, dict):
            return False, "Input must be a dictionary of features"

        # Check for required features
        expected_features = getattr(model, 'feature_names_in_', None)
        if expected_features is not None:
            missing = [f for f in expected_features if f not in data]
            if missing:
                missing_str = ', '.join(missing)
                return False, f"Missing required features: {missing_str}"

        # Check for non-numeric values
        for key, value in data.items():
            if not isinstance(value, (int, float)):
                return False, f"Feature '{key}' has non-numeric value: {value}"

        return True, None

# API endpoints
@app.route('/api/info', methods=['GET'])
@app.route(versioned_url('/info'), methods=['GET'])
def api_info():
    """
    GET /api/info or /api/v1.1/info
    Returns information about the API and available models.
    """
    try:
        return jsonify({
            'api_version': API_VERSION,
            'models': get_model_metadata(),
            'endpoints': [
                # Legacy endpoints
                {'path': '/api/info', 'method': 'GET', 'description': 'Get API information'},
                {'path': '/api/models', 'method': 'GET', 'description': 'Get available models'},
                {'path': '/api/predict', 'method': 'POST', 'description': 'Make classification prediction'},
                {'path': '/api/predict/timeseries', 'method': 'POST', 'description': 'Make time-series prediction'},
                {'path': '/api/explain', 'method': 'POST', 'description': 'Get model explanation'},
                {'path': '/api/history', 'method': 'GET', 'description': 'Get prediction history'},
                {'path': '/api/reload', 'method': 'POST', 'description': 'Reload models (admin only)'},

                # Versioned endpoints
                {'path': versioned_url('/info'), 'method': 'GET', 'description': 'Get API information (versioned)'},
                {'path': versioned_url('/models'), 'method': 'GET', 'description': 'Get available models (versioned)'},
                {'path': versioned_url('/predict'), 'method': 'POST', 'description': 'Make classification prediction (versioned)'},
                {'path': versioned_url('/predict/timeseries'), 'method': 'POST', 'description': 'Make time-series prediction (versioned)'},
                {'path': versioned_url('/explain'), 'method': 'POST', 'description': 'Get model explanation (versioned)'},
                {'path': versioned_url('/history'), 'method': 'GET', 'description': 'Get prediction history (versioned)'},
                {'path': versioned_url('/reload'), 'method': 'POST', 'description': 'Reload models (admin only) (versioned)'}
            ],
            'status': 'operational'
        })
    except Exception as e:
        app.logger.error(f"Error in /api/info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
@app.route(versioned_url('/models'), methods=['GET'])
def get_models():
    """
    GET /api/models or /api/v1.1/models
    Returns information about available models.
    """
    try:
        return jsonify({
            'models': get_model_metadata(),
            'success': True
        })
    except Exception as e:
        app.logger.error(f"Error in /api/models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
@app.route(versioned_url('/predict'), methods=['POST'])
def predict():
    """
    POST /api/predict or /api/v1.1/predict
    Request JSON: { "model": "model_name", "data": {...features...} }
    Returns: { "prediction": [...], "probability": [...], "success": true }
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Parse input JSON
        try:
            # Only log detailed request info in debug mode
            if app.debug:
                raw_data = request.get_data(as_text=True)
                app.logger.debug(f"Raw request data: {raw_data}")
                content_type = request.headers.get('Content-Type', '')
                app.logger.debug(f"Content-Type: {content_type}")

            payload = request.get_json(force=True)

            # Log a summary of the payload instead of the full content
            if app.debug:
                app.logger.debug(f"Parsed payload: {payload}")
            else:
                app.logger.info(f"Received prediction request for model: {payload.get('model', 'lightgbm_model')}")

            if not isinstance(payload, dict):
                return jsonify({
                    'error': 'Invalid JSON payload. Expected an object.',
                    'request_id': request_id
                }), 400
        except Exception as e:
            app.logger.error(f"JSON parsing error: {e}")
            return jsonify({
                'error': f'Invalid JSON payload: {e}',
                'request_id': request_id
            }), 400

        # Extract model name and data
        model_name = payload.get('model', 'lightgbm_model')
        data = payload.get('data')

        if data is None:
            return jsonify({
                'error': 'Missing required parameter: "data"',
                'request_id': request_id
            }), 400

        # Validate input data
        valid, error_msg = validate_input_data(data, model_name)
        if not valid:
            return jsonify({
                'error': error_msg,
                'request_id': request_id
            }), 400

        # The model existence check is already done in validate_input_data
        # so we don't need to check it again here

        model = MODELS[model_name]

        # Make prediction
        df = pd.DataFrame([data])

        # Predict
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(df)[0].tolist()
                prediction = model.predict(df)[0]

                # Convert prediction to class name if label encoder is available
                if 'label_encoder' in MODELS:
                    prediction = MODELS['label_encoder'].inverse_transform([prediction])[0]

                result = {
                    'prediction': prediction,
                    'probability': probabilities,
                    'success': True,
                    'model': model_name,
                    'request_id': request_id,
                    'processing_time': time.time() - start_time
                }
            else:
                prediction = model.predict(df)[0]

                result = {
                    'prediction': prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
                    'success': True,
                    'model': model_name,
                    'request_id': request_id,
                    'processing_time': time.time() - start_time
                }

            # Add to history
            PREDICTION_HISTORY.append({
                'timestamp': datetime.now().isoformat(),
                'request_id': request_id,
                'type': 'classification',
                'model': model_name,
                'input': data,
                'output': result,
                'processing_time': time.time() - start_time
            })

            return jsonify(result)
        except Exception as e:
            app.logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
            return jsonify({
                'error': f'Error making prediction: {str(e)}',
                'request_id': request_id
            }), 500

    except Exception as e:
        app.logger.error(f"Unexpected error in predict endpoint: {e}\n{traceback.format_exc()}")
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'request_id': request_id
        }), 500

@app.route('/api/predict/timeseries', methods=['POST'])
@app.route(versioned_url('/predict/timeseries'), methods=['POST'])
def predict_timeseries():
    """
    POST /api/predict/timeseries or /api/v1.1/predict/timeseries
    Request JSON: {
        "model": "model_name",
        "series": [...],
        "seq_length": <int>,
        "horizon": <int>
    }
    Returns: { "predictions": [...], "success": true }
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Parse input JSON
        try:
            payload = request.get_json(force=True)
            if not isinstance(payload, dict):
                return jsonify({'error': 'Invalid JSON payload. Expected an object.'}), 400
        except Exception as e:
            app.logger.error(f"JSON parsing error: {e}")
            return jsonify({'error': f'Invalid JSON payload: {e}'}), 400

        # Extract parameters
        model_name = payload.get('model', 'engine_temp_predictor')
        series = payload.get('series')
        seq_length = payload.get('seq_length', 10)
        horizon = payload.get('horizon', 1)

        # Validate series
        if series is None:
            return jsonify({
                'error': 'Missing required parameter: "series"',
                'request_id': request_id
            }), 400

        if not isinstance(series, (list, tuple)):
            return jsonify({
                'error': '"series" must be a list of numbers.',
                'request_id': request_id
            }), 400

        if len(series) == 0:
            return jsonify({
                'error': 'Empty series provided.',
                'request_id': request_id
            }), 400

        # Validate seq_length
        try:
            seq_length = int(seq_length)
            if seq_length <= 0:
                return jsonify({
                    'error': '"seq_length" must be a positive integer.',
                    'request_id': request_id
                }), 400
        except ValueError:
            return jsonify({
                'error': '"seq_length" must be an integer.',
                'request_id': request_id
            }), 400

        # Validate horizon
        try:
            horizon = int(horizon)
            if horizon <= 0:
                return jsonify({
                    'error': '"horizon" must be a positive integer.',
                    'request_id': request_id
                }), 400
            if horizon > 100:
                return jsonify({
                    'error': '"horizon" must be <= 100 for performance reasons.',
                    'request_id': request_id
                }), 400
        except ValueError:
            return jsonify({
                'error': '"horizon" must be an integer.',
                'request_id': request_id
            }), 400

        # Check if model exists
        if model_name not in MODELS:
            return jsonify({
                'error': f'Model {model_name} not found',
                'request_id': request_id
            }), 404

        model_data = MODELS[model_name]

        # Check if it's a time-series model
        if not isinstance(model_data, dict) or 'model' not in model_data:
            return jsonify({
                'error': f'{model_name} is not a time-series model',
                'request_id': request_id
            }), 400

        model = model_data['model']
        model_info = model_data['info']

        # Check if series is long enough
        if len(series) < seq_length:
            return jsonify({
                'error': f'Series length ({len(series)}) must be >= seq_length ({seq_length}).',
                'request_id': request_id
            }), 400

        # Load scaler
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', f'{model_name}_scaler.pkl')
        if not os.path.exists(scaler_path):
            return jsonify({
                'error': f'Scaler for model {model_name} not found',
                'request_id': request_id
            }), 500

        scaler = joblib.load(scaler_path)

        # Prepare input data
        try:
            # Check if the input is univariate (simple list of values)
            is_univariate_input = all(isinstance(x, (int, float)) for x in series)
            app.logger.info(f"Input type: {'univariate' if is_univariate_input else 'multivariate'}")

            if is_univariate_input:
                # For univariate input, we need to create a multivariate array
                # with the target feature and zeros for other features
                app.logger.info(f"Univariate input detected, target_indices: {model_info.get('target_indices', [0])}")

                # Create a multivariate array with zeros
                multivariate_series = []
                target_idx = model_info.get('target_indices', [0])[0]  # Use the first target index

                for value in series:
                    # Create a feature vector with zeros
                    feature_vector = [0.0] * model_info['input_size']
                    # Set the target feature value
                    feature_vector[target_idx] = value
                    multivariate_series.append(feature_vector)

                series_array = np.array(multivariate_series)
                app.logger.info(f"Created multivariate array with shape {series_array.shape}")
            else:
                # For multivariate input
                # Check if series is a list of lists (multivariate)
                if not all(isinstance(x, (list, tuple)) for x in series):
                    app.logger.error(f"Invalid multivariate input: expected list of lists but got {type(series[0])}")
                    return jsonify({
                        'error': 'For multivariate input, series must be a list of lists. Each inner list should contain values for all features.',
                        'request_id': request_id
                    }), 400

                # Check if each inner list has the correct number of features
                if not all(len(x) == model_info['input_size'] for x in series):
                    app.logger.error(f"Invalid multivariate input: expected {model_info['input_size']} features but got varying lengths")
                    return jsonify({
                        'error': f'Each element in series must have {model_info["input_size"]} features',
                        'request_id': request_id
                    }), 400

                # Convert to numpy array
                series_array = np.array(series)

            # Scale the data
            scaled_series = scaler.transform(series_array)

            # Create input sequence
            input_seq = scaled_series[-seq_length:]

            # Check if we have extreme temperatures that require special handling
            last_temp = series[-1]
            is_extreme_temp = last_temp > 120 or last_temp < 40

            # Calculate trend for extreme temperature handling
            if len(series) >= 2:
                temp_trend = (series[-1] - series[0]) / (len(series) - 1)
            else:
                temp_trend = 0

            # Make predictions
            predictions = []
            current_seq = input_seq.copy()

            app.logger.info(f"Starting prediction loop for {horizon} steps (extreme_temp: {is_extreme_temp}, trend: {temp_trend:.2f})")

            for step in range(horizon):
                if is_extreme_temp and abs(temp_trend) > 1:
                    # For extreme temperatures with clear trends, use trend-based prediction
                    # This ensures we don't get unrealistic predictions for dangerous temperatures
                    next_temp = last_temp + temp_trend * (step + 1) * 0.9  # Slight dampening
                    pred_orig = [next_temp]
                    app.logger.info(f"Step {step}: Using trend-based prediction: {next_temp:.2f}°C")
                else:
                    # Use LSTM for normal temperature ranges
                    # Prepare input tensor
                    input_tensor = torch.tensor(current_seq.reshape(1, seq_length, -1), dtype=torch.float32)

                    # Make prediction
                    with torch.no_grad():
                        pred_scaled = model(input_tensor).numpy()[0]

                    # Create a full feature vector for the next time step
                    next_step = np.zeros(model_info['input_size'])
                    for i, idx in enumerate(model_info.get('target_indices', range(model_info['output_size']))):
                        next_step[idx] = pred_scaled[i]

                    # For non-predicted features, use the last values
                    for i in range(model_info['input_size']):
                        if i not in model_info.get('target_indices', range(model_info['output_size'])):
                            next_step[i] = current_seq[-1, i]

                    # Update sequence for next prediction
                    current_seq = np.vstack([current_seq[1:], next_step])

                    # Inverse transform prediction
                    # Create a dummy full-feature array for inverse transform
                    dummy = np.zeros((1, model_info['input_size']))
                    for i, idx in enumerate(model_info.get('target_indices', range(model_info['output_size']))):
                        dummy[0, idx] = pred_scaled[i]

                    # Inverse transform
                    inverse_dummy = scaler.inverse_transform(dummy)

                    # Extract target values
                    pred_orig = []
                    for i, idx in enumerate(model_info.get('target_indices', range(model_info['output_size']))):
                        pred_orig.append(inverse_dummy[0, idx])

                    app.logger.info(f"Step {step}: LSTM prediction: {pred_orig[0]:.2f}°C")

                # Store prediction
                predictions.append(pred_orig)

            # Add to history
            PREDICTION_HISTORY.append({
                'timestamp': datetime.now().isoformat(),
                'request_id': request_id,
                'type': 'timeseries',
                'model': model_name,
                'input': {
                    'series': series,
                    'seq_length': seq_length,
                    'horizon': horizon
                },
                'output': predictions,
                'processing_time': time.time() - start_time
            })

            # Return response
            if horizon == 1:
                return jsonify({
                    'prediction': predictions[0],
                    'success': True,
                    'model': model_name,
                    'request_id': request_id,
                    'processing_time': time.time() - start_time
                })

            return jsonify({
                'predictions': predictions,
                'success': True,
                'model': model_name,
                'request_id': request_id,
                'processing_time': time.time() - start_time
            })

        except Exception as e:
            app.logger.error(f"Error in time-series prediction: {e}\n{traceback.format_exc()}")
            return jsonify({
                'error': f'Error in time-series prediction: {str(e)}',
                'request_id': request_id
            }), 500

    except Exception as e:
        app.logger.error(f"Unexpected error in predict_timeseries endpoint: {e}\n{traceback.format_exc()}")
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'request_id': request_id
        }), 500

@app.route('/api/history', methods=['GET'])
@app.route(versioned_url('/history'), methods=['GET'])
def history():
    """
    GET /api/history or /api/v1.1/history
    Optional query parameters:
      - limit: int (default: 100) - Number of history items to return
      - type: string (default: all) - Filter by prediction type ('classification', 'timeseries')
      - model: string (default: all) - Filter by model name
    Returns: Array of prediction history items
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', '100')
        pred_type = request.args.get('type', 'all').lower()
        model = request.args.get('model', 'all')

        # Validate limit
        try:
            limit = int(limit)
            if limit <= 0:
                return jsonify({'error': '"limit" must be a positive integer.'}), 400
            if limit > 1000:
                limit = 1000
        except ValueError:
            return jsonify({'error': '"limit" must be an integer.'}), 400

        # Validate type
        if pred_type not in ['all', 'classification', 'timeseries']:
            return jsonify({'error': '"type" must be one of: "all", "classification", "timeseries".'}), 400

        # Filter history
        filtered_history = PREDICTION_HISTORY

        if pred_type != 'all':
            filtered_history = [item for item in filtered_history if item.get('type') == pred_type]

        if model != 'all':
            filtered_history = [item for item in filtered_history if item.get('model') == model]

        # Return the requested number of items (most recent first)
        return jsonify({
            'history': filtered_history[-limit:],
            'total_count': len(filtered_history),
            'returned_count': min(limit, len(filtered_history)),
            'success': True
        })

    except Exception as e:
        app.logger.error(f"Unexpected error in history endpoint: {e}\n{traceback.format_exc()}")
        return jsonify({
            'error': f'Unexpected error: {str(e)}'
        }), 500

# Explainability endpoint removed

@app.route('/api/reload', methods=['POST'])
@app.route(versioned_url('/reload'), methods=['POST'])
def reload_models():
    """
    POST /api/reload or /api/v1.1/reload
    Reloads all models from disk.
    Requires admin authentication via API key.
    """
    request_id = str(uuid.uuid4())

    try:
        # Check for API key in headers
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            app.logger.warning("Reload models attempt without API key")
            return jsonify({
                'error': 'Authentication required. Please provide a valid API key in the X-API-Key header.',
                'request_id': request_id
            }), 401

        # In a real application, you would validate the API key against a secure store
        # For this example, we'll use a simple environment variable or hardcoded value for demonstration
        valid_api_key = os.environ.get('ADMIN_API_KEY', 'dev-api-key-for-testing-only')

        if api_key != valid_api_key:
            app.logger.warning("Reload models attempt with invalid API key")
            return jsonify({
                'error': 'Invalid API key.',
                'request_id': request_id
            }), 403

        # If authentication passes, reload the models
        global MODELS
        MODELS = load_models()

        return jsonify({
            'message': f'Successfully reloaded {len(MODELS)} models',
            'models': get_model_metadata(),
            'success': True,
            'request_id': request_id
        })

    except Exception as e:
        app.logger.error(f"Error reloading models: {e}\n{traceback.format_exc()}")
        return jsonify({
            'error': f'Error reloading models: {str(e)}',
            'request_id': request_id
        }), 500

# Authentication endpoints
@app.route('/api/auth/register', methods=['POST'])
def register():
    """
    POST /api/auth/register
    Register a new user account.
    Request JSON: { "username": "...", "email": "...", "password": "..." }
    """
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        username = payload.get('username')
        email = payload.get('email')
        password = payload.get('password')

        if not all([username, email, password]):
            return jsonify({'error': 'Username, email, and password are required'}), 400

        result = auth_manager.register_user(username, email, password)

        if result['success']:
            return jsonify({
                'success': True,
                'message': result['message'],
                'user_id': result['user_id']
            }), 201
        else:
            return jsonify({'error': result['error']}), 400

    except Exception as e:
        app.logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """
    POST /api/auth/login
    Authenticate user and create session.
    Request JSON: { "username": "...", "password": "..." }
    """
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        username = payload.get('username')
        password = payload.get('password')

        if not all([username, password]):
            return jsonify({'error': 'Username and password are required'}), 400

        result = auth_manager.login_user(username, password)

        if result['success']:
            return jsonify({
                'success': True,
                'message': result['message'],
                'session_token': result['session_token'],
                'user': result['user']
            }), 200
        else:
            return jsonify({'error': result['error']}), 401

    except Exception as e:
        app.logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/logout', methods=['POST'])
@require_auth
def logout():
    """
    POST /api/auth/logout
    Logout user and invalidate session.
    Requires Authorization header with Bearer token.
    """
    try:
        auth_header = request.headers.get('Authorization')
        session_token = auth_header.split(' ')[1]

        result = auth_manager.logout_user(session_token)

        return jsonify({
            'success': True,
            'message': result['message']
        }), 200

    except Exception as e:
        app.logger.error(f"Logout error: {e}")
        return jsonify({'error': 'Logout failed'}), 500

@app.route('/api/auth/verify', methods=['GET'])
@require_auth
def verify_session():
    """
    GET /api/auth/verify
    Verify current session and return user info.
    Requires Authorization header with Bearer token.
    """
    try:
        return jsonify({
            'success': True,
            'user': request.user
        }), 200

    except Exception as e:
        app.logger.error(f"Session verification error: {e}")
        return jsonify({'error': 'Session verification failed'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.logger.setLevel(logging.DEBUG)
    app.debug = True
    app.run(host='0.0.0.0', port=port, debug=True)
