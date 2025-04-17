# model_utils.py
# Utilities for loading models and generating predictions/explanations
import os
import joblib
import numpy as np
import shap
import lightgbm as lgb
import xgboost as xgb

def load_model(model_path):
    if model_path.endswith('.pkl'):
        return joblib.load(model_path)
    elif model_path.endswith('.txt'):
        return lgb.Booster(model_file=model_path)
    elif model_path.endswith('.json'):
        return xgb.Booster(model_file=model_path)
    else:
        raise ValueError('Unsupported model format: ' + model_path)

def predict_with_model(model, data):
    # Assumes data is a pandas DataFrame
    if hasattr(model, 'predict'):
        return model.predict(data)
    elif hasattr(model, 'predict_proba'):
        return model.predict_proba(data)
    else:
        raise ValueError('Model does not support prediction.')

def explain_with_shap(model, data, background=None):
    # Use TreeExplainer for tree models
    explainer = shap.TreeExplainer(model, data if background is None else background)
    shap_values = explainer.shap_values(data)
    return shap_values
